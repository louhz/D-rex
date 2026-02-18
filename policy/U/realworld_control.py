#!/usr/bin/env python3

import os
import sys

# Append the scripts folder to Python's search path
sys.path.append("/home/haozhe/leap_ws/src/ros2_module/scripts")

import time
import rclpy
import numpy as np
import threading
import csv
from datetime import datetime

from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

# Import or copy in your existing services
from leap_hand.srv import LeapPosition, LeapVelocity, LeapEffort, LeapPosVelEff

# If you have these utilities in a separate package, import them.
# Otherwise, copy them directly here or ensure they are on your PYTHONPATH.
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu


###############################################################################
# 1. LEAPNode - the main “hand” node (copied from your code, slightly shortened)
###############################################################################
class LeapNode(Node):
    def __init__(self,control_force=100.0):
        super().__init__('leaphand_node')
        # Some parameters to control the hand
        self.kP = self.declare_parameter('kP', 800.0).get_parameter_value().double_value
        self.kI = self.declare_parameter('kI', 0.0).get_parameter_value().double_value
        self.kD = self.declare_parameter('kD', 200.0).get_parameter_value().double_value
        self.curr_lim = self.declare_parameter('curr_lim', control_force).get_parameter_value().double_value

        self.prev_pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))

        # Subscribers
        self.create_subscription(JointState, 'cmd_leap', self._receive_pose, 10)
        self.create_subscription(JointState, 'cmd_allegro', self._receive_allegro, 10)
        self.create_subscription(JointState, 'cmd_ones', self._receive_ones, 10)

        # Service servers
        self.create_service(LeapPosition, 'leap_position', self.pos_srv)
        self.create_service(LeapVelocity, 'leap_velocity', self.vel_srv)
        self.create_service(LeapEffort, 'leap_effort', self.eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel_eff', self.pos_vel_eff_srv)
        self.create_service(LeapPosVelEff, 'leap_pos_vel', self.pos_vel_srv)

        # Connect motors
        self.motors = list(range(16))
        self.dxl_client = self._connect_dynamixel_on_first_available(
            ports=['/dev/ttyUSB0','/dev/ttyUSB1','/dev/ttyUSB2'],
            baud=4000000
        )
        # Set up position-current control mode
        self._setup_motor_gains()

        # Write default position
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _connect_dynamixel_on_first_available(self, ports, baud):
        for port in ports:
            try:
                client = DynamixelClient(self.motors, port, baud)
                client.connect()
                self.get_logger().info(f"Connected to LEAP Hand at {port}")
                return client
            except Exception as e:
                self.get_logger().warn(f"Failed to connect on {port}: {e}")
        raise RuntimeError("Could not connect to any specified port.")

    def _setup_motor_gains(self):
        # Position-current control
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(self.motors, True)
        # P, I, D gains
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
        # Current limits
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

    def _receive_pose(self, msg):
        pose = np.array(msg.position)
        self.prev_pos = self.curr_pos
        self.curr_pos = pose
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _receive_allegro(self, msg):
        pose = lhu.allegro_to_LEAPhand(msg.position, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _receive_ones(self, msg):
        pose = lhu.sim_ones_to_LEAPhand(np.array(msg.position))
        self.prev_pos = self.curr_pos
        self.curr_pos = pose
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    # Services
    def pos_srv(self, request, response):
        response.position = self.dxl_client.read_pos().tolist()
        return response

    def vel_srv(self, request, response):
        response.velocity = self.dxl_client.read_vel().tolist()
        return response

    def eff_srv(self, request, response):
        response.effort = self.dxl_client.read_cur().tolist()
        return response

    def pos_vel_srv(self, request, response):
        output = self.dxl_client.read_pos_vel()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = np.zeros_like(output[1]).tolist()
        return response

    def pos_vel_eff_srv(self, request, response):
        output = self.dxl_client.read_pos_vel_cur()
        response.position = output[0].tolist()
        response.velocity = output[1].tolist()
        response.effort = output[2].tolist()
        return response


###############################################################################
# 2. MinimalClientAsync - the client node that sends commands and queries
###############################################################################
class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')
        # Create a publisher to send positions to /cmd_ones
        self.pub_hand = self.create_publisher(JointState, '/cmd_ones', 10)

        # Create a client for the combined position/velocity/effort
        self.cli = self.create_client(LeapPosVelEff, '/leap_pos_vel_eff')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for leap_pos_vel_eff service...')
        self.req = LeapPosVelEff.Request()

    def send_request(self):
        """Call the /leap_pos_vel_eff service (async) and wait for the result."""
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

    def send_position_command(self, positions):
        """Send a JointState command to /cmd_ones topic."""
        stater = JointState()
        stater.position = positions
        self.pub_hand.publish(stater)


###############################################################################
# 4. Helper function to record the current state to a CSV file
###############################################################################
def record_current_state(response, filename='hand_data.csv'):
    """
    Appends the current state (position, velocity, effort) and a timestamp
    to a CSV file. One row per call.
    """
    # Unpack data
    pos = response.position
    vel = response.velocity
    eff = response.effort

    # Current time as a string
    timestamp = datetime.now().isoformat()

    # If the file doesn't exist, we create it with a header row
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            # First row: column names
            # We'll label them as position_i, velocity_i, effort_i for each motor
            headers = ['timestamp']
            for i in range(len(pos)):
                headers.append(f'pos_{i}')
            for i in range(len(vel)):
                headers.append(f'vel_{i}')
            for i in range(len(eff)):
                headers.append(f'eff_{i}')
            writer.writerow(headers)

        # Data row
        row = [timestamp]
        row.extend(pos)
        row.extend(vel)
        row.extend(eff)
        writer.writerow(row)

###############################################################################
# 3. Main entry point - launching both the server (LeapNode) and the client
###############################################################################


def LEAPsim_limits(type="regular"):
    sim_min = np.array([
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -1.047, -0.314, -0.506, -0.366,
        -0.349, -0.470, -1.200, -1.340
    ])
    sim_max = np.array([
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        1.047, 2.230, 1.885, 2.042,
        2.094, 2.443, 1.900, 1.880
    ])
    return sim_min, sim_max


def scale(x, lower, upper):
    """
    Scales each element of x from the range [-1,1] to [lower[i], upper[i]].
    """
    # Ensure inputs are NumPy arrays for element-wise arithmetic
    x = np.array(x, dtype=float)
    lower = np.array(lower, dtype=float)
    upper = np.array(upper, dtype=float)

    return 0.5 * (x + 1.0) * (upper - lower) + lower


#


# action_array=np.array([-0.95,
# 9.506860375404357910e-01,
# 1.434897541999816895e+00,
# 2.941232025623321533e-01,
# 3.547320961952209473e-01,
# 1.073872074484825134e-01,
# 5.337845683097839355e-01,
# 1.264732122421264648e+00,
# 4.409432113170623779e-01,
# -4.022725522518157959e-01,
# 6.938337087631225586e-01,
# 4.564338326454162598e-01,
# 9.723923206329345703e-01,
# 7.091732025146484375e-01,
# 2.029704451560974121e-01,
# 1.421890258789062500e+00])



# action_array=np.array([0.62605834, -0.03505718 , 0.29571724,  1.0513484 ,  0.4975195 ,  0.21455798,
#  -0.730583 ,   0.7814647 ,  0.14476797 , 1.0266266  , 0.10708403 , 0.90562475,
#   1.6820216  , 1.8892763 , -0.45290288 , 1.2549754])


# action_array=np.array([0.0297336 ,  0.15594059 , 1.0997438 ,  0.6966978 ,  0.2665866  , 1.0365664,
#   0.37705225,  0.22079311, -0.6849231,   0.54823774 , 0.677567  ,  0.16510376,
#   1.4361334,   0.39595047 , 0.37835258, 1.9554603 ])


# action_array=np.array([
# -0.01195235 , 0.6766392 ,  1.0432594 ,  0.5440032,  -0.00501621 , 0.9546762,
#   1.0434008 ,  0.71900964, -0.21668713 , 0.9927786 ,  0.9538724 ,  0.9062224,
#   1.1594981 ,  1.2746018 , -0.28693497 , 1.2444932 ])
# renormalize this action array from [-1.47,1.47] to the range of the leap hand acition [-0.5,0.5]


# general control:


# action_array=np.array([ 0.0297336 ,  0.15594059 , 1.0997438 ,  0.6966978 ,  0.2665866 ,  1.0365664,
#   0.37705225 , 0.22079311 ,-0.6849231  , 0.54823774 , 0.677567 ,   0.16510376,
#   1.4361334 ,  0.39595047 , 0.37835258 , 1.9554603 ])


#200g
action_array=np.array([-1.2579879 ,  0.40826976 , 0.28859434 , 1.0574762 ,  0.5580783  , 1.123451,
  1.205674 , -0.5383857 , -1.294615 ,   0.71414083 , 1.229035 ,   1.6270905,
  1.2873634 ,  0.693488 ,  -0.07616547, -0.6797438,
])


#126g

# action_array=np.array([0.8124344 ,  0.299228 ,  -0.11970427,  0.09359938 ,-0.5971941,   0.8560429,
#   0.8359419,  -0.9045451 ,  0.79267544,  0.89426804 ,-0.2677437 ,  1.3074456,
#   1.1172608 ,  1.3110616 , -1.3843669,   0.8756635,
# ])


# 80g
# action_array=np.array([0.42499572,  0.19120781 , 1.0634233,   0.00620188,  0.21192083,  0.8419687,
#  -0.06676156 ,-0.07401133, -0.10224228 , 0.51908296,  0.38253304,  1.2504752,
#   0.6201371 , -0.26427966 ,-0.3665491 ,  0.6669311,
# ])


old_min, old_max = -1.47, 1.47
new_min, new_max = -0.5, 0.5

# Vectorized scaling:
scaled_array = new_min + (action_array - old_min) * (new_max - new_min) / (old_max - old_min)


def main(args=None):
    rclpy.init(args=args)

    # Create both nodes
    
    # gt 
    # prediced_force=0.65

    #200 
    prediced_force=0.78 # range from 0 to 1 given the object mass we learn from single push

    #126
    # prediced_force=0.49
    #80
    # prediced_force=0.313

    num_finger=16
    control_current=prediced_force*num_finger*15
    leap_node = LeapNode(control_force=control_current)
    client_node = MinimalClientAsync()

    # Executor to handle them in separate threads
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(leap_node)
    executor.add_node(client_node)

    # We’ll run the main “control loop” in a separate Python thread
    # so the node callbacks can still be processed by the executor.
    sim_min, sim_max = LEAPsim_limits()
    def control_loop():
        x = scaled_array.tolist()  # Use the scaled action array
        # x =[0.0]*16
        # Convert to NumPy array and clip
        x_array = np.array(x)
        # x_clipped = scale(x_array, sim_min, sim_max)

            # Send clipped positions
        
        client_node.send_position_command(x)
        time.sleep(0.05)

    # Start the control loop in a background thread
    loop_thread = threading.Thread(target=control_loop, daemon=True)
    loop_thread.start()

    # Now spin both nodes in the executor, handling callbacks
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Clean up
    leap_node.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
