
#!/usr/bin/env python3

import os
import sys

# Append the scripts folder to Python's search path
sys.path.append("/home/haozhe/leap_ws/src/ros2_module/scripts")


import sys
import time
import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState

from leap_hand.srv import LeapPosition, LeapVelocity, LeapEffort, LeapPosVelEff




# If you have these utilities in a separate package, import them.
# Otherwise, copy them directly here or ensure they are on your PYTHONPATH.
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu


###############################################################################
# 1. LEAPNode - the main “hand” node (copied from your code, slightly shortened)
###############################################################################
class LeapNode(Node):
    def __init__(self):
        super().__init__('leaphand_node')
        # Some parameters to control the hand
        self.kP = self.declare_parameter('kP', 800.0).get_parameter_value().double_value
        self.kI = self.declare_parameter('kI', 0.0).get_parameter_value().double_value
        self.kD = self.declare_parameter('kD', 200.0).get_parameter_value().double_value
        self.curr_lim = self.declare_parameter('curr_lim', 350.0).get_parameter_value().double_value

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

#
# MinimalClientAsync from your existing code.
#
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

#
# A simple Slider class to manage a horizontal slider, with a draggable handle
#
class Slider:
    def __init__(self, x, y, width, height, min_val=-1.0, max_val=1.0, initial=0.0):
        self.rect = pygame.Rect(x, y, width, height)
        self.handle_width = 20
        self.handle_height = height
        self.handle_rect = pygame.Rect(x, y, self.handle_width, self.handle_height)

        self.min_val = min_val
        self.max_val = max_val
        self._value = initial  # stored as float in [min_val, max_val]

        self.dragging = False
        # Position the handle according to self._value
        self.update_handle_position_from_value()

    def update_handle_position_from_value(self):
        slider_range = self.rect.width - self.handle_width
        fraction = (self._value - self.min_val) / (self.max_val - self.min_val)
        new_x = self.rect.x + fraction * slider_range
        self.handle_rect.x = int(new_x)

    def value_from_handle_position(self):
        slider_range = self.rect.width - self.handle_width
        fraction = (self.handle_rect.x - self.rect.x) / slider_range
        val = self.min_val + fraction * (self.max_val - self.min_val)
        return val

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_val):
        self._value = np.clip(new_val, self.min_val, self.max_val)
        self.update_handle_position_from_value()

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            new_x = event.pos[0] - self.handle_width // 2
            new_x = max(new_x, self.rect.x)
            new_x = min(new_x, self.rect.right - self.handle_width)
            self.handle_rect.x = new_x
            self._value = self.value_from_handle_position()

    def draw(self, surface):
        pygame.draw.rect(surface, (180, 180, 180), self.rect)
        pygame.draw.rect(surface, (80, 80, 200), self.handle_rect)



import threading


def main(args=None):
    rclpy.init(args=args)

    # Create the server node (LeapNode) and the client node (MinimalClientAsync)
    leap_node = LeapNode()
    client_node = MinimalClientAsync()

    # Create an executor with at least two threads (recommended),
    # and add BOTH nodes.
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(leap_node)
    executor.add_node(client_node)

    # Define our PyGame control loop in a separate thread
    def control_loop():
        pygame.init()
        screen_width = 600
        screen_height = 500
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("16 Slider Control (All Start at 0.0)")

        clock = pygame.time.Clock()

        # Create 16 sliders
        sliders = []
        slider_width = 400
        slider_height = 20
        start_x = 100
        start_y = 30
        spacing = 30

        for i in range(16):
            s = Slider(
                x=start_x,
                y=start_y + i*(slider_height + spacing),
                width=slider_width,
                height=slider_height,
                min_val=-1.0,
                max_val=1.0,
                initial=0.0
            )
            sliders.append(s)

        running = True
        while running:
            # Handle PyGame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                else:
                    for slider in sliders:
                        slider.handle_event(event)

            # Clear and draw
            screen.fill((255, 255, 255))
            for slider in sliders:
                slider.draw(screen)
            pygame.display.flip()
            clock.tick(30)

            # Publish slider values
            x = [slider.value for slider in sliders]
            client_node.send_position_command(x)

            # You can reduce this sleep to publish more often
            time.sleep(0.1)

        pygame.quit()

    # Start the PyGame thread (daemon=True so it ends with the main program)
    loop_thread = threading.Thread(target=control_loop, daemon=True)
    loop_thread.start()

    # Now spin the executor in the main thread, which allows
    # LeapNode to process the incoming commands from /cmd_ones.
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    # Cleanup
    leap_node.destroy_node()
    client_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()