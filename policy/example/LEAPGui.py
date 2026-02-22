
import os
import sys

# Append the scripts folder to Python's search path
sys.path.append("/home/haozhe/leap_ws/src/ros2_module/scripts")

import numpy as np
from leap_hand_utils.dynamixel_client import DynamixelClient
import leap_hand_utils.leap_hand_utils as lhu
import time
import asyncio
import glob
import tkinter as tk
from tkinter import ttk
import signal
import sys

USE_GUI = True

class ImprovedLeapHand:
    def __init__(self):
        # 初始参数设置
        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 550
        self.initial_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        self.prev_pos = self.curr_pos = self.initial_pos.copy()
        
        self.motors = list(range(16))
        self.dxl_client = None
        if not self.connect():
            raise Exception("Failed to connect to LEAP Hand")
        self.initialize_hand()

    def connect(self):
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*') + ['COM13']
        for port in ports:
            try:
                self.dxl_client = DynamixelClient(self.motors, port, 4000000)
                self.dxl_client.connect()
                print(f"Connected to LEAP Hand on {port}")
                return True
            except Exception as e:
                print(f"Failed to connect on {port}: {e}")
        print("Failed to connect to LEAP Hand on any available port")
        return False

    def initialize_hand(self):
        if not self.dxl_client:
            print("LEAP Hand is not connected, skipping initialization")
            return
        try:
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors))*5, 11, 1)
            self.dxl_client.set_torque_enabled(self.motors, True)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
            self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
            self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2)
            self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
            self.set_initial_position()
        except Exception as e:
            print(f"Error during hand initialization: {e}")
            raise

    def set_initial_position(self):
        if not self.dxl_client:
            print("LEAP Hand is not connected, cannot set initial position")
            return
        self.dxl_client.write_desired_pos(self.motors, self.initial_pos)
        time.sleep(2)
        self.curr_pos = self.read_pos()
        print("Hand set to initial position:", self.curr_pos)

    def set_leap(self, pose):
        if not self.dxl_client:
            print("LEAP Hand is not connected, cannot set pose")
            return
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.set_leap(pose)

    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.set_leap(pose)

    def set_specific_joint(self, joint_positions):
        if not self.dxl_client:
            print("LEAP Hand is not connected, cannot set joint positions")
            return

        current_pose = self.read_pos()
        for joint, position in joint_positions.items():
            if 0 <= joint < 16:
                current_pose[joint] = position
            else:
                print(f"Invalid joint index: {joint}")

        self.set_leap(current_pose)

    def read_pos(self):
        return self.dxl_client.read_pos() if self.dxl_client else np.zeros(16)

    def read_vel(self):
        return self.dxl_client.read_vel() if self.dxl_client else np.zeros(16)

    def read_cur(self):
        return self.dxl_client.read_cur() if self.dxl_client else np.zeros(16)

    def get_status(self):
        return {
            "current_position": self.read_pos().tolist(),
            "target_position": self.curr_pos.tolist(),
            "velocity": self.read_vel().tolist(),
            "current": self.read_cur().tolist()
        }

class LeapHandHandler:
    def __init__(self):
        try:
            self.leap_hand = ImprovedLeapHand()
            self.available = True
        except Exception as e:
            print(f"Warning: Unable to initialize LeapHandHandler: {e}")
            self.available = False
            self.leap_hand = None

    def control(self, joint_positions):
        if not self.available:
            print("LeapHandHandler is not available")
            return
        self.leap_hand.set_specific_joint(joint_positions)

    def get_current_positions(self):
        if not self.available:
            print("LeapHandHandler is not available")
            return None
        return self.leap_hand.read_pos().tolist()

class LeapHandGUI:
    def __init__(self, master, leap_hand_handler):
        self.master = master
        self.leap_hand_handler = leap_hand_handler
        self.master.title("LEAP Hand Control")

        self.sliders = []
        self.entries = []

        # Create sliders and entry boxes
        for i in range(16):
            frame = ttk.Frame(self.master)
            frame.grid(row=i, column=0, sticky="w", padx=10, pady=5)

            label = ttk.Label(frame, text=f"Joint {i}:")
            label.pack(side=tk.LEFT)

            slider = ttk.Scale(frame, from_=1.4, to=5.1, orient=tk.HORIZONTAL, length=200, command=lambda value, index=i: self.update_hand_from_slider(value, index))
            slider.pack(side=tk.LEFT)
            slider.set(3.25)  # Set default value
            self.sliders.append(slider)

            entry = ttk.Entry(frame, width=10)
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, "3.25")  # Set default value
            self.entries.append(entry)

        # Create update button for entries
        update_button = ttk.Button(self.master, text="Update from Entries", command=self.update_hand_from_entries)
        update_button.grid(row=16, column=0, pady=10)

        # Create status display
        self.status_frame = ttk.Frame(self.master)
        self.status_frame.grid(row=0, column=1, rowspan=17, padx=20)

        self.status_labels = []
        for i in range(16):
            label = ttk.Label(self.status_frame, text=f"Joint {i}: 0.00")
            label.grid(row=i//4, column=i%4, padx=5, pady=5)
            self.status_labels.append(label)

        self.update_status()

    def update_hand_from_slider(self, value, index):
        self.entries[index].delete(0, tk.END)
        self.entries[index].insert(0, f"{float(value):.2f}")
        joint_positions = {index: float(value)}
        self.leap_hand_handler.control(joint_positions)

    def update_hand_from_entries(self):
        joint_positions = {}
        for i, entry in enumerate(self.entries):
            try:
                value = float(entry.get())
                if 1.4 <= value <= 5.1:
                    joint_positions[i] = value
                    self.sliders[i].set(value)
                else:
                    print(f"Invalid value for Joint {i}. Must be between 1.4 and 5.1.")
            except ValueError:
                print(f"Invalid input for Joint {i}. Please enter a number.")
        if joint_positions:
            self.leap_hand_handler.control(joint_positions)

    def update_status(self):
        positions = self.leap_hand_handler.get_current_positions()
        if positions:
            for i, pos in enumerate(positions):
                self.status_labels[i].config(text=f"Joint {i}: {pos:.2f}")
        self.master.after(100, self.update_status)  # Update every 100ms

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    if USE_GUI:
        root.quit()  
    sys.exit(0)  

async def main():
    signal.signal(signal.SIGINT, signal_handler) 

    leap_hand_handler = LeapHandHandler()
    
    if USE_GUI:
        global root  
        root = tk.Tk()
        app = LeapHandGUI(root, leap_hand_handler)
        try:
            root.mainloop()
        except KeyboardInterrupt:
            print("Caught keyboard interrupt. Exiting...")
        finally:
            root.quit()
            root.destroy()
    else:
        while True:
            try:
                command = input("Enter command (joint:position pairs, 'status' for current positions, or 'quit' to exit): ")
                if command.lower() == 'quit':
                    break
                elif command.lower() == 'status':
                    positions = leap_hand_handler.get_current_positions()
                    if positions:
                        print("Current joint positions:", positions)
                else:
                    try:
                        joint_positions = {}
                        pairs = command.split(',')
                        for pair in pairs:
                            joint, position = map(float, pair.strip().split(':'))
                            joint_positions[int(joint)] = position
                        leap_hand_handler.control(joint_positions)
                    except ValueError:
                        print("Invalid input. Please enter joint:position pairs separated by commas.")
            except KeyboardInterrupt:
                print("\nCaught keyboard interrupt. Exiting...")
                break

    print("Cleaning up...")
    
    if leap_hand_handler.available:
        leap_hand_handler.leap_hand.dxl_client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())