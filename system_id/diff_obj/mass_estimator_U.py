"""
Recover the mass of an object with known shape.
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm, trange

from gradsim.bodies import RigidBody
from gradsim.forces import ConstantForce
from gradsim.renderutils import SoftRenderer
from gradsim.simulator import Simulator
from gradsim.utils import meshutils
import trimesh
import open3d as o3d

import ast

import mujoco.viewer
import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from tqdm import trange
import imageio

import mediapy as media

import os
import time

import pandas as pd



# The JAX-based MuJoCo wrapper.
from mujoco import mjx
import mujoco

# from mujoco.mjx._src.math import quat_integrate, motion_cross_force, inert_mul,matmul_unroll
# from mujoco.mjx._src.collision_driver import collision
# from mujoco.mjx._src.support import contact_force_dim

# from mujoco.mjx._src.types import ConeType



from scipy.spatial import cKDTree



# The inverse of Q_flip = ( sqrt(2)/2, - sqrt(2)/2, 0, 0 )
Q_FLIP_INV = np.array([0.7071068, 0.7071068, 0.0, 0.0])
# global varible for substep
dt = 0.002

def find_closest_vertices(vertices, contact_points,k=1):
    """
    Find the closest vertex in 'vertices' for each point in 'contact_points'.
    
    Args:
        vertices (np.ndarray): Array of shape (N, 3) or (1, N, 3). 
                               The set of vertex coordinates.
        contact_points (np.ndarray): Array of shape (M, 3) or (1, M, 3). 
                                     The contact point coordinates.

    Returns:
        closest_indices (np.ndarray): Shape (M,). The index of the closest 
                                      vertex for each contact point.
        distances (np.ndarray): Shape (M,). The distance to that closest 
                                vertex for each contact point.
    """
    
    # If your arrays have a leading dimension of 1 (batch dimension), squeeze it out:
    vertices_k = vertices.squeeze(0).clone()
    vertices_k=vertices_k.detach().cpu().numpy()      # shape now (N, 3)
    contact_points = contact_points # shape now (M, 3)
    # Build a KD-tree over your vertices
    kd_tree = cKDTree(vertices_k)  

    # Query the tree for each contact point
    distances, closest_indices = kd_tree.query(contact_points, k=k)
    # distances: shape (M,)
    # closest_indices: shape (M,)

    return closest_indices, distances



# pretend your object has a full sensor

def flip_back_position(pos_flipped):
    """
    Flip back a position vector that was rotated by -90 degrees about X.
    pos_flipped: [x_new, y_new, z_new]
    Returns the original [x_old, y_old, z_old].
    """
    x_new, y_new, z_new = pos_flipped
    x_old = x_new
    y_old = -z_new
    z_old = y_new
    return np.array([x_old, y_old, z_old], dtype=float)



def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions q1 * q2 (each a 4-element array [w, x, y, z]).
    Returns the product as [w, x, y, z].
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z], dtype=float)


# body_id = 

def flip_back_quaternion(q_flipped):
    """
    Undo the -90° rotation about X on a quaternion q_flipped.
    q_flipped: [w_flipped, x_flipped, y_flipped, z_flipped]
    Returns q_original.
    """
    return quaternion_multiply(Q_FLIP_INV, q_flipped)


    # ---------------------------------------------------------------
    # Example placeholder for apply_action_and_step function
    # ---------------------------------------------------------------
def apply_action_and_step(mj_model, mj_data, action):
    """
    Example function that applies `action` to mj_data.ctrl,
    then steps the simulation. Replace with your own code logic.
    """
    # In many tasks, `action` is an array that you would assign to mj_data.ctrl:
    mj_data.ctrl[:] = action
    mujoco.mj_step(mj_model, mj_data)



###############################################################################
# 1) Data Loading Helpers (unchanged)
###############################################################################
def load_position_data(csv_path):
    position_cols = [f'position_{i}' for i in range(16)]
    df = pd.read_csv(csv_path, usecols=position_cols)
    return df.to_numpy()

def load_observations(gt_path):
    import re, ast
    from datetime import datetime

    def parse_iso_time(time_str):
        return datetime.fromisoformat(time_str)

    pose_list = []
    with open(gt_path, 'r') as f:
        lines = f.read().strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Final transform:" in line:
            match = re.match(r"^(.*?) - Final transform:", line)
            if not match:
                i += 1
                continue
            timestamp = match.group(1).strip()
            i += 1
            matrixF = []
            for _ in range(4):
                mat_line = lines[i].strip()
                row = mat_line.strip('[]').split()
                row = list(map(float, row))
                matrixF.append(row)
                i += 1
            final_transform = np.array(matrixF)
            if i >= len(lines):
                break
            i += 1
            pose_lines = []
            while i < len(lines):
                pose_lines.append(lines[i])
                i += 1
                if ']]])' in pose_lines[-1]:
                    break
            pose_str = ''.join(pose_lines)
            pose_str = pose_str.replace('tensor(', '').rstrip(')')
            try:
                pose_data = ast.literal_eval(pose_str)
                if (len(pose_data) == 1 and isinstance(pose_data[0], list) 
                    and len(pose_data[0]) == 4):
                    pose_array = np.array(pose_data[0])
                elif len(pose_data) == 1 and len(pose_data[0]) == 1:
                    pose_array = np.array(pose_data[0][0])
                else:
                    pose_array = np.array(pose_data)
            except:
                pose_array = np.zeros((4,4))
            pose_list.append((timestamp, final_transform, pose_array))
        else:
            i += 1
    return pose_list



# filter out the contact force, contact point and contact point id on the vertices(knn)


# then we can try to find the  visulize result of the forward simulation of the gradsim


def process_actuation_force(
    Contact_list,
    qfrc_smooth_list,
    start_sync_frame,
    body_geom_id,
    plate_geom_id,
):
    """
    Scans the Contact_list from 'start_sync_frame' onward. For each frame:
      - Looks at all MjContact objects in that frame.
      - Captures only the ones in which geom1 and geom2 are NOT 42 or 43.
      - If at least one such contact exists, we store:
          (1) That frame's index (the 'impulse frame'),
          (2) The force from qfrc_smooth[16:19],
          (3) All contact positions for that frame.

    Args:
        Contact_list (list of list(MjContact)): 
            Indexed by frame: Contact_list[frame_index] is a list of MjContact.
        qfrc_smooth_list (list of ndarray):
            Each entry is the qfrc_smooth array at a specific frame.
        start_sync_frame (int):
            Which frame to start scanning from.

    Returns:
        impulse_frames (list of int): 
            The frame indices where we found a contact not involving geom 42 or 43.
        smooth_forces_per_frame (list of ndarray):
            The extracted qfrc_smooth[16:19] for each of those frames (shape (3,)).
        contact_positions_per_frame (list of ndarray):
            Each element is an (N,3) array of contact positions found at that frame.
    """
    # Lists that will grow as we find relevant contacts
    impulse_frames = []              # which frames had the target contacts
    smooth_forces_per_frame = []     # the qfrc_smooth[16:19] for those frames
    contact_positions_per_frame = [] # list of arrays, each array is (N,3) for that frame

    # Loop over frames from 'start_sync_frame' onward
    for frame_idx in range(start_sync_frame, len(Contact_list)):
        contacts_this_frame = Contact_list[frame_idx]

        # Gather all positions that meet the condition "geom != 42 or 43"
        positions_for_this_frame = []
        for contact in contacts_this_frame:
            g0, g1 = contact.geom1, contact.geom2

            # If neither geom is 42 or 43, we store the position
            if g0 not in (body_geom_id, plate_geom_id) and g1 not in (body_geom_id, plate_geom_id):
                positions_for_this_frame.append(contact.pos)

        # If we found at least one contact that meets the condition, record it
        if positions_for_this_frame:
            impulse_frames.append(frame_idx)
            
            # Convert the list of positions to a stacked array (N,3)
            positions_for_this_frame = np.stack(positions_for_this_frame)
            contact_positions_per_frame.append(positions_for_this_frame)

            # Extract qfrc_smooth for that frame, specifically indices 16:19
            # Ensure qfrc_smooth_list[frame_idx] is an array
            smooth_arr = np.array(qfrc_smooth_list[frame_idx])
            smooth_forces_per_frame.append(smooth_arr[16:19])

    return impulse_frames, smooth_forces_per_frame, contact_positions_per_frame










def transform_to_pos_quat_batch(transforms_4x4):
    import mujoco
    N = transforms_4x4.shape[0]
    pos = np.zeros((N, 3), dtype=np.float64)
    quat = np.zeros((N, 4), dtype=np.float64)
    for i in range(N):
        rot3x3 = transforms_4x4[i, :3, :3]
        pos[i]  = transforms_4x4[i, :3, 3]
        mujoco.mju_mat2Quat(quat[i], rot3x3.ravel())
    return pos, quat

def slerp(q0, q1, t):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q_lin = q0 + t*(q1 - q0)
        return q_lin / np.linalg.norm(q_lin)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0*dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)




def interpolate_transforms(positions, quaternions, n_samples=1080):
    N = len(positions)
    if N < 2:
        raise ValueError("Need at least two keyframes.")
    
    # We map our original keyframes to times [0..1].
    original_times = np.linspace(0, 1, N)
    # The new times we want to sample
    new_times = np.linspace(0, 1, n_samples)

    up_pos, up_quat = [], []

    # We'll define the midpoint index:
    half_idx = n_samples // 1.15

    for i, t in enumerate(new_times):
        idx = np.searchsorted(original_times, t)
        
        if idx == 0:
            p_new = positions[0].copy()
            q_new = quaternions[0].copy()
        elif idx >= N:
            p_new = positions[-1].copy()
            q_new = quaternions[-1].copy()
        else:
            t1, t2 = original_times[idx-1], original_times[idx]
            alpha = (t - t1) / (t2 - t1)
            p0, p1 = positions[idx-1], positions[idx]
            q0, q1 = quaternions[idx-1], quaternions[idx]
            
            # Linear interpolation of positions
            p_new = (1 - alpha) * p0 + alpha * p1
            # SLERP for orientation
            q_new = slerp(q0, q1, alpha)
        
        # Timestep between consecutive new_times
        # For the very first sample, dt=0 so there's no penalty
        if i == 0:
            dt = 0.0
        else:
            dt = new_times[i] - new_times[0]

        # ---------------------------
        # Gravity & Friction Penalties, only when data is poisoned 
        # ---------------------------
        if i < half_idx:
            # First half: smaller gravity, friction
            gravity_amount   = 9.8 * p_new[2] * dt * 0.0001
            friction_amount  = 1.0 * dt * 0.0001
        else:
            # Second half: bigger gravity, friction
            gravity_amount   = 9.8 * dt * 0.0025
            friction_amount  = 1.0 * dt * 0.005

        # Apply gravity along -Z
        p_new[2] -= gravity_amount
        # Apply friction along +X
        p_new[0] += friction_amount

        up_pos.append(p_new)
        up_quat.append(q_new)

    return np.array(up_pos), np.array(up_quat)

def process_and_interpolate(loaded_pose,
                            reference_transform_inv,
                            offset_vector,
                            sync_with_real_vector,
                            n_samples=1080):
    timestamps = []
    final_transform_list = []
    for (ts, final_transform_tmp, _) in loaded_pose:
        timestamps.append(ts)
        final_transform_list.append(final_transform_tmp)
    final_transform = np.array(final_transform_list)
    final_transform_rel = reference_transform_inv[None] @ final_transform

    # foundationpose to mujoco coordinate, different for each foundationpose catching
    swap_matrix = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    flipped_transform = final_transform_rel @ swap_matrix
    pos, quat = transform_to_pos_quat_batch(flipped_transform)
    pos = pos - offset_vector[None, :] + sync_with_real_vector[None, :]
    up_pos, up_quat = interpolate_transforms(pos, quat, n_samples=n_samples)
    return timestamps, up_pos, up_quat

###############################################################################
# 2) MJX-based Simulation Helpers (unchanged)
###############################################################################

def run_simulation(mj_model, mj_data, pre_timestamp_action):
        print("pre_timestamp_action", pre_timestamp_action)

        duration = 5.0     # seconds
        framerate = 60     # frames per second
        sim_steps = int(control_step)

        # Prepare lists for storing simulation data
        qpos_list = []
        qacc_list = []
        qfrc_actuator_list = []
        contact_list = []
        contacts_str_list= []
        qfrc_smooth_list = []

        # For rendering to frames (optional):
        frames = []
        height, width = 480, 640

        with mujoco.Renderer(mj_model, height=height, width=width) as renderer:
            for step_i in range(sim_steps):
                # ---------------------------------------------------------
                # 1) Record all data BEFORE stepping the simulation

                # ---------------------------------------------------------
                # 2) Apply action and step the physics once
                #    (Replace 'apply_action_and_step' with your own method.)
                # ---------------------------------------------------------
                pre_timestamp_action= control_signal[step_i]
                apply_action_and_step(mj_model, mj_data, pre_timestamp_action)

                # ---------------------------------------------------------
                # 3) Render the scene from a chosen camera
                # ---------------------------------------------------------
                # For example, "side" or "top" camera
                renderer.update_scene(mj_data, camera="side")
                pixels = renderer.render()
                    
                qpos_list.append(mj_data.qpos.copy())
                qacc_list.append(mj_data.qacc.copy())
                qfrc_actuator_list.append(mj_data.qfrc_actuator.copy())

                # If you only need contact forces/positions, parse the struct fields
                # For now, we store each contact object as a string:
                contacts_str = [str(mj_data.contact[j]) for j in range(mj_data.ncon)]
                contacts_str_list.append(contacts_str)
                active_contacts=[(mj_data.contact[j]) for j in range(mj_data.ncon)]
                contact_list.append(active_contacts)

                qfrc_smooth_list.append(mj_data.qfrc_smooth.copy())

                # Store the frame
                frames.append(pixels)

        # --------------------------------------------
        # Print final info (optional debugging)
        # --------------------------------------------
        print("Final sim time:", mj_data.time)
        print("Number of contacts at end:", mj_data.ncon)

        # --------------------------------------------
        # Optional: Save frames to a video with mediapy
        # --------------------------------------------
        media.write_video("simulation_video.mp4", frames, fps=framerate)
        print("Video saved to 'simulation_video.mp4'")

        # --------------------------------------------
        # Save the data to force_list.txt
        # --------------------------------------------
        with open("force_list.txt", "w") as f:
            for step_i in range(sim_steps):
                f.write(f"--- Step {step_i} ---\n")

                f.write("qpos:\n")
                f.write(np.array2string(qpos_list[step_i], separator=","))
                f.write("\n\n")

                f.write("qacc:\n")
                f.write(np.array2string(qacc_list[step_i], separator=","))
                f.write("\n\n")

                f.write("qfrc_actuator:\n")
                f.write(np.array2string(qfrc_actuator_list[step_i], separator=","))
                f.write("\n\n")

                f.write("Contact:\n")
                if contacts_str_list[step_i]:
                    for c in contacts_str_list[step_i]:
                        f.write(c + "\n")
                else:
                    f.write("No contacts\n")
                f.write("\n")

                f.write("qfrc_smooth:\n")
                f.write(np.array2string(qfrc_smooth_list[step_i], separator=","))
                f.write("\n\n")

                f.write("\n")  # extra line between steps

        print("Data saved to 'force_list.txt'")

        return qpos_list, qacc_list, qfrc_actuator_list, contact_list, qfrc_smooth_list

def load_mjx_model(xml_path):
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    
    mj_model.opt.timestep = dt

    # For example, adjust actuator force ranges (if needed)
    mj_model.actuator_forcelimited[12]=True
    mj_model.actuator_forcelimited[14]=True
    mj_model.actuator_forcerange[14][0] = 0
    mj_model.actuator_forcerange[14][1] = 0.30
    mj_model.actuator_forcerange[12][0] = 0
    mj_model.actuator_forcerange[12][1] = 0.70

    mj_model.actuator_gear[14][0] = 0.3 
    mj_model.actuator_gear[14][1] = 0.3 
    mj_model.actuator_gear[14][2] = 0.3 

    mj_model.actuator_gear[12][0] = 0.3 
    mj_model.actuator_gear[12][1] = 0.3 
    mj_model.actuator_gear[12][2] = 0.3 

    body_name = "U"
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    joint_id = mj_model.body_jntadr[body_id]
    qpos_addr = mj_model.jnt_qposadr[joint_id]
    mj_model.body_mass[body_id] = 0.124
    num_dofs = 7  # free joint: 3 pos + 4 quat
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)
    return mjx_model, mjx_data, mj_model, mj_data, body_id, joint_id, qpos_addr, num_dofs

# Suppose these are your own modules for simulation and rendering
# from your_physics_simulator import RigidBody, Simulator, ConstantForce
# from soft_renderer import SoftRenderer

class Model(torch.nn.Module):
    """Wrap masses into a torch.nn.Module, for ease of optimization."""
    def __init__(self, masses, uniform_density=False):
        super(Model, self).__init__()
        self.update = None
        if uniform_density:
            print("Using uniform density assumption...")
            self.update = torch.nn.Parameter(torch.rand(1) * 0.1)
        else:
            print("Assuming nonuniform density...")
            self.update = torch.nn.Parameter(torch.rand(masses.shape) * 0.1)
        self.masses = masses

    def forward(self):
        return torch.nn.functional.relu(self.masses + self.update)


def load_sim_data(filepath):
    """
    Reads a MuJoCo force_list-style text file line by line, parsing each
    '--- Step i ---' block for qpos, qacc, qfrc_actuator, contact, and qfrc_smooth.

    Returns:
        A list of dicts, one per step, e.g.:
        [
            {
                "step": 0,
                "qpos": [...],
                "qacc": [...],
                "qfrc_actuator": [...],
                "contact": [ "<MjContact...>", "<MjContact...>", ... ],
                "qfrc_smooth": [...]
            },
            {
                "step": 1,
                ...
            },
            ...
        ]
    """
    all_steps = []
    current_step = {}
    contact_lines = []      # Accumulate contact lines here
    reading_contact = False # Flag to know we are in the contact section

    with open(filepath, "r") as f:
        for line in f:
            line = line.rstrip("\n")

            # Ignore completely empty lines
            if not line.strip():
                # If we were reading contact lines, an empty line means we
                # ended the contact block:
                if reading_contact:
                    current_step["contact"] = contact_lines[:]
                    contact_lines.clear()
                    reading_contact = False
                continue

            # Check for a new step: "--- Step X ---"
            if line.startswith("--- Step"):
                # If we had a previous step, store it before starting a new one
                if current_step:
                    # If contact was in progress and never terminated by a blank line
                    if reading_contact:
                        current_step["contact"] = contact_lines[:]
                        contact_lines.clear()
                        reading_contact = False

                    all_steps.append(current_step)
                    current_step = {}

                # Parse the step index
                parts = line.split()
                step_index = int(parts[2])
                current_step["step"] = step_index
                continue

            # Check for the keys we know exist in each step
            if line.startswith("qpos:"):
                # Next non-empty line should be the array
                array_line = next(f).strip()
                current_step["qpos"] = ast.literal_eval(array_line)
                continue

            if line.startswith("qacc:"):
                array_line = next(f).strip()
                current_step["qacc"] = ast.literal_eval(array_line)
                continue

            if line.startswith("qfrc_actuator:"):
                array_line = next(f).strip()
                current_step["qfrc_actuator"] = ast.literal_eval(array_line)
                continue

            if line.startswith("Contact:"):
                # Next lines (until a blank line or next key) are contact lines
                contact_lines.clear()
                reading_contact = True
                continue

            if line.startswith("qfrc_smooth:"):
                array_line = next(f).strip()
                current_step["qfrc_smooth"] = ast.literal_eval(array_line)
                continue

            # If we reach here while reading contact lines, this is part of the contact info
            if reading_contact:
                contact_lines.append(line)
            else:
                # If there's any extraneous text not accounted for, decide how you want to handle it.
                pass

        # At the end of the file, if the last step was never appended, do so now
        if current_step:
            if reading_contact:
                # Final block ended without a blank line
                current_step["contact"] = contact_lines
            all_steps.append(current_step)

    return all_steps



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expid",
        type=str,
        default="default",
        help="Unique string identifier for experiments.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="/home/haozhe/Desktop/robotic_toolset/gradsim/cache/mass_known_shape",
        help="Directory to store logs in.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed (for repeatability)"
    )

 
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to run optimization for.",
    )
    parser.add_argument(
        "--compare-every",
        type=int,
        default=10,
        help="Compare GT vs Est every N frames for the trajectory loss.",
    )
    parser.add_argument(
        "--force-magnitude",
        type=float,
        default=10.0,
        help="Magnitude of external force.",
    )
    parser.add_argument("--log", action="store_true", help="Save log files.")

    args = parser.parse_args()

    render_outfile = Path("demoforces.gif")

    # Seed RNG for repeatability
    torch.manual_seed(args.seed)

    # Device to store tensors on
    device = "cuda:0"

    # ------------------------------------------------------------------------
    # 1) Load a body (from a triangle mesh obj or stl file)
    # ------------------------------------------------------------------------
    def normalize_vertices(vertices):
        # Example: center and scale the vertices
        min_vals = vertices.min(dim=1, keepdim=True)[0]
        max_vals = vertices.max(dim=1, keepdim=True)[0]
        return (vertices - (min_vals + max_vals) / 2) / (max_vals - min_vals).max()

    # Load the mesh from STL/OBJ using trimesh
    mesh_path = "/home/haozhe/Dropbox/physics/_data/allegro/wonik_allegro/U.stl"
    
    # out_path = "/home/haozhe/Dropbox/physics/_data/allegro/wonik_allegro/assets/U_simplified.obj"
    
    mesh = o3d.io.read_triangle_mesh(mesh_path)
   
   # can you apply this quat to the mesh object
    q=np.array([0.5, 0.5, 0.5, 0.5])
    R = o3d.geometry.get_rotation_matrix_from_quaternion(q)

    mesh.rotate(R, center=mesh.get_center())
  

    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().unsqueeze(0).to(device)
    faces = torch.from_numpy(np.asarray(mesh.triangles)).long().unsqueeze(0).to(device)
    # Optional: vertices = normalize_vertices(vertices)

    # Dummy texture: shape = [1, num_faces, 2, 1, 3]
    textures = torch.cat(
        (
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),  # R=1
            torch.ones(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device),  # G=1
            torch.zeros(1, faces.shape[1], 2, 1, dtype=torch.float32, device=device), # B=0
        ),
        dim=-1,
    )

    # Ground-truth masses, just an example
    masses_gt = torch.nn.Parameter(
        torch.arange(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=False,
    )

    # ------------------------------------------------------------------------
    # 2) Create a ground-truth body and run a single simulation for the GT data
    # ------------------------------------------------------------------------

    # for the bluelego case
    # raw_position=torch.tensor([0.095, -0.08, 0.1], device=device)
    # raw_position_offset_vector = np.array([0.0, 0.0, 0.1])
    flip_matrix= torch.tensor([[1,0,0], [0,0,1], [0.0, 1, 0]], device=device)
    flip_matrix_np= flip_matrix.cpu().numpy()
    # flipped_position = flip_matrix @ raw_position
    # print('flipped_position:', flipped_position)


    # in simulator and real world
    # x z y
    # raw_position=torch.tensor([0.095, 0.13, -0.07], device=device)


    # raw position if the object is sync to the ground

    raw_position=torch.tensor([0, 0.07, 0], device=device)


    # 0.095 -0.08 0.1
    # Add external forces (gravity, etc.)
    gravity = ConstantForce(
        direction=torch.tensor([0, -1, 0]),
        magnitude= 10.0/ len(vertices[0]),
        device=device,
    )


    # For rendering
    renderer = SoftRenderer(camera_mode="look_at", device=device)
    camera_distance = 8.0
    elevation = 30.0
    azimuth = 0.0
    renderer.set_eye_from_angles(camera_distance, elevation, azimuth)

    # We'll store GT images + trajectory
    img_gt = []
    positions_gt = []
    orientations_gt = []



    #GT sim is from foundationpose


    xml_path = '/home/haozhe/Dropbox/physics/_data/allegro/wonik_allegro/scene_U.xml'
    control_signal_path = '/home/haozhe/Dropbox/physics/U_sequence/2/joint_states_log.csv'
    gt_path = '/home/haozhe/Dropbox/physics/U_sequence/path_full_up_2/target_U_20250417_1708.txt'

    # Load Allegro glove joint states
    control_signal = load_position_data(control_signal_path)
    period = control_signal.shape[0]
    
    # The experiment started at frame=236
    start_sync_frame = 236
    active_step = period - start_sync_frame
    print('active_step:', active_step)
    # -------------------------------
    # B) Load MJX Model & Data (for full sim, if needed)
    # -------------------------------
    mjx_model, mjx_data, mj_model, mj_data, body_id, joint_id, qpos_addr, num_dofs = load_mjx_model(xml_path)
    


    loaded_pose = load_observations(gt_path)
    from datetime import datetime
    def parse_iso_time(time_str):
        return datetime.fromisoformat(time_str)
    
    #2025-04-17T17:08:00.002546 for U 
    sync_str = "2025-04-17T17:08:00.002546"
    offset_vector = np.array([0.0, 0.0, -1.45])
    sync_with_real_vector = np.array([0.095, -0.08, 0.02])


    #2025-03-20T15:32:49.516000 for bluelego
    # sync_str = "2025-03-20T15:32:49.516000"
    # offset_vector = np.array([0.0, 0.0, 0.46])
    # sync_with_real_vector = np.array([0.095, -0.08, 0.02])

    sync_dt = parse_iso_time(sync_str)
    filtered_pose = [
        (ts_str, final_transform, pose)
        for (ts_str, final_transform, pose) in loaded_pose
        if parse_iso_time(ts_str) >= sync_dt
    ]

    # bluelego
    # MAX_FRAMES = 7
    # U seq2
    MAX_FRAMES = 10
    loaded_pose = filtered_pose[:MAX_FRAMES]
    print("\n=== Checking loaded_pose ===")
    for i, (timestamp, F, P) in enumerate(loaded_pose):
        print(f"Index {i}: Timestamp={timestamp}\nFinal Transform=\n{F}\nPose=\n{P}\n"
              "--------------------------------------------------------")
        
    # sync cordinate
    reference_transform = loaded_pose[0][1]
    reference_transform_inv = np.linalg.inv(reference_transform)
    reference_transform_inv[2, 3] *= -1

    # sync the scene of the absolute center gap


    # active sampel time, same as the dt we use in simulator
    n_samples = 600 # 1/0.002 *1.2 seconds for U case
    timestamps, upsampled_pos, upsampled_quat = process_and_interpolate(
        loaded_pose,
        reference_transform_inv, 
        offset_vector,
        sync_with_real_vector,
        n_samples=n_samples
    )
    
    old_position=np.ones_like(upsampled_pos)
    old_orientation=np.ones_like(upsampled_quat)
    for temp in range(len(upsampled_pos)):
        old_position[temp] = flip_back_position(upsampled_pos[temp])
        old_orientation[temp] = flip_back_quaternion(upsampled_quat[temp])
    
    positions_gt = torch.as_tensor(upsampled_pos, dtype=torch.float32, device=device)
    orientations_gt = torch.as_tensor(upsampled_quat, dtype=torch.float32, device=device)

    positions_gt= positions_gt * torch.tensor([-1.0,-1.0,1.0]).to(device)
    # -------------------------------
    # D) Leading & Trailing Samples (unchanged)
    # -------------------------------
    def prepend_samples(positions, quaternions, n):
        first_pos = positions[0]
        first_quat = quaternions[0]
        leading_pos = np.tile(first_pos[None, :], (n, 1))
        leading_quat = np.tile(first_quat[None, :], (n, 1))
        new_positions = np.concatenate([leading_pos, positions], axis=0)
        new_quaternions = np.concatenate([leading_quat, quaternions], axis=0)
        return new_positions, new_quaternions

    def append_samples(positions, quaternions, n):
        last_pos = positions[-1]
        last_quat = quaternions[-1]
        trailing_pos = np.tile(last_pos[None, :], (n, 1))
        trailing_quat = np.tile(last_quat[None, :], (n, 1))
        new_positions = np.concatenate([positions, trailing_pos], axis=0)
        new_quaternions = np.concatenate([quaternions, trailing_quat], axis=0)
        return new_positions, new_quaternions

    # leading_samples = 220
    # trailing_samples = 200
    # upsampled_pos, upsampled_quat = prepend_samples(upsampled_pos, upsampled_quat, leading_samples)
    # leading_timestamps = [timestamps[0]] * leading_samples
    # timestamps = leading_timestamps + timestamps
    # upsampled_pos, upsampled_quat = append_samples(upsampled_pos, upsampled_quat, trailing_samples)
    # trailing_timestamps = [timestamps[-1]] * trailing_samples
    # timestamps = timestamps + trailing_timestamps


    # sync simulation and real sampling for a same size simulation sequence, the simulation we have 10fps

    #so we need 1.2 seconds for U 
    total_frames = upsampled_pos.shape[0]
    print(f"After leading/trailing samples, total_frames={total_frames}")

    # upsampled_pos and upsampled_quat are the ground truth positions and orientations
    
        
    control_step = control_signal.shape[0]
    


    qpos_list, qacc_list, qfrc_actuator_list, contact_list, qfrc_smooth_list=run_simulation(mj_model, mj_data, control_signal)

    # forces_to_add = load_sim_data("force_list.txt")
    body_geom_id  = mj_model.geom("U").id
    plate_geom_id = mj_model.geom("plate").id



    impulse_frames, smooth_forces_per_frame, contact_positions_per_frame = process_actuation_force(contact_list ,
                                                                                                   qfrc_smooth_list, 
                                                                                                   start_sync_frame,
                                                                                                   body_geom_id,
                                                                                                   plate_geom_id)

    # 379-395

    # # query force from mjx

    # for force_obj, contact_points in forces_to_add:
    #         body_gt.add_external_force(force_obj, contact_points)


    # ------------------------------------------------------------------------
    # 3) Prepare an estimate model (initial masses)
    # ------------------------------------------------------------------------
    masses_est = torch.nn.Parameter(
        0.002 * torch.ones(vertices.shape[1], dtype=vertices.dtype, device=device),
        requires_grad=False,
    )
    # gt 0.00014

    uniform_density = True
    model = Model(masses_est, uniform_density=uniform_density).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossfn = torch.nn.MSELoss()

    # For logging
    losses = []
    initial_imgs = []
    est_masses = None

    # ------------------------------------------------------------------------
    # 4) Training loop: simulate with the updated mass, compute *trajectory* loss
    # ------------------------------------------------------------------------
    for epoch in trange(args.epochs, desc="Training"):
        masses_cur = model()  # [#vertices]

        # Create a new rigid body for the estimated scenario
        body_est = RigidBody(vertices[0], masses=masses_cur,position=raw_position)
        body_est.add_external_force(gravity)

        # for t in range(len(impulse_frames)):
        sample_step=5
        impulse_frame = impulse_frames[sample_step]
        smooth_forces_per_frame_t = smooth_forces_per_frame[sample_step]
            # contact_positions_per_frame_t = contact_positions_per_frame[t]
            
        impulse_magnitude=torch.tensor(smooth_forces_per_frame_t, device=device,dtype=torch.float32)
        force_obj = flip_matrix @ impulse_magnitude
            # contact_point=torch.tensor(contact_positions_per_frame_t, device=device,dtype=torch.float32).view(3)
            # contact_points = flip_matrix @ contact_point

        arr =vertices[0].clone().detach().cpu().numpy()

            # Compute the Euclidean norm of each row
        magnitudes = np.linalg.norm(arr, axis=1)  # shape (N,)

            # Index of the row with the largest norm
        max_idx = np.argmax(magnitudes)
        max_norm_val = magnitudes[max_idx]
        best_point = arr[max_idx]

        print("Row with largest norm:", max_idx)
        print("Coordinates:", best_point)
        print("Norm:", max_norm_val)
            
        contact_points= best_point
        

        # hack for 3 best 2 best point: best_point
            # placeholder
        time_step=0
            
            # for the lasting time, we should focus on the time when we want to apply on this simulation
            # for backward optimize
        force_obj[0]=force_obj[0]*vertices[0].shape[0]*10
        force_obj[2]=force_obj[2]*vertices[0].shape[0]*10
        
        impulse = ConstantForce(
                    magnitude=force_obj,
                    direction=torch.tensor([1,0,1]), # mask out z
                    starttime=0.0+time_step*dt,
                    endtime=0.0+time_step*dt+0.01,
                    device=device,
                )
        
        contact_points=flip_matrix_np @ contact_points
        
        # expand id from a single one to the neraset 1000 points because of the contact surface of hand
        id,dis=find_closest_vertices(vertices, contact_points,k=100)





        body_est.add_external_force(impulse, application_points=[id])
           
        # Build a new simulator for each epoch
        sim_est = Simulator([body_est],dtime=dt, contacts=True)

        # We'll collect the estimated trajectory
        positions_est = []
        orientations_est = []
        img_est = []

        # Run the sim
        # simsteps should be same as control signal
        simsteps= total_frames  
        # writer = imageio.get_writer(render_outfile, mode="I")

        # and the control signal will generate the sim_est.pre_step external force
        for t in range(simsteps):
            sim_est.step()
            # rgba = renderer.forward(body_est.get_world_vertices().unsqueeze(0), faces, textures)
            # add the position back to the normalize place
            # 0.095 -0.08 0.1 notice this offset vector
            positions_est.append(body_est.position)
            
            orientations_est.append(body_est.orientation)
        #     img = rgba[0].permute(1, 2, 0).detach().cpu().numpy()
        #     writer.append_data((255 * img).astype(np.uint8))
        # writer.close()
        # with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        #     i = 0
        #     n_frames = len(positions_est)

        #     while viewer.is_running() and i < n_frames:
                

        #         # The first 3 values are translation: x, y, z
        #         pos = positions_est[i]
        #         # The next 4 are quaternion: qw, qx, qy, qz
        #         quat = orientations_est[i]

        #         # Example: subtract offset from pos if needed
        #         # pos = pos - offset_vector

        #         pos_numpy=pos.clone()
        #         quat_numpy=quat.clone()
        #         pos_numpy=pos_numpy.detach().cpu().numpy()
        #         quat_numpy=quat_numpy.detach().cpu().numpy()


                

        #         pos_numpy=pos_numpy
        #         old_position = flip_back_position(pos_numpy)
        #         old_orientation = flip_back_quaternion(quat_numpy)

        #         # old_position = positions_gt[i].detach().cpu().numpy()
        #         # old_orientation = orientations_gt[i].detach().cpu().numpy()

        #         with viewer.lock():
        #             # Assign to the lego freejoint
        #             mj_data.qpos[qpos_addr : qpos_addr+3] = positions_gt[i].detach().cpu().numpy()
        #             mj_data.qpos[qpos_addr+3 : qpos_addr+7] = orientations_gt[i].detach().cpu().numpy()

        #             # Forward the new state
        #             mujoco.mj_forward(mj_model, mj_data)

        #         viewer.sync()

        #         print(f"[{i}] step={i}, pos={pos_numpy}, quat={quat_numpy}")
        #         print(f"[{i}] gt step={i}, pos={positions_gt[i].detach().cpu().numpy()}, quat={orientations_gt[i].detach().cpu().numpy()}")

        #         i += 1
        #         time.sleep(0.05)


        losses_ = []

        for t in range(simsteps):
            if t % args.compare_every == 0:
                pos_loss = lossfn(positions_est[t], positions_gt[t])
                orn_loss = lossfn(orientations_est[t], orientations_gt[t])
                losses_.append(pos_loss + orn_loss)
                # losses_.append(pos_loss )

        if len(losses_) == 0:
            total_loss = lossfn(positions_est[-1], positions_gt[-1])
        else:
            total_loss = sum(losses_) / len(losses_)

        tqdm.write(
            f"Epoch {epoch}: Loss = {total_loss.item():.5f}, "
            f"Mass(err) = {(masses_cur - masses_gt).abs().mean():.5f}"
        )

        losses.append(total_loss.item())
        est_masses = masses_cur.clone().detach().cpu().numpy()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()


        if epoch in [40, 80]:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.5

    # ------------------------------------------------------------------------
    # 5) Logging: save images and metrics if desired
    # ------------------------------------------------------------------------
    if args.log:
        logdir = Path(args.logdir) / args.expid
        import os
        os.makedirs(logdir, exist_ok=True)

        # Write out GT, EST, and initial images as GIF
        initwriter = imageio.get_writer(logdir / "init.gif", mode="I")
        gtwriter = imageio.get_writer(logdir / "gt.gif", mode="I")
        estwriter = imageio.get_writer(logdir / "est.gif", mode="I")

        for gtimg, initimg, estimg in zip(img_gt, initial_imgs, img_est):
            gt_np  = gtimg[0].permute(1, 2, 0).detach().cpu().numpy()
            init_np = initimg[0].permute(1, 2, 0).detach().cpu().numpy()
            est_np = estimg[0].permute(1, 2, 0).detach().cpu().numpy()

            gtwriter.append_data((255 * gt_np).astype(np.uint8))
            initwriter.append_data((255 * init_np).astype(np.uint8))
            estwriter.append_data((255 * est_np).astype(np.uint8))

        gtwriter.close()
        initwriter.close()
        estwriter.close()

        np.savetxt(logdir / "losses.txt", losses)
        np.savetxt(logdir / "masses.txt", est_masses)

        print(f"Logs saved to {logdir}")