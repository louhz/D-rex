import os
import math
import numpy as np
import torch

from curobo.geom.types import Cuboid, Mesh, WorldConfig
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)


import open3d as o3d
import matplotlib.pyplot as plt







def load_mesh_from_file(file_path):
    # Load the main mesh from file
    mesh = o3d.io.read_triangle_mesh(file_path)
    # mesh.compute_vertex_normals()  # If you need normals; optional

    # Get the AABB (axis-aligned bounding box) of the mesh
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()  # [width, height, depth]

    # Create a cuboid matching the bounding box dimensions
    cuboid = o3d.geometry.TriangleMesh.create_box(
        width=extent[0],
        height=extent[1],
        depth=extent[2]
    )

    # By default, create_box() has its lower-left corner at (0,0,0).
    # We want its center to align with the bounding box center:
    cuboid_center = cuboid.get_center()
    translation = center - cuboid_center
    cuboid.translate(translation)

    return mesh, cuboid,translation

def deg_to_rad(deg):
    return deg * math.pi / 180.0

def quaternion_about_axis(angle_rad, axis):
    ax, ay, az = axis
    half_angle = angle_rad / 2.0
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)
    qx = ax * sin_half
    qy = ay * sin_half
    qz = az * sin_half
    qw = cos_half
    return [qw, qx, qy, qz]

# Output directory setup
output_dir = "trajectories"
os.makedirs(output_dir, exist_ok=True)

# Define table mesh and world
table_path='/home/haozhe/Desktop/robotic_toolset/dextwin/curobo_asset/table_downsample_reori.stl'
q_90_x = quaternion_about_axis(deg_to_rad(90), [1, 0, 0])
table = Mesh(
    name="table",
    file_path=table_path,
    pose=[0, 0, 0.15, 1,0,0,0],
    scale=[1.0, 1.0, 1.0],
    color=[0.2, 0.6, 0.8, 1.0],
)


U_object_file='/home/haozhe/Dropbox/real_deployment/seq1/U_real.stl'
mesh, cuboid,translation=load_mesh_from_file(U_object_file)


box = Mesh(
    name="box",
    file_path=table_path,
    pose=[1, 1, 0, 1,0,0,0],
    scale=[1.0, 1.0, 1.0],
    color=[0.2, 0.6, 0.8, 1.0],
)

world_model = WorldConfig(  
    mesh=[box],
)


# Initialize cuRobo motion generator
motion_gen_config = MotionGenConfig.load_from_robot_config(
    "franka_allegro_right.yml",
    world_model,
    interpolation_dt=0.01,
)
motion_gen = MotionGen(motion_gen_config)
motion_gen.warmup()
print("Finished MotionGen warmup.")






# Sampling points
def sample_upper_hemisphere_points(n_points=60, center=(0.0, 0.0, 0.0), radius=1.0, max_angle_deg=60):
    x_center, y_center, z_center = center
    max_angle_rad = np.deg2rad(max_angle_deg)
    cos_min = np.cos(max_angle_rad)
    samples = []

    for _ in range(n_points):
        phi = 2.0 * np.pi * np.random.rand()
        u = cos_min + (1.0 - cos_min) * np.random.rand()
        theta = np.arccos(u)
        x_local = radius * np.sin(theta) * np.cos(phi)
        y_local = radius * np.sin(theta) * np.sin(phi)
        z_local = radius * np.cos(theta)
        x = x_local + x_center
        y = y_local + y_center
        z = z_local + z_center
        yaw = np.arctan2(y_local, x_local)
        pitch = np.arctan2(-z_local, np.sqrt(x_local**2 + y_local**2))
        roll = 0.0
        samples.append((x, y, z, roll, pitch, yaw))
    return samples

def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qw, qx, qy, qz)

# Sample points on sphere cap


# Initial joint state (Franka)
deg=np.array([-3, -39, 3, -150, 5, 113, -15])
initial_state=torch.tensor(np.deg2rad(deg)).float().cuda()
current_state = JointState.from_position(
    initial_state.view(1,7).cuda(),
    joint_names=[
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
)







print(translation)

# Main planning loop
# fix the coordinate
# fix the scale


        # Main planning loop
        # fix the coordinate
        # fix the scale
translation_new= -translation

        # x,y,z,qw,qx,qy,qz = translation_new[0],0.05,translation_new[2]+0.05,0.5,0.5,0.5,0.5

x,y,z,qw,qx,qy,qz = translation_new[0]-0.2,0,translation_new[2],0,0,0.78,0.78
# from the object pose and a learned relationship from mcc-ho




goal_pose = Pose.from_list([x, y, z, qw, qx, qy, qz])
result = motion_gen.plan_single(current_state, goal_pose, MotionGenPlanConfig(max_attempts=40))

if result.success:
        
        traj = result.get_interpolated_plan()
        if isinstance(traj, JointState):
            joint_traj = traj.position.cpu().numpy()  # (T, 7)

            # Save trajectory to text
            traj_filename = os.path.join(output_dir, f"traj_{0:02d}.txt")
            np.savetxt(traj_filename, joint_traj, fmt="%.6f")

            # Save pose to text
            pose_filename = os.path.join(output_dir, f"pose_{0:02d}.txt")
            with open(pose_filename, "w") as f:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}")

            # Update current state to final joint position
            last_joint_state = traj[-1].position.unsqueeze(0)
            current_state = JointState.from_position(
                last_joint_state,
                joint_names=[
                    "panda_joint1",
                    "panda_joint2",
                    "panda_joint3",
                    "panda_joint4",
                    "panda_joint5",
                    "panda_joint6",
                    "panda_joint7",
                ],
            )
else:
        print(f"Point {1} planning failed. Skipping this point.")

print("Done planning and saving successful trajectories.")
