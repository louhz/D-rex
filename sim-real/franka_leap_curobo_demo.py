#!/usr/bin/env python3
"""CuRobo collision checking + motion generation for the Franka+LEAP robot.

This script is written to work with the robot YAMLs generated alongside it:
  - franka_leap_arm_only.yml   -> plan only the 7 Franka joints; LEAP joints fixed by lock_joints
  - franka_leap_full.yml       -> plan all 23 joints

Typical workflow:
  1. Convert your Franka+LEAP model to a URDF that keeps the same link/joint names as the MJCF.
  2. Pass that URDF path with --urdf.
  3. Use --robot-yaml franka_leap_arm_only.yml if you want arm-only reaching.
  4. Provide a start configuration and a goal pose [x y z qw qx qy qz].

Examples:
  python franka_leap_curobo_demo.py \
      --robot-yaml franka_leap_arm_only.yml \
      --urdf /abs/path/to/franka_leap.urdf \
      --start-joints 0 -1.3 0 -2.5 0 1.0 0 \
      --goal-pose 0.45 0.0 0.25 1 0 0 0

  python franka_leap_curobo_demo.py \
      --robot-yaml franka_leap_full.yml \
      --urdf /abs/path/to/franka_leap.urdf \
      --start-joints 0 -1.3 0 -2.5 0 1.0 0  0 0 0 0  0 0 0 0  0 0 0 0  0 0 0 0 \
      --goal-pose 0.45 0.0 0.25 1 0 0 0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def patch_robot_cfg_paths(robot_cfg_file: str | Path, urdf_path: str | Path, asset_root: str | Path | None = None) -> dict[str, Any]:
    cfg = load_yaml_file(robot_cfg_file)
    kin = cfg["robot_cfg"]["kinematics"]
    urdf_path = str(Path(urdf_path).resolve())
    if asset_root is None:
        asset_root = str(Path(urdf_path).resolve().parent)
    else:
        asset_root = str(Path(asset_root).resolve())
    kin["urdf_path"] = urdf_path
    kin["asset_root_path"] = asset_root

    # If the collision sphere file is a relative path, resolve it relative to the robot yaml file.
    sphere_ref = kin.get("collision_spheres", "")
    if isinstance(sphere_ref, str) and sphere_ref and not Path(sphere_ref).is_absolute():
        kin["collision_spheres"] = str((Path(robot_cfg_file).resolve().parent / sphere_ref).resolve())
    return cfg


def build_world(world_yaml: str | Path | None) -> dict[str, Any]:
    if world_yaml is None:
        return {
            "cuboid": {
                "table": {
                    "dims": [1.2, 1.2, 0.08],
                    "pose": [0.45, 0.0, -0.04, 1.0, 0.0, 0.0, 0.0],
                },
                "back_wall": {
                    "dims": [0.05, 1.2, 1.0],
                    "pose": [0.85, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0],
                },
                "left_post": {
                    "dims": [0.08, 0.08, 0.30],
                    "pose": [0.45, 0.25, 0.15, 1.0, 0.0, 0.0, 0.0],
                },
            }
        }
    return load_yaml_file(world_yaml)


def build_robot_world(robot_cfg_dict: dict[str, Any], world_cfg: dict[str, Any], tensor_args: TensorDeviceType) -> RobotWorld:
    config = RobotWorldConfig.load_from_config(
        robot_cfg_dict,
        world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_activation_distance=0.01,
        self_collision_activation_distance=0.0,
    )
    return RobotWorld(config)


def build_motion_gen(robot_cfg_dict: dict[str, Any], world_cfg: dict[str, Any], tensor_args: TensorDeviceType) -> MotionGen:
    mg_cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg_dict,
        world_cfg,
        tensor_args=tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=0.02,
        trajopt_tsteps=32,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        evaluate_interpolated_trajectory=True,
        collision_cache={"obb": 32, "mesh": 16},
    )
    motion_gen = MotionGen(mg_cfg)
    # Current releases use parallel_finetune by default; keeping it explicit is helpful for repeatability.
    motion_gen.warmup(enable_graph=True, parallel_finetune=True)
    return motion_gen


def joint_state_from_list(q: list[float], joint_names: list[str], tensor_args: TensorDeviceType) -> JointState:
    q_tensor = tensor_args.to_device([q])
    return JointState.from_position(q_tensor, joint_names=joint_names)


def ee_pose_from_joint_state(motion_gen: MotionGen, js: JointState) -> Pose:
    state = motion_gen.compute_kinematics(js)
    return Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())


def collision_report(robot_world: RobotWorld, js: JointState) -> dict[str, Any]:
    d_world, d_self = robot_world.get_world_self_collision_distance_from_joints(js.position)
    max_world = float(torch.max(d_world).item())
    max_self = float(torch.max(d_self).item())
    return {
        "world_penetration": max_world,
        "self_penetration": max_self,
        "in_world_collision": max_world > 0.0,
        "in_self_collision": max_self > 0.0,
    }


def trajectory_collision_report(robot_world: RobotWorld, traj: JointState) -> dict[str, Any]:
    q = traj.position.unsqueeze(0)  # [1, T, dof]
    d_world, d_self = robot_world.get_world_self_collision_distance_from_joint_trajectory(q)
    max_world = float(torch.max(d_world).item())
    max_self = float(torch.max(d_self).item())
    return {
        "traj_world_penetration": max_world,
        "traj_self_penetration": max_self,
        "traj_in_world_collision": max_world > 0.0,
        "traj_in_self_collision": max_self > 0.0,
    }


def plan_to_goal_pose(
    motion_gen: MotionGen,
    robot_world: RobotWorld,
    start_state: JointState,
    goal_pose_list: list[float],
) -> tuple[Any, JointState | None, dict[str, Any], dict[str, Any]]:
    start_col = collision_report(robot_world, start_state)
    goal_pose = Pose.from_list(goal_pose_list)  # [x, y, z, qw, qx, qy, qz]
    plan_cfg = MotionGenPlanConfig(
        max_attempts=8,
        enable_graph=True,
        enable_opt=True,
        parallel_finetune=True,
    )
    result = motion_gen.plan_single(start_state, goal_pose, plan_cfg)
    success = bool(result.success.item()) if hasattr(result.success, 'item') else bool(result.success)
    if not success:
        return result, None, start_col, {}

    traj = result.get_interpolated_plan()
    traj_col = trajectory_collision_report(robot_world, traj)
    return result, traj, start_col, traj_col


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-yaml", required=True, type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--asset-root", default=None, type=Path)
    parser.add_argument("--world-yaml", default=None, type=Path)
    parser.add_argument(
        "--start-joints",
        required=True,
        nargs="+",
        type=float,
        help="Start joints. Use 7 values for franka_leap_arm_only.yml, 23 for franka_leap_full.yml.",
    )
    parser.add_argument(
        "--goal-pose",
        required=True,
        nargs=7,
        type=float,
        metavar=("x", "y", "z", "qw", "qx", "qy", "qz"),
        help="Goal end-effector pose in CuRobo order.",
    )
    args = parser.parse_args()

    setup_curobo_logger("warn")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    tensor_args = TensorDeviceType()
    robot_cfg_dict = patch_robot_cfg_paths(args.robot_yaml, args.urdf, args.asset_root)
    world_cfg = build_world(args.world_yaml)

    joint_names = robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
    if len(args.start_joints) != len(joint_names):
        raise ValueError(
            f"Expected {len(joint_names)} start joints for {args.robot_yaml.name}, "
            f"got {len(args.start_joints)}."
        )

    start_state = joint_state_from_list(args.start_joints, joint_names, tensor_args)

    robot_world = build_robot_world(robot_cfg_dict, world_cfg, tensor_args)
    motion_gen = build_motion_gen(robot_cfg_dict, world_cfg, tensor_args)

    start_pose = ee_pose_from_joint_state(motion_gen, start_state)
    print("Start ee pose [x y z qw qx qy qz]:")
    print(
        [
            *start_pose.position.view(-1).detach().cpu().numpy().tolist(),
            *start_pose.quaternion.view(-1).detach().cpu().numpy().tolist(),
        ]
    )

    result, traj, start_col, traj_col = plan_to_goal_pose(
        motion_gen,
        robot_world,
        start_state,
        args.goal_pose,
    )

    print("Start-state collision report:")
    print(start_col)
    success = bool(result.success.item()) if hasattr(result.success, "item") else bool(result.success)
    print("Planning success:", success)
    print("Planner status:", getattr(result, "status", ""))
    print("Interpolation dt:", float(getattr(result, "interpolation_dt", 0.0)))

    if traj is None:
        return

    print("Trajectory collision report:")
    print(traj_col)
    print("Trajectory joint names:")
    print(traj.joint_names)
    print("Trajectory positions shape [T, dof]:", list(traj.position.shape))
    print("First waypoint:", traj.position[0].detach().cpu().numpy().tolist())
    print("Last waypoint:", traj.position[-1].detach().cpu().numpy().tolist())


if __name__ == "__main__":
    main()
