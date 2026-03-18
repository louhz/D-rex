#!/usr/bin/env python3
"""
Bridge the object trajectory from D-Rex's mass_estimator_ketchup.py into RoboGS.

This script keeps the RoboGS scene/background/robot splats but replaces the object's
frame-to-frame pose with a trajectory reconstructed from the FoundationPose log and
trajectory processing used in D-Rex.

Main capabilities:
1. Parse the FoundationPose-style observation log used by D-Rex.
2. Reproduce the trajectory preprocessing from mass_estimator_ketchup.py.
3. Convert the resulting absolute poses into a relative object motion sequence.
4. Apply that motion to the object splats (semantic_id == object_semantic_id).
5. Optionally render the animated scene through RoboGS Runner.render_step.
6. Export pose files that can be reused to drive the object free joint in MuJoCo.

The default trajectory variant matches the tensors that are actually fed into the
trajectory loss in D-Rex ("used_gt"), not the unused debug buffers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import numpy as np


Q_FLIP_INV = np.array([0.7071068, 0.7071068, 0.0, 0.0], dtype=np.float64)


@dataclass
class TrajectoryBundle:
    mode: str
    dt: float
    timestamps_keyframes: List[str]
    positions_abs: np.ndarray  # [T, 3]
    quats_abs_wxyz: np.ndarray  # [T, 4]
    poses_abs: np.ndarray  # [T, 4, 4]
    poses_rel: np.ndarray  # [T, 4, 4]


def parse_iso_time(time_str: str) -> datetime:
    return datetime.fromisoformat(time_str)


def normalize_quaternion_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quaternion_multiply_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply q1 * q2 for wxyz quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def flip_back_position(pos_flipped: Sequence[float]) -> np.ndarray:
    x_new, y_new, z_new = pos_flipped
    x_old = x_new
    y_old = -z_new
    z_old = y_new
    return np.array([x_old, y_old, z_old], dtype=np.float64)


def flip_back_quaternion(q_flipped: Sequence[float]) -> np.ndarray:
    return normalize_quaternion_wxyz(quaternion_multiply_wxyz(Q_FLIP_INV, np.asarray(q_flipped, dtype=np.float64)))


def rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion in wxyz order."""
    rot = np.asarray(rot, dtype=np.float64)
    trace = np.trace(rot)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion_wxyz(np.array([w, x, y, z], dtype=np.float64))


def quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    quat = normalize_quaternion_wxyz(np.asarray(quat, dtype=np.float64))
    w, x, y, z = quat
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_rotmat_batch(quats: np.ndarray) -> np.ndarray:
    return np.stack([quat_wxyz_to_rotmat(q) for q in quats], axis=0)


def rotmat_to_quat_wxyz_batch(rotmats: np.ndarray) -> np.ndarray:
    return np.stack([rotmat_to_quat_wxyz(r) for r in rotmats], axis=0)


def make_pose_matrix(pos: Sequence[float], quat_wxyz: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = quat_wxyz_to_rotmat(np.asarray(quat_wxyz, dtype=np.float64))
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)
    return pose


def pose_matrix_to_pos_quat(pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return pose[:3, 3].copy(), rotmat_to_quat_wxyz(pose[:3, :3])


def load_observations(gt_path: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Copied/adapted from D-Rex mass_estimator_ketchup.py."""
    pose_list: List[Tuple[str, np.ndarray, np.ndarray]] = []
    with gt_path.open("r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if "Final transform:" not in line:
            i += 1
            continue

        match = re.match(r"^(.*?) - Final transform:", line)
        if not match:
            i += 1
            continue

        timestamp = match.group(1).strip()
        i += 1
        matrix_f: List[List[float]] = []
        for _ in range(4):
            if i >= len(lines):
                raise ValueError(f"Unexpected EOF while parsing transform in {gt_path}")
            row = list(map(float, lines[i].strip().strip("[]").split()))
            matrix_f.append(row)
            i += 1
        final_transform = np.array(matrix_f, dtype=np.float64)

        if i >= len(lines):
            break
        i += 1  # skip blank separator as in D-Rex parser

        pose_lines: List[str] = []
        while i < len(lines):
            pose_lines.append(lines[i])
            i += 1
            if "]]])" in pose_lines[-1]:
                break

        pose_str = "".join(pose_lines).replace("tensor(", "").rstrip(")")
        try:
            pose_data = ast.literal_eval(pose_str)
            if len(pose_data) == 1 and isinstance(pose_data[0], list) and len(pose_data[0]) == 4:
                pose_array = np.array(pose_data[0], dtype=np.float64)
            elif len(pose_data) == 1 and len(pose_data[0]) == 1:
                pose_array = np.array(pose_data[0][0], dtype=np.float64)
            else:
                pose_array = np.array(pose_data, dtype=np.float64)
        except Exception:
            pose_array = np.zeros((4, 4), dtype=np.float64)

        pose_list.append((timestamp, final_transform, pose_array))

    if not pose_list:
        raise ValueError(f"No 'Final transform' entries were parsed from {gt_path}")
    return pose_list


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quaternion_wxyz(q0)
    q1 = normalize_quaternion_wxyz(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q_lin = q0 + t * (q1 - q0)
        return normalize_quaternion_wxyz(q_lin)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return normalize_quaternion_wxyz(q0 * math.cos(theta) + q2 * math.sin(theta))


def interpolate_transforms(positions: np.ndarray, quaternions: np.ndarray, n_samples: int = 1080) -> Tuple[np.ndarray, np.ndarray]:
    """Copied/adapted from D-Rex mass_estimator_ketchup.py."""
    n = len(positions)
    if n < 2:
        raise ValueError("Need at least two keyframes to interpolate a trajectory.")
    original_times = np.linspace(0.0, 1.0, n)
    new_times = np.linspace(0.0, 1.0, n_samples)

    up_pos: List[np.ndarray] = []
    up_quat: List[np.ndarray] = []
    for t in new_times:
        idx = np.searchsorted(original_times, t)
        if idx == 0:
            p_new = positions[0].copy()
            q_new = quaternions[0].copy()
        elif idx >= n:
            p_new = positions[-1].copy()
            q_new = quaternions[-1].copy()
        else:
            t1, t2 = original_times[idx - 1], original_times[idx]
            alpha = float((t - t1) / (t2 - t1))
            p0, p1 = positions[idx - 1], positions[idx]
            q0, q1 = quaternions[idx - 1], quaternions[idx]
            p_new = (1.0 - alpha) * p0 + alpha * p1
            q_new = slerp_wxyz(q0, q1, alpha)
        up_pos.append(p_new)
        up_quat.append(q_new)
    return np.asarray(up_pos, dtype=np.float64), np.asarray(up_quat, dtype=np.float64)


def process_and_interpolate(
    loaded_pose: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    reference_transform_inv: np.ndarray,
    offset_vector: np.ndarray,
    sync_with_real_vector: np.ndarray,
    n_samples: int = 1080,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Copied/adapted from D-Rex mass_estimator_ketchup.py."""
    timestamps: List[str] = []
    final_transform_list: List[np.ndarray] = []
    for ts, final_transform_tmp, _ in loaded_pose:
        timestamps.append(ts)
        final_transform_list.append(final_transform_tmp)

    final_transform = np.asarray(final_transform_list, dtype=np.float64)
    final_transform_rel = reference_transform_inv[None] @ final_transform

    swap_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    flipped_transform = final_transform_rel @ swap_matrix
    positions = flipped_transform[:, :3, 3]
    quats = rotmat_to_quat_wxyz_batch(flipped_transform[:, :3, :3])
    positions = positions - offset_vector[None, :] + sync_with_real_vector[None, :]
    up_pos, up_quat = interpolate_transforms(positions, quats, n_samples=n_samples)
    return timestamps, up_pos, up_quat


def filter_loaded_pose(
    loaded_pose: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    sync_str: Optional[str],
    max_frames: Optional[int],
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    filtered = list(loaded_pose)
    if sync_str:
        sync_dt = parse_iso_time(sync_str)
        filtered = [(ts, f, p) for (ts, f, p) in filtered if parse_iso_time(ts) >= sync_dt]
    if max_frames is not None:
        filtered = filtered[: max_frames]
    if not filtered:
        raise ValueError("No frames remain after sync/max_frames filtering.")
    return filtered


def build_drex_trajectory(
    gt_path: Path,
    *,
    sync_str: Optional[str],
    offset_vector: np.ndarray,
    sync_with_real_vector: np.ndarray,
    max_frames: Optional[int],
    n_samples: int,
    dt: float,
    invert_reference_z_translation: bool,
    trajectory_variant: str,
    position_axis_sign: np.ndarray,
) -> TrajectoryBundle:
    loaded_pose = load_observations(gt_path)
    loaded_pose = filter_loaded_pose(loaded_pose, sync_str=sync_str, max_frames=max_frames)

    reference_transform = loaded_pose[0][1]
    reference_transform_inv = np.linalg.inv(reference_transform)
    if invert_reference_z_translation:
        reference_transform_inv[2, 3] *= -1.0

    timestamps_keyframes, upsampled_pos, upsampled_quat = process_and_interpolate(
        loaded_pose,
        reference_transform_inv,
        offset_vector=offset_vector,
        sync_with_real_vector=sync_with_real_vector,
        n_samples=n_samples,
    )

    if trajectory_variant == "used_gt":
        positions_abs = upsampled_pos.copy()
        positions_abs *= position_axis_sign[None, :]
        quats_abs = upsampled_quat.copy()
    elif trajectory_variant == "flip_back_debug":
        positions_abs = np.stack([flip_back_position(p) for p in upsampled_pos], axis=0)
        positions_abs *= position_axis_sign[None, :]
        quats_abs = np.stack([flip_back_quaternion(q) for q in upsampled_quat], axis=0)
    else:
        raise ValueError(f"Unknown trajectory variant: {trajectory_variant}")

    poses_abs = np.stack([make_pose_matrix(p, q) for p, q in zip(positions_abs, quats_abs)], axis=0)
    pose0_inv = np.linalg.inv(poses_abs[0])
    poses_rel = np.asarray([pose @ pose0_inv for pose in poses_abs], dtype=np.float64)

    return TrajectoryBundle(
        mode=trajectory_variant,
        dt=dt,
        timestamps_keyframes=timestamps_keyframes,
        positions_abs=positions_abs,
        quats_abs_wxyz=quats_abs,
        poses_abs=poses_abs,
        poses_rel=poses_rel,
    )


def add_repo_paths(robogs_root: Path) -> None:
    root = robogs_root.resolve()
    robogs_pkg = root / "robogs"
    for path in (str(root), str(robogs_pkg)):
        if path not in sys.path:
            sys.path.insert(0, path)


@dataclass
class RoboGSModules:
    load_ply_sam: object
    save_ply_sam: object
    sh_rotation_torch: object
    Runner: object
    Config: object


def import_robogs_modules(robogs_root: Path, need_runner: bool) -> RoboGSModules:
    add_repo_paths(robogs_root)
    try:
        from assign import load_ply_sam, save_ply_sam  # type: ignore
        from deform_util import sh_rotation_torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import error path
        raise RuntimeError(
            "Failed to import RoboGS modules. Confirm --robogs-root points to the repository root "
            "that contains the 'robogs/' package directory."
        ) from exc

    Runner = None
    Config = None
    if need_runner:
        try:
            from runner import Runner  # type: ignore
            from vis.gsplat_trainer import Config  # type: ignore
        except Exception as exc:  # pragma: no cover - import error path
            raise RuntimeError(
                "Rendering was requested but RoboGS Runner/Config could not be imported. "
                "Install the RoboGS environment first, then rerun."
            ) from exc

    return RoboGSModules(
        load_ply_sam=load_ply_sam,
        save_ply_sam=save_ply_sam,
        sh_rotation_torch=sh_rotation_torch,
        Runner=Runner,
        Config=Config,
    )


def export_trajectory_files(bundle: TrajectoryBundle, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "drex_object_trajectory.npz",
        dt=np.array(bundle.dt, dtype=np.float64),
        mode=np.array(bundle.mode),
        timestamps_keyframes=np.array(bundle.timestamps_keyframes, dtype=object),
        positions_abs=bundle.positions_abs,
        quats_abs_wxyz=bundle.quats_abs_wxyz,
        poses_abs=bundle.poses_abs,
        poses_rel=bundle.poses_rel,
    )

    with (output_dir / "drex_object_trajectory_rel.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "frame",
                "time_s",
                "tx",
                "ty",
                "tz",
                "qw",
                "qx",
                "qy",
                "qz",
            ]
        )
        for idx, pose in enumerate(bundle.poses_rel):
            pos, quat = pose_matrix_to_pos_quat(pose)
            writer.writerow([idx, idx * bundle.dt, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])

    meta = {
        "mode": bundle.mode,
        "dt": bundle.dt,
        "num_frames": int(bundle.poses_rel.shape[0]),
        "timestamps_keyframes": bundle.timestamps_keyframes,
    }
    with (output_dir / "drex_object_trajectory_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def maybe_export_sim_qpos(
    bundle: TrajectoryBundle,
    output_dir: Path,
    initial_sim_pose: Optional[np.ndarray],
) -> None:
    if initial_sim_pose is None:
        return
    if initial_sim_pose.shape != (7,):
        raise ValueError("initial_sim_pose must contain exactly 7 values: x y z qw qx qy qz")

    init_pose = make_pose_matrix(initial_sim_pose[:3], initial_sim_pose[3:])
    with (output_dir / "drex_object_sim_world_qpos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "time_s", "x", "y", "z", "qw", "qx", "qy", "qz"])
        for idx, rel_pose in enumerate(bundle.poses_rel):
            pose_world = rel_pose @ init_pose
            pos, quat = pose_matrix_to_pos_quat(pose_world)
            writer.writerow([idx, idx * bundle.dt, pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]])


def frame_indices(num_frames: int, start: int, stop: Optional[int], stride: int) -> List[int]:
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if stop is None:
        stop = num_frames
    start = max(0, start)
    stop = min(num_frames, stop)
    return list(range(start, stop, stride))


def apply_pose_to_object_splats(
    *,
    xyz_obj: np.ndarray,
    rots_obj_wxyz: np.ndarray,
    fextra_obj: np.ndarray,
    fdc_obj: np.ndarray,
    rel_pose: np.ndarray,
    rotate_sh: bool,
    sh_rotation_torch,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    rel_rot = rel_pose[:3, :3].astype(np.float32)
    rel_t = rel_pose[:3, 3].astype(np.float32)

    xyz_out = (xyz_obj.astype(np.float32) @ rel_rot.T) + rel_t[None, :]

    obj_rotmats = quat_wxyz_to_rotmat_batch(rots_obj_wxyz.astype(np.float64)).astype(np.float32)
    rot_out = np.matmul(rel_rot[None, :, :], obj_rotmats)
    rots_out = rotmat_to_quat_wxyz_batch(rot_out.astype(np.float64)).astype(np.float32)

    if rotate_sh:
        fextra_out = (
            sh_rotation_torch(
                torch.as_tensor(fextra_obj, dtype=torch.float32),
                torch.as_tensor(fdc_obj, dtype=torch.float32),
                torch.as_tensor(rel_rot, dtype=torch.float32),
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        fextra_out = fextra_obj.astype(np.float32, copy=True)

    return xyz_out, rots_out, fextra_out


@dataclass
class SceneArrays:
    xyz: np.ndarray
    features_dc: np.ndarray
    features_extra: np.ndarray
    opacities: np.ndarray
    scales: np.ndarray
    rots: np.ndarray
    semantic_id: np.ndarray


class RoboGSRenderer:
    def __init__(
        self,
        modules: RoboGSModules,
        *,
        data_dir: Path,
        result_dir: Path,
        camera_index: int,
        data_factor: int,
        test_every: int,
    ) -> None:
        import torch

        cfg = modules.Config(
            disable_viewer=True,
            data_dir=str(data_dir),
            data_factor=data_factor,
            result_dir=str(result_dir),
            test_every=test_every,
        )
        self.runner = modules.Runner(0, 0, 1, cfg)
        self.device = self.runner.device
        self.data = self.runner.trainset[camera_index]
        self.Ks = self.data["K"].to(self.device).view(1, 3, 3)
        image = self.data["image"]
        self.height, self.width = int(image.shape[0]), int(image.shape[1])
        self.masks = self.data.get("mask", None)
        if self.masks is not None:
            self.masks = self.masks.to(self.device)

        camtoworld = self.data.get("camtoworld", None)
        if camtoworld is not None:
            self.camtoworld = camtoworld.view(1, 4, 4).to(self.device)
        else:
            raise RuntimeError(
                "RoboGS dataset sample does not contain 'camtoworld'. "
                "Use the same data directory you used for RoboGS training."
            )

        # Warm up CUDA state for more predictable first frame timing.
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def render_frame(self, scene: SceneArrays, frame_idx: int, out_dir: Path) -> Path:
        fdc_render = np.transpose(scene.features_dc, (0, 2, 1)).astype(np.float32)
        self.runner.render_step(
            scene.xyz.astype(np.float32),
            scene.rots.astype(np.float32),
            scene.scales.astype(np.float32),
            scene.opacities.astype(np.float32),
            fdc_render,
            scene.features_extra.astype(np.float32),
            self.device,
            self.camtoworld,
            self.Ks,
            self.width,
            self.height,
            self.masks,
            self.runner.cfg,
            stage="drex_object",
            step=frame_idx,
            frame=frame_idx,
            out_path=str(out_dir),
        )
        return out_dir / f"{frame_idx}_drex_object_step{frame_idx:03d}.png"


def build_scene_frame(
    base_scene: SceneArrays,
    obj_mask: np.ndarray,
    rel_pose: np.ndarray,
    *,
    rotate_sh: bool,
    sh_rotation_torch,
) -> SceneArrays:
    xyz_obj = base_scene.xyz[obj_mask]
    rots_obj = base_scene.rots[obj_mask]
    fextra_obj = base_scene.features_extra[obj_mask]
    fdc_obj = base_scene.features_dc[obj_mask]

    xyz_obj_out, rots_obj_out, fextra_obj_out = apply_pose_to_object_splats(
        xyz_obj=xyz_obj,
        rots_obj_wxyz=rots_obj,
        fextra_obj=fextra_obj,
        fdc_obj=fdc_obj,
        rel_pose=rel_pose,
        rotate_sh=rotate_sh,
        sh_rotation_torch=sh_rotation_torch,
    )

    xyz = base_scene.xyz.astype(np.float32, copy=True)
    rots = base_scene.rots.astype(np.float32, copy=True)
    features_extra = base_scene.features_extra.astype(np.float32, copy=True)

    xyz[obj_mask] = xyz_obj_out
    rots[obj_mask] = rots_obj_out
    features_extra[obj_mask] = fextra_obj_out

    return SceneArrays(
        xyz=xyz,
        features_dc=base_scene.features_dc.astype(np.float32, copy=True),
        features_extra=features_extra,
        opacities=base_scene.opacities.astype(np.float32, copy=True),
        scales=base_scene.scales.astype(np.float32, copy=True),
        rots=rots,
        semantic_id=base_scene.semantic_id.astype(np.float32, copy=True),
    )


def write_video_from_pngs(render_dir: Path, video_path: Path, fps: int) -> None:
    def sort_key(path: Path) -> Tuple[int, str]:
        match = re.match(r"^(\d+)_", path.name)
        frame_idx = int(match.group(1)) if match else 10**9
        return frame_idx, path.name

    frames = sorted(render_dir.glob("*.png"), key=sort_key)
    if not frames:
        raise RuntimeError(f"No PNG frames were found in {render_dir} to assemble a video.")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the D-Rex ketchup trajectory to drive the object splats inside a RoboGS scene."
    )
    parser.add_argument("--robogs-root", type=Path, required=True, help="Path to the RoboGS repository root.")
    parser.add_argument("--ply-file", type=Path, required=True, help="Path to RoboGS final_scene_with_ids.ply.")
    parser.add_argument(
        "--gt-path",
        type=Path,
        required=True,
        help="Path to the FoundationPose/observation log used by D-Rex (e.g. target_ketchup_*.txt).",
    )
    parser.add_argument("--result-dir", type=Path, required=True, help="Output directory for poses, renders, and optional PLYs.")

    parser.add_argument("--object-semantic-id", type=int, default=15, help="Semantic ID of the manipulated object in final_scene_with_ids.ply.")
    parser.add_argument(
        "--trajectory-variant",
        choices=["used_gt", "flip_back_debug"],
        default="used_gt",
        help=(
            "'used_gt' matches the actual D-Rex trajectory tensors used in the loss. "
            "'flip_back_debug' uses the debug flip-back buffers instead."
        ),
    )
    parser.add_argument("--sync-str", type=str, default="2025-04-28T17:45:38.996320", help="D-Rex sync timestamp for the ketchup example.")
    parser.add_argument("--offset-vector", type=float, nargs=3, default=[0.0, 0.0, 0.75], metavar=("X", "Y", "Z"))
    parser.add_argument(
        "--sync-with-real-vector",
        type=float,
        nargs=3,
        default=[0.095, -0.08, 0.02],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--max-frames", type=int, default=16, help="Number of keyframes to keep before interpolation.")
    parser.add_argument("--n-samples", type=int, default=600, help="Number of interpolated trajectory frames.")
    parser.add_argument("--dt", type=float, default=0.002, help="Frame dt used for exported time stamps.")
    parser.add_argument(
        "--position-axis-sign",
        type=float,
        nargs=3,
        default=[-1.0, -1.0, 1.0],
        metavar=("SX", "SY", "SZ"),
        help="Per-axis sign correction applied to positions after interpolation (matches D-Rex default for ketchup).",
    )
    parser.add_argument(
        "--no-invert-reference-z-translation",
        action="store_true",
        help="Disable the reference_transform_inv[2,3] *= -1 step from D-Rex.",
    )

    parser.add_argument("--start", type=int, default=0, help="First interpolated frame to export/render.")
    parser.add_argument("--stop", type=int, default=None, help="Exclusive stop frame. Default: all frames.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for export/render.")

    parser.add_argument(
        "--export-scene-ply-every",
        type=int,
        default=0,
        help=(
            "If > 0, export a full scene PLY every N selected frames. Disabled by default because full-scene PLY sequences are large."
        ),
    )
    parser.add_argument(
        "--disable-sh-rotation",
        action="store_true",
        help="Do not rotate SH coefficients with the object. Faster, but slightly less faithful visually.",
    )

    parser.add_argument("--initial-sim-pose", type=float, nargs=7, default=None, metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"))

    parser.add_argument("--render", action="store_true", help="Render the selected frames through RoboGS Runner.render_step.")
    parser.add_argument(
        "--robogs-data-dir",
        type=Path,
        default=None,
        help="RoboGS data_dir used for training / camera intrinsics / camera poses. Required when --render is set.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Dataset camera index to use for rendering.")
    parser.add_argument("--data-factor", type=int, default=4, help="RoboGS data_factor for the dataset parser.")
    parser.add_argument("--test-every", type=int, default=8, help="RoboGS test_every passed into Config.")
    parser.add_argument("--write-video", action="store_true", help="Assemble rendered PNGs into an MP4 after rendering.")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for the assembled MP4.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_drex_trajectory(
        args.gt_path,
        sync_str=args.sync_str,
        offset_vector=np.asarray(args.offset_vector, dtype=np.float64),
        sync_with_real_vector=np.asarray(args.sync_with_real_vector, dtype=np.float64),
        max_frames=args.max_frames,
        n_samples=args.n_samples,
        dt=args.dt,
        invert_reference_z_translation=not args.no_invert_reference_z_translation,
        trajectory_variant=args.trajectory_variant,
        position_axis_sign=np.asarray(args.position_axis_sign, dtype=np.float64),
    )

    export_trajectory_files(bundle, args.result_dir)
    initial_sim_pose = None if args.initial_sim_pose is None else np.asarray(args.initial_sim_pose, dtype=np.float64)
    maybe_export_sim_qpos(bundle, args.result_dir, initial_sim_pose)

    selected_frames = frame_indices(bundle.poses_rel.shape[0], args.start, args.stop, args.stride)
    if not selected_frames:
        raise RuntimeError("No frames selected. Check --start/--stop/--stride.")

    modules = import_robogs_modules(args.robogs_root, need_runner=args.render)
    xyz, features_dc, features_extra, opacities, scales, rots, semantic_id = modules.load_ply_sam(str(args.ply_file))

    scene = SceneArrays(
        xyz=np.asarray(xyz, dtype=np.float32),
        features_dc=np.asarray(features_dc, dtype=np.float32),
        features_extra=np.asarray(features_extra, dtype=np.float32),
        opacities=np.asarray(opacities, dtype=np.float32),
        scales=np.asarray(scales, dtype=np.float32),
        rots=np.asarray(rots, dtype=np.float32),
        semantic_id=np.asarray(semantic_id, dtype=np.float32),
    )

    obj_mask = scene.semantic_id.reshape(-1) == float(args.object_semantic_id)
    if not np.any(obj_mask):
        raise RuntimeError(
            f"No splats were found with semantic_id == {args.object_semantic_id}. "
            "Double-check your final_scene_with_ids.ply and the object semantic ID."
        )

    render_dir = args.result_dir / "renders"
    ply_dir = args.result_dir / "scene_ply_sequence"
    if args.render:
        if args.robogs_data_dir is None:
            raise RuntimeError("--robogs-data-dir is required when --render is used.")
        render_dir.mkdir(parents=True, exist_ok=True)
        renderer = RoboGSRenderer(
            modules,
            data_dir=args.robogs_data_dir,
            result_dir=args.result_dir / "robogs_runtime",
            camera_index=args.camera_index,
            data_factor=args.data_factor,
            test_every=args.test_every,
        )
    else:
        renderer = None

    export_every = int(args.export_scene_ply_every)
    rotate_sh = not args.disable_sh_rotation

    frame_manifest: List[Dict[str, object]] = []

    for selected_idx, frame_idx in enumerate(selected_frames):
        scene_frame = build_scene_frame(
            scene,
            obj_mask,
            bundle.poses_rel[frame_idx],
            rotate_sh=rotate_sh,
            sh_rotation_torch=modules.sh_rotation_torch,
        )

        rel_pos, rel_quat = pose_matrix_to_pos_quat(bundle.poses_rel[frame_idx])
        frame_record: Dict[str, object] = {
            "frame": frame_idx,
            "time_s": frame_idx * bundle.dt,
            "rel_translation": rel_pos.tolist(),
            "rel_quaternion_wxyz": rel_quat.tolist(),
        }

        if export_every > 0 and (selected_idx % export_every == 0):
            ply_dir.mkdir(parents=True, exist_ok=True)
            ply_path = ply_dir / f"scene_frame_{frame_idx:04d}.ply"
            modules.save_ply_sam(
                xyz=scene_frame.xyz,
                f_dc=scene_frame.features_dc,
                f_rest=scene_frame.features_extra,
                opacities=scene_frame.opacities,
                semantic_id=scene_frame.semantic_id,
                scale=scene_frame.scales,
                rotation=scene_frame.rots,
                path=str(ply_path),
            )
            frame_record["scene_ply"] = str(ply_path)

        if renderer is not None:
            frame_png = renderer.render_frame(scene_frame, frame_idx, render_dir)
            frame_record["render_png"] = str(frame_png)

        frame_manifest.append(frame_record)

    with (args.result_dir / "frame_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(frame_manifest, f, indent=2)

    if args.render and args.write_video:
        video_path = args.result_dir / "drex_robogs_render.mp4"
        write_video_from_pngs(render_dir, video_path, fps=args.video_fps)

    with (args.result_dir / "run_args.json").open("w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, f, indent=2)

    print("Done.")
    print(f"Results written to: {args.result_dir}")
    print(f"Selected frames: {len(selected_frames)} / {bundle.poses_rel.shape[0]}")
    print(f"Object semantic id: {args.object_semantic_id}")
    print(f"Trajectory mode: {bundle.mode}")


if __name__ == "__main__":
    main()
