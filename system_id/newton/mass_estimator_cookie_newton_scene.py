"""
Newton full-scene rewrite of `mass_estimator_cookie.py`.

What changes versus the original D-rex script
---------------------------------------------
1. The MJCF / URDF scene is loaded directly with Newton (`ModelBuilder.add_mjcf`
   or `add_urdf`) instead of MuJoCo.
2. The hand / robot control replay is driven through Newton joint targets instead
   of `mj_data.ctrl`.
3. The object rollout and backward pass are done in the same Newton simulation
   with `SolverSemiImplicit` + `CollisionPipeline(requires_grad=True)`, so the
   separate MuJoCo contact-prepass and the separate gradsim object-only rollout
   are removed.
4. The trainable cookie mass is still parameterized over mesh vertices, but each
   forward pass reduces the vertex masses to rigid-body mass / COM / inertia and
   writes them into the imported Newton body before simulating.

Notes
-----
- This script is intentionally conservative about scene paths and preprocessing:
  the default data paths, GT parsing, synchronization, and interpolation mirror
  the original file as closely as possible.
- The control replay assumes the CSV columns map to the first scalar controlled
  DOFs in the imported articulation, which matches the original script's usage
  of `mj_data.ctrl[:] = action` for an Allegro-hand scene. Use the offset args
  below if your imported Newton model orders controlled DOFs differently.
- Newton quaternions use xyzw order, while the original MuJoCo preprocessing
  yields wxyz; the script converts them before computing pose loss.
- The script targets Newton builds with the differentiable rigid-contact path
  introduced around PR #2164.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pandas as pd
import warp
import warp as wp
import warp.optim

import newton


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DT_DEFAULT = 0.002
MASS_EPS = 1.0e-8
INERTIA_REG_EPS = 1.0e-6


# -----------------------------------------------------------------------------
# Data loading / preprocessing from the original script
# -----------------------------------------------------------------------------
def load_position_data(csv_path: str | Path) -> np.ndarray:
    position_cols = [f"position_{i}" for i in range(16)]
    df = pd.read_csv(csv_path, usecols=position_cols)
    return df.to_numpy(dtype=np.float32)


def load_observations(gt_path: str | Path):
    import re

    pose_list = []
    with open(gt_path, "r") as f:
        lines = f.read().strip().split("\n")

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
            matrix_f = []
            for _ in range(4):
                mat_line = lines[i].strip()
                row = mat_line.strip("[]").split()
                row = list(map(float, row))
                matrix_f.append(row)
                i += 1
            final_transform = np.array(matrix_f, dtype=np.float64)
            if i >= len(lines):
                break
            i += 1
            pose_lines = []
            while i < len(lines):
                pose_lines.append(lines[i])
                i += 1
                if "]]])" in pose_lines[-1]:
                    break
            pose_str = "".join(pose_lines)
            pose_str = pose_str.replace("tensor(", "").rstrip(")")
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
        else:
            i += 1
    return pose_list


def transform_to_pos_quat_batch(transforms_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert homogeneous transforms to position + quaternion.

    The original script used MuJoCo's `mju_mat2Quat`, which returns quaternions
    in wxyz order. We keep that order here and convert to Newton xyzw later.
    """
    pos = transforms_4x4[:, :3, 3].astype(np.float32)
    quat_wxyz = np.zeros((transforms_4x4.shape[0], 4), dtype=np.float32)

    for i, R in enumerate(transforms_4x4[:, :3, :3]):
        trace = np.trace(R)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        q /= max(np.linalg.norm(q), 1.0e-8)
        quat_wxyz[i] = q
    return pos, quat_wxyz


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q_lin = q0 + t * (q1 - q0)
        return q_lin / np.linalg.norm(q_lin)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def interpolate_transforms(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    n_samples: int = 1080,
) -> tuple[np.ndarray, np.ndarray]:
    n_keyframes = len(positions)
    if n_keyframes < 2:
        raise ValueError("Need at least two keyframes.")

    original_times = np.linspace(0.0, 1.0, n_keyframes)
    new_times = np.linspace(0.0, 1.0, n_samples)

    up_pos = []
    up_quat = []
    for t in new_times:
        idx = np.searchsorted(original_times, t)
        if idx == 0:
            p_new = positions[0].copy()
            q_new = quaternions_wxyz[0].copy()
        elif idx >= n_keyframes:
            p_new = positions[-1].copy()
            q_new = quaternions_wxyz[-1].copy()
        else:
            t1, t2 = original_times[idx - 1], original_times[idx]
            alpha = (t - t1) / (t2 - t1)
            p0, p1 = positions[idx - 1], positions[idx]
            q0, q1 = quaternions_wxyz[idx - 1], quaternions_wxyz[idx]
            p_new = (1.0 - alpha) * p0 + alpha * p1
            q_new = slerp(q0, q1, alpha)
        up_pos.append(p_new)
        up_quat.append(q_new)

    return np.asarray(up_pos, dtype=np.float32), np.asarray(up_quat, dtype=np.float32)


def process_and_interpolate(
    loaded_pose,
    reference_transform_inv: np.ndarray,
    offset_vector: np.ndarray,
    sync_with_real_vector: np.ndarray,
    n_samples: int = 1080,
):
    timestamps = []
    final_transform_list = []
    for ts, final_transform_tmp, _ in loaded_pose:
        timestamps.append(ts)
        final_transform_list.append(final_transform_tmp)
    final_transform = np.asarray(final_transform_list, dtype=np.float64)
    final_transform_rel = reference_transform_inv[None] @ final_transform

    # FoundationPose -> MuJoCo scene alignment, preserved from the original file.
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
    pos, quat_wxyz = transform_to_pos_quat_batch(flipped_transform)
    pos = pos - offset_vector[None, :] + sync_with_real_vector[None, :]
    up_pos, up_quat_wxyz = interpolate_transforms(pos, quat_wxyz, n_samples=n_samples)
    return timestamps, up_pos, up_quat_wxyz


def parse_iso_time(time_str: str) -> datetime:
    return datetime.fromisoformat(time_str)


def quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz, dtype=np.float32)
    return np.ascontiguousarray(q_wxyz[..., [1, 2, 3, 0]])


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _as_numpy_f32(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if shape is not None:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def _as_numpy_i32(x: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.int32)
    if shape is not None:
        arr = arr.reshape(shape)
    return np.ascontiguousarray(arr)


def normalize_quat_xyzw_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / max(np.linalg.norm(q), 1.0e-8)


def quat_mul_xyzw_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_rotate_xyzw_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = normalize_quat_xyzw_np(q)
    q_xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(q_xyz, v)
    return v + w * t + np.cross(q_xyz, t)


def transform_compose_np(parent_tf: np.ndarray, child_tf: np.ndarray) -> np.ndarray:
    p_t = parent_tf[:3]
    p_q = parent_tf[3:7]
    c_t = child_tf[:3]
    c_q = child_tf[3:7]
    world_t = p_t + quat_rotate_xyzw_np(p_q, c_t)
    world_q = normalize_quat_xyzw_np(quat_mul_xyzw_np(p_q, c_q))
    return np.concatenate([world_t, world_q]).astype(np.float32)


def find_named_index(labels: list[str], target: str) -> int:
    if target in labels:
        return labels.index(target)
    for i, label in enumerate(labels):
        if label.endswith(f"/{target}"):
            return i
    for i, label in enumerate(labels):
        if label.split("/")[-1] == target:
            return i
    for i, label in enumerate(labels):
        if target in label:
            return i
    raise ValueError(f"Could not find '{target}' in labels: {labels}")


# -----------------------------------------------------------------------------
# Warp kernels
# -----------------------------------------------------------------------------
@wp.kernel
def relu_masses_uniform(
    base_masses: wp.array(dtype=float),
    delta: wp.array(dtype=float),
    masses_out: wp.array(dtype=float),
):
    tid = wp.tid()
    masses_out[tid] = wp.max(base_masses[tid] + delta[0], MASS_EPS)


@wp.kernel
def relu_masses_nonuniform(
    base_masses: wp.array(dtype=float),
    delta: wp.array(dtype=float),
    masses_out: wp.array(dtype=float),
):
    tid = wp.tid()
    masses_out[tid] = wp.max(base_masses[tid] + delta[tid], MASS_EPS)


@wp.kernel
def reduce_point_mass_properties(
    vertices_mesh_local: wp.array(dtype=wp.vec3),
    vertex_masses: wp.array(dtype=float),
    vertex_count: int,
    total_mass_out: wp.array(dtype=float),
    com_out: wp.array(dtype=wp.vec3),
    inertia_out: wp.array(dtype=wp.mat33),
):
    if wp.tid() != 0:
        return

    total_mass = float(0.0)
    com = wp.vec3(0.0, 0.0, 0.0)
    for i in range(vertex_count):
        m = vertex_masses[i]
        total_mass = total_mass + m
        com = com + vertices_mesh_local[i] * m

    total_mass = wp.max(total_mass, MASS_EPS)
    com = com / total_mass

    I = wp.mat33(
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    )
    for i in range(vertex_count):
        m = vertex_masses[i]
        r = vertices_mesh_local[i] - com
        x = r[0]
        y = r[1]
        z = r[2]
        I = I + m * wp.mat33(
            y * y + z * z, -x * y, -x * z,
            -y * x, x * x + z * z, -y * z,
            -z * x, -z * y, x * x + y * y,
        )

    total_mass_out[0] = total_mass
    com_out[0] = com
    inertia_out[0] = I


@wp.kernel
def write_body_properties(
    body_mass: wp.array(dtype=float),
    body_inv_mass: wp.array(dtype=float),
    body_com: wp.array(dtype=wp.vec3),
    body_inertia: wp.array(dtype=wp.mat33),
    body_inv_inertia: wp.array(dtype=wp.mat33),
    body_id: int,
    total_mass: wp.array(dtype=float),
    com: wp.array(dtype=wp.vec3),
    inertia: wp.array(dtype=wp.mat33),
):
    if wp.tid() != 0:
        return

    m = wp.max(total_mass[0], MASS_EPS)
    c = com[0]
    I = inertia[0] + wp.mat33(
        INERTIA_REG_EPS, 0.0, 0.0,
        0.0, INERTIA_REG_EPS, 0.0,
        0.0, 0.0, INERTIA_REG_EPS,
    )

    body_mass[body_id] = m
    body_inv_mass[body_id] = 1.0 / m
    body_com[body_id] = c
    body_inertia[body_id] = I
    body_inv_inertia[body_id] = wp.inverse(I)


@wp.kernel
def set_control_targets_for_frame(
    control_signal: wp.array2d(dtype=float),
    control_indices: wp.array(dtype=wp.int32),
    frame_idx: int,
    joint_target_pos: wp.array(dtype=float),
):
    tid = wp.tid()
    joint_target_pos[control_indices[tid]] = control_signal[frame_idx, tid]


@wp.kernel
def accumulate_shape_pose_loss(
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    gt_positions: wp.array(dtype=wp.vec3),
    gt_orientations_xyzw: wp.array(dtype=wp.quat),
    body_id: int,
    shape_id: int,
    gt_frame_idx: int,
    loss: wp.array(dtype=float),
    pos_weight: float,
    quat_weight: float,
):
    if wp.tid() != 0:
        return

    body_tf = body_q[body_id]
    mesh_tf = wp.transform_multiply(body_tf, shape_transform[shape_id])

    pos = wp.transform_get_translation(mesh_tf)
    quat = wp.transform_get_rotation(mesh_tf)
    gt_pos = gt_positions[gt_frame_idx]
    gt_quat = gt_orientations_xyzw[gt_frame_idx]

    dp = pos - gt_pos
    pos_term = wp.dot(dp, dp)

    dqp0 = quat[0] - gt_quat[0]
    dqp1 = quat[1] - gt_quat[1]
    dqp2 = quat[2] - gt_quat[2]
    dqp3 = quat[3] - gt_quat[3]
    qp = dqp0 * dqp0 + dqp1 * dqp1 + dqp2 * dqp2 + dqp3 * dqp3

    dqm0 = quat[0] + gt_quat[0]
    dqm1 = quat[1] + gt_quat[1]
    dqm2 = quat[2] + gt_quat[2]
    dqm3 = quat[3] + gt_quat[3]
    qm = dqm0 * dqm0 + dqm1 * dqm1 + dqm2 * dqm2 + dqm3 * dqm3

    quat_term = wp.min(qp, qm)
    loss[0] = loss[0] + pos_weight * pos_term + quat_weight * quat_term


@wp.kernel
def scale_loss_in_place(loss: wp.array(dtype=float), scale: float):
    if wp.tid() != 0:
        return
    loss[0] = loss[0] * scale


# -----------------------------------------------------------------------------
# Config + estimator
# -----------------------------------------------------------------------------
@dataclass
class NewtonSceneEstimatorConfig:
    dt: float = DT_DEFAULT
    epochs: int = 130
    compare_every: int = 10
    uniform_density: bool = True
    init_total_mass: float = 0.20
    joint_ke: float = 650.0
    joint_kd: float = 80.0
    pos_weight: float = 1.0
    quat_weight: float = 1.0
    soft_contact_margin: float = 0.01
    enable_contact_normal_gradients: bool = True
    control_target_offset: int = 0
    control_q_offset: int = 0
    lr: float = 5.0e-3
    device: str | None = None
    parse_visuals_as_colliders: bool = False
    enable_self_collisions: bool = True


class NewtonFullSceneMassEstimator:
    def __init__(
        self,
        scene_path: str | Path,
        mesh_vertices: np.ndarray,
        mesh_faces: np.ndarray,
        control_signal: np.ndarray,
        gt_positions_world: np.ndarray,
        gt_quat_xyzw: np.ndarray,
        start_sync_frame: int,
        cookie_body_name: str = "cookie",
        cookie_shape_name: str | None = None,
        config: NewtonSceneEstimatorConfig | None = None,
    ) -> None:
        self.cfg = config or NewtonSceneEstimatorConfig()
        self.device = wp.get_device(self.cfg.device) if self.cfg.device else None

        self.scene_path = str(scene_path)
        self.cookie_body_name = cookie_body_name
        self.cookie_shape_name = cookie_shape_name

        self.vertices_np = _as_numpy_f32(mesh_vertices, (-1, 3))
        self.faces_np = _as_numpy_i32(mesh_faces, (-1, 3))
        self.control_signal_np = _as_numpy_f32(control_signal)
        self.gt_positions_np = _as_numpy_f32(gt_positions_world, (-1, 3))
        self.gt_quat_np = _as_numpy_f32(gt_quat_xyzw, (-1, 4))
        self.start_sync_frame = int(start_sync_frame)
        self.gt_frame_count = int(self.gt_positions_np.shape[0])
        self.rollout_steps = min(self.control_signal_np.shape[0], self.start_sync_frame + self.gt_frame_count)
        self.control_count = int(self.control_signal_np.shape[1])
        self.vertex_count = int(self.vertices_np.shape[0])
        self.base_vertex_masses_np = np.full(
            self.vertex_count,
            float(self.cfg.init_total_mass) / float(max(self.vertex_count, 1)),
            dtype=np.float32,
        )

        self.vertices_wp = wp.array(self.vertices_np, dtype=wp.vec3, device=self.device)
        self.gt_positions_wp = wp.array(self.gt_positions_np, dtype=wp.vec3, device=self.device)
        self.gt_quat_wp = wp.array(self.gt_quat_np, dtype=wp.quat, device=self.device)
        self.control_signal_wp = wp.array(self.control_signal_np[: self.rollout_steps], dtype=float, device=self.device)
        self.base_vertex_masses_wp = wp.array(self.base_vertex_masses_np, dtype=float, device=self.device)

        self._build_newton_scene()
        self._build_optimization_state()

        self.loss_history: list[float] = []
        self.mass_history: list[float] = []

    def _import_scene(self, builder: newton.ModelBuilder) -> None:
        suffix = Path(self.scene_path).suffix.lower()
        if suffix == ".urdf":
            builder.add_urdf(
                self.scene_path,
                floating=False,
                parse_visuals_as_colliders=self.cfg.parse_visuals_as_colliders,
                enable_self_collisions=self.cfg.enable_self_collisions,
            )
        else:
            builder.add_mjcf(
                self.scene_path,
                enable_self_collisions=self.cfg.enable_self_collisions,
            )

    def _select_cookie_shape_id(self, builder: newton.ModelBuilder, cookie_body_id: int) -> int:
        cookie_shapes = [i for i, b in enumerate(builder.shape_body) if int(b) == cookie_body_id]
        if not cookie_shapes:
            raise RuntimeError(f"No shapes found on body '{self.cookie_body_name}'.")

        if self.cookie_shape_name and hasattr(builder, "shape_label"):
            try:
                return find_named_index(list(builder.shape_label), self.cookie_shape_name)
            except Exception:
                pass

        if hasattr(builder, "shape_label"):
            shape_labels = list(builder.shape_label)
            for sid in cookie_shapes:
                label = shape_labels[sid]
                if self.cookie_body_name in label:
                    return sid

        return cookie_shapes[0]

    def _build_newton_scene(self) -> None:
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        self._import_scene(builder)

        body_labels = list(builder.body_label)
        self.cookie_body_id = find_named_index(body_labels, self.cookie_body_name)
        self.cookie_shape_id = self._select_cookie_shape_id(builder, self.cookie_body_id)

        # Zero all target gains by default, then enable only the replayed hand/robot DOFs.
        if hasattr(builder, "joint_target_ke"):
            builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
        if hasattr(builder, "joint_target_kd"):
            builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)

        control_target_end = self.cfg.control_target_offset + self.control_count
        control_q_end = self.cfg.control_q_offset + self.control_count
        if control_target_end > len(builder.joint_target_pos):
            raise ValueError(
                f"Control target range [{self.cfg.control_target_offset}, {control_target_end}) exceeds "
                f"builder.joint_target_pos length {len(builder.joint_target_pos)}."
            )
        if control_q_end > len(builder.joint_q):
            raise ValueError(
                f"Control q range [{self.cfg.control_q_offset}, {control_q_end}) exceeds "
                f"builder.joint_q length {len(builder.joint_q)}."
            )

        initial_control = self.control_signal_np[0]
        for i in range(self.control_count):
            builder.joint_target_pos[self.cfg.control_target_offset + i] = float(initial_control[i])
            builder.joint_q[self.cfg.control_q_offset + i] = float(initial_control[i])
            if hasattr(builder, "joint_target_ke"):
                builder.joint_target_ke[self.cfg.control_target_offset + i] = float(self.cfg.joint_ke)
            if hasattr(builder, "joint_target_kd"):
                builder.joint_target_kd[self.cfg.control_target_offset + i] = float(self.cfg.joint_kd)

        self.model = builder.finalize(requires_grad=True, device=self.device)
        self.control = self.model.control()
        self.states = [self.model.state(requires_grad=True) for _ in range(self.rollout_steps + 1)]
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        try:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                requires_grad=True,
                soft_contact_margin=self.cfg.soft_contact_margin,
                enable_contact_normal_gradients=self.cfg.enable_contact_normal_gradients,
            )
        except TypeError:
            self.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="explicit",
                requires_grad=True,
                soft_contact_margin=self.cfg.soft_contact_margin,
            )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverSemiImplicit(self.model, enable_tri_contact=False)

        self.control_indices_np = np.arange(
            self.cfg.control_target_offset,
            self.cfg.control_target_offset + self.control_count,
            dtype=np.int32,
        )
        self.control_indices_wp = wp.array(self.control_indices_np, dtype=wp.int32, device=self.device)

        self.shape_local_tf_np = np.asarray(self.model.shape_transform.numpy()[self.cookie_shape_id], dtype=np.float32).copy()

    def _build_optimization_state(self) -> None:
        if self.cfg.uniform_density:
            self.mass_update = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)
        else:
            self.mass_update = wp.zeros(self.vertex_count, dtype=float, requires_grad=True, device=self.device)

        self.vertex_masses = wp.zeros(self.vertex_count, dtype=float, requires_grad=True, device=self.device)
        self.total_mass = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)
        self.com = wp.zeros(1, dtype=wp.vec3, requires_grad=True, device=self.device)
        self.inertia = wp.zeros(1, dtype=wp.mat33, requires_grad=True, device=self.device)
        self.loss = wp.zeros(1, dtype=float, requires_grad=True, device=self.device)
        self.optimizer = warp.optim.Adam([self.mass_update.flatten()], lr=self.cfg.lr)

    def set_learning_rate(self, lr: float) -> None:
        self.optimizer.lr = float(lr)

    def _materialize_vertex_masses(self) -> None:
        if self.cfg.uniform_density:
            wp.launch(
                relu_masses_uniform,
                dim=self.vertex_count,
                inputs=[self.base_vertex_masses_wp, self.mass_update],
                outputs=[self.vertex_masses],
                device=self.device,
            )
        else:
            wp.launch(
                relu_masses_nonuniform,
                dim=self.vertex_count,
                inputs=[self.base_vertex_masses_wp, self.mass_update],
                outputs=[self.vertex_masses],
                device=self.device,
            )

    def _update_cookie_body_properties(self) -> None:
        self._materialize_vertex_masses()
        wp.launch(
            reduce_point_mass_properties,
            dim=1,
            inputs=[self.vertices_wp, self.vertex_masses, self.vertex_count],
            outputs=[self.total_mass, self.com, self.inertia],
            device=self.device,
        )
        wp.launch(
            write_body_properties,
            dim=1,
            inputs=[
                self.model.body_mass,
                self.model.body_inv_mass,
                self.model.body_com,
                self.model.body_inertia,
                self.model.body_inv_inertia,
                self.cookie_body_id,
                self.total_mass,
                self.com,
                self.inertia,
            ],
            outputs=[],
            device=self.device,
        )
        if hasattr(self.solver, "notify_model_changed"):
            self.solver.notify_model_changed()

    def _compare_frame_indices(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        end_frame = min(self.rollout_steps, self.start_sync_frame + self.gt_frame_count)
        for sim_frame in range(self.start_sync_frame, end_frame):
            gt_idx = sim_frame - self.start_sync_frame
            if (gt_idx % max(self.cfg.compare_every, 1)) == 0:
                pairs.append((sim_frame, gt_idx))
        if not pairs and end_frame > self.start_sync_frame:
            pairs.append((end_frame - 1, end_frame - 1 - self.start_sync_frame))
        return pairs

    def forward(self) -> wp.array:
        self.loss.zero_()
        self._update_cookie_body_properties()
        self.states[0] = self.model.state(requires_grad=True)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        compare_pairs = self._compare_frame_indices()
        compare_lookup = {sim_idx: gt_idx for sim_idx, gt_idx in compare_pairs}

        for frame in range(self.rollout_steps):
            state_in = self.states[frame]
            state_out = self.states[frame + 1]
            state_in.clear_forces()

            wp.launch(
                set_control_targets_for_frame,
                dim=self.control_count,
                inputs=[self.control_signal_wp, self.control_indices_wp, frame, self.control.joint_target_pos],
                outputs=[],
                device=self.device,
            )

            self.collision_pipeline.collide(state_in, self.contacts)
            self.solver.step(state_in, state_out, self.control, self.contacts, self.cfg.dt)

            if frame in compare_lookup:
                wp.launch(
                    accumulate_shape_pose_loss,
                    dim=1,
                    inputs=[
                        state_out.body_q,
                        self.model.shape_transform,
                        self.gt_positions_wp,
                        self.gt_quat_wp,
                        self.cookie_body_id,
                        self.cookie_shape_id,
                        compare_lookup[frame],
                        self.loss,
                        self.cfg.pos_weight,
                        self.cfg.quat_weight,
                    ],
                    outputs=[],
                    device=self.device,
                )

        if compare_pairs:
            wp.launch(
                scale_loss_in_place,
                dim=1,
                inputs=[self.loss, 1.0 / float(len(compare_pairs))],
                outputs=[],
                device=self.device,
            )
        return self.loss

    def step(self) -> float:
        tape = wp.Tape()
        with tape:
            loss = self.forward()
        tape.backward(loss)

        if self.mass_update.grad is None:
            raise RuntimeError("mass_update.grad is None after backward().")

        self.optimizer.step([self.mass_update.grad.flatten()])

        loss_value = float(self.loss.numpy()[0])
        total_mass_value = float(self.total_mass.numpy()[0])
        self.loss_history.append(loss_value)
        self.mass_history.append(total_mass_value)

        tape.zero()
        self.loss.zero_()
        return loss_value

    def train(self, epochs: int) -> dict[str, np.ndarray | float]:
        for epoch in range(int(epochs)):
            loss_value = self.step()
            if epoch in [40, 80]:
                self.optimizer.lr *= 0.5
            print(f"[Newton][epoch={epoch:04d}] loss={loss_value:.6f} total_mass={self.mass_history[-1]:.6f}")

        # Forward once more with the final updated mass parameter for output trajectories.
        self.forward()
        traj_pos, traj_quat = self.extract_cookie_shape_trajectory()
        return {
            "loss_history": np.asarray(self.loss_history, dtype=np.float32),
            "mass_history": np.asarray(self.mass_history, dtype=np.float32),
            "final_total_mass": float(self.total_mass.numpy()[0]),
            "final_vertex_masses": self.current_vertex_masses(),
            "trajectory_positions": traj_pos,
            "trajectory_orientations_xyzw": traj_quat,
        }

    def current_vertex_masses(self) -> np.ndarray:
        self._materialize_vertex_masses()
        return np.asarray(self.vertex_masses.numpy(), dtype=np.float32).copy()

    def extract_cookie_shape_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        traj = []
        for state in self.states[1:]:
            body_tf = np.asarray(state.body_q.numpy()[self.cookie_body_id], dtype=np.float32)
            shape_tf = transform_compose_np(body_tf, self.shape_local_tf_np)
            traj.append(shape_tf)
        traj = np.asarray(traj, dtype=np.float32)
        return traj[:, :3].copy(), traj[:, 3:7].copy()


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expid", type=str, default="default")
    parser.add_argument(
        "--logdir",
        type=str,
        default="./cache/mass_known_shape_cookie_newton_scene",
        help="Directory to store logs in.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=130)
    parser.add_argument("--compare-every", type=int, default=10)
    parser.add_argument("--scene-path", type=str, default="./wonik_allegro/scene_cookie.xml")
    parser.add_argument("--mesh-path", type=str, default="./wonik_allegro/assets/cookie.stl")
    parser.add_argument(
        "--control-signal-path",
        type=str,
        default="./realworldobject_sequence/cookie/1/joint_states_log.csv",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default="./realworldobject_sequence/cookie/path_full_up_1/target_cookie_20250428_1714.txt",
    )
    parser.add_argument("--cookie-body-name", type=str, default="cookie")
    parser.add_argument("--cookie-shape-name", type=str, default=None)
    parser.add_argument("--sync-str", type=str, default="2025-04-28T17:14:19.920929")
    parser.add_argument("--start-sync-frame", type=int, default=236)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--dt", type=float, default=DT_DEFAULT)
    parser.add_argument("--init-total-mass", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.set_defaults(uniform_density=True)
    parser.add_argument("--uniform-density", dest="uniform_density", action="store_true")
    parser.add_argument("--nonuniform-density", dest="uniform_density", action="store_false")
    parser.add_argument("--joint-ke", type=float, default=650.0)
    parser.add_argument("--joint-kd", type=float, default=80.0)
    parser.add_argument("--soft-contact-margin", type=float, default=0.01)
    parser.add_argument("--control-target-offset", type=int, default=0)
    parser.add_argument("--control-q-offset", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    wp.init()

    mesh = o3d.io.read_triangle_mesh(args.mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)

    control_signal = load_position_data(args.control_signal_path)
    loaded_pose = load_observations(args.gt_path)
    sync_dt = parse_iso_time(args.sync_str)
    filtered_pose = [
        (ts_str, final_transform, pose)
        for (ts_str, final_transform, pose) in loaded_pose
        if parse_iso_time(ts_str) >= sync_dt
    ]

    loaded_pose = filtered_pose[:13]
    reference_transform = loaded_pose[0][1]
    reference_transform_inv = np.linalg.inv(reference_transform)
    reference_transform_inv[2, 3] *= -1.0

    offset_vector = np.array([0.0, 0.0, 1.6], dtype=np.float32)
    sync_with_real_vector = np.array([0.095, -0.08, 0.02], dtype=np.float32)

    _timestamps, upsampled_pos, upsampled_quat_wxyz = process_and_interpolate(
        loaded_pose,
        reference_transform_inv,
        offset_vector,
        sync_with_real_vector,
        n_samples=args.n_samples,
    )
    upsampled_quat_xyzw = quat_wxyz_to_xyzw(upsampled_quat_wxyz)

    cfg = NewtonSceneEstimatorConfig(
        dt=args.dt,
        epochs=args.epochs,
        compare_every=args.compare_every,
        uniform_density=args.uniform_density,
        init_total_mass=args.init_total_mass,
        joint_ke=args.joint_ke,
        joint_kd=args.joint_kd,
        soft_contact_margin=args.soft_contact_margin,
        control_target_offset=args.control_target_offset,
        control_q_offset=args.control_q_offset,
        lr=args.lr,
        device=args.device,
    )

    estimator = NewtonFullSceneMassEstimator(
        scene_path=args.scene_path,
        mesh_vertices=vertices,
        mesh_faces=faces,
        control_signal=control_signal,
        gt_positions_world=upsampled_pos,
        gt_quat_xyzw=upsampled_quat_xyzw,
        start_sync_frame=args.start_sync_frame,
        cookie_body_name=args.cookie_body_name,
        cookie_shape_name=args.cookie_shape_name,
        config=cfg,
    )
    estimator.set_learning_rate(args.lr)

    results = estimator.train(args.epochs)

    final_total_mass = float(results["final_total_mass"])
    print(f"\nFinal Newton-estimated total mass for '{args.cookie_body_name}': {final_total_mass:.6f}")

    if args.log:
        logdir = Path(args.logdir) / args.expid
        logdir.mkdir(parents=True, exist_ok=True)
        np.savetxt(logdir / "losses.txt", results["loss_history"])
        np.savetxt(logdir / "mass_history.txt", results["mass_history"])
        np.savetxt(logdir / "final_vertex_masses.txt", results["final_vertex_masses"])
        np.savetxt(logdir / "trajectory_positions.txt", results["trajectory_positions"])
        np.savetxt(logdir / "trajectory_orientations_xyzw.txt", results["trajectory_orientations_xyzw"])
        with open(logdir / "final_total_mass.txt", "w") as f:
            f.write(f"{final_total_mass:.8f}\n")
        print(f"Saved logs to {logdir}")


if __name__ == "__main__":
    main()
