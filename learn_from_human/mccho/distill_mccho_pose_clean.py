from __future__ import annotations

"""Distill HaMeR/MANO wrist pose into the object-centered frame.

Expected per-sequence layout:

    seq_dir/
        output_0.1_hand.ply      # full reconstruction: hand + object
        output_0.1.ply           # hand-only reconstruction
        pose/
            000123__1mano_output.yaml

This refactor fixes and clarifies the original workflow:
- hand points are removed from the full reconstruction using nearest-neighbor
  distance instead of selecting indices from the wrong point cloud;
- output is written to `seq_dir/pose_distilled/` by default;
- CLI arguments make the expected filenames explicit.
"""

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial.transform import Rotation as R


class ScriptError(RuntimeError):
    """Raised for user-facing failures in the pipeline."""


def load_point_cloud(path: Path, description: str) -> o3d.geometry.PointCloud:
    if not path.is_file():
        raise ScriptError(f"{description} not found: {path}")

    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ScriptError(f"{description} is empty: {path}")
    return cloud


def estimate_subtraction_threshold(hand_cloud: o3d.geometry.PointCloud, scale: float, minimum: float) -> float:
    nn_distances = np.asarray(hand_cloud.compute_nearest_neighbor_distance())
    nn_distances = nn_distances[np.isfinite(nn_distances) & (nn_distances > 0)]
    if nn_distances.size == 0:
        return minimum
    return max(float(np.median(nn_distances) * scale), minimum)


def subtract_hand_from_full_cloud(
    full_cloud: o3d.geometry.PointCloud,
    hand_cloud: o3d.geometry.PointCloud,
    threshold: float,
) -> o3d.geometry.PointCloud:
    distances = np.asarray(full_cloud.compute_point_cloud_distance(hand_cloud))
    keep_indices = np.where(distances > threshold)[0]
    object_cloud = full_cloud.select_by_index(keep_indices)
    if object_cloud.is_empty():
        raise ScriptError(
            "Object cloud became empty after subtracting hand points. "
            "Try a smaller --subtract-threshold or inspect the input point clouds."
        )
    return object_cloud


def estimate_object_pose(object_cloud: o3d.geometry.PointCloud) -> tuple[np.ndarray, np.ndarray]:
    obb = object_cloud.get_oriented_bounding_box()
    rotation = np.asarray(obb.R)
    translation = np.asarray(obb.center)
    return rotation, translation


def load_joint_rotations(mano_yaml_path: Path) -> tuple[dict, np.ndarray]:
    with mano_yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if "hand_pose" not in data:
        raise ScriptError(f"'hand_pose' key not found in YAML: {mano_yaml_path}")

    joints = np.asarray(data["hand_pose"], dtype=np.float32)
    if joints.size % 9 != 0:
        raise ScriptError(f"'hand_pose' size is not divisible by 9 in: {mano_yaml_path}")

    joints = joints.reshape(-1, 3, 3)
    if joints.shape[0] == 0:
        raise ScriptError(f"No joint rotations found in YAML: {mano_yaml_path}")

    return data, joints


def align_mano_to_object_frame(joints: np.ndarray, object_rotation: R) -> tuple[np.ndarray, np.ndarray]:
    relative_rotation = object_rotation.inv()
    aligned_joints = relative_rotation.as_matrix() @ joints
    wrist_quaternion_xyzw = R.from_matrix(aligned_joints[0]).as_quat()
    return aligned_joints, wrist_quaternion_xyzw


def write_distilled_yaml(
    output_path: Path,
    aligned_joints: np.ndarray,
    wrist_quaternion_xyzw: np.ndarray,
    object_center_xyz: np.ndarray,
    object_quaternion_xyzw: np.ndarray,
) -> None:
    payload = {
        "wrist_quaternion_xyzw": wrist_quaternion_xyzw.tolist(),
        "hand_pose_15x3x3": aligned_joints.reshape(-1, 9).tolist(),
        "object_center_xyz": object_center_xyz.tolist(),
        "object_quaternion_xyzw": object_quaternion_xyzw.tolist(),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def process_sequence(
    seq_dir: Path,
    pose_dir_name: str,
    output_pose_dir_name: str,
    full_cloud_name: str,
    hand_cloud_name: str,
    subtract_threshold: float | None,
    threshold_scale: float,
    min_threshold: float,
    save_object_cloud: bool,
) -> None:
    pose_dir = seq_dir / pose_dir_name
    if not pose_dir.is_dir():
        raise ScriptError(f"Pose directory not found: {pose_dir}")

    yaml_files = sorted(pose_dir.glob("*__1mano_output.yaml"))
    if not yaml_files:
        raise ScriptError(f"No MANO YAML files found under: {pose_dir}")

    full_cloud = load_point_cloud(seq_dir / full_cloud_name, "Full cloud")
    hand_cloud = load_point_cloud(seq_dir / hand_cloud_name, "Hand-only cloud")

    used_threshold = subtract_threshold
    if used_threshold is None:
        used_threshold = estimate_subtraction_threshold(hand_cloud, scale=threshold_scale, minimum=min_threshold)

    object_cloud = subtract_hand_from_full_cloud(full_cloud, hand_cloud, threshold=used_threshold)
    object_rotation_mat, object_center_xyz = estimate_object_pose(object_cloud)
    object_rotation = R.from_matrix(object_rotation_mat)

    output_pose_dir = seq_dir / output_pose_dir_name
    output_pose_dir.mkdir(parents=True, exist_ok=True)

    if save_object_cloud:
        debug_dir = seq_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(debug_dir / "object_only.ply"), object_cloud)

    print(f"[INFO] {seq_dir.name}: subtract-threshold = {used_threshold:.6g}")

    for mano_yaml_path in yaml_files:
        _, joints = load_joint_rotations(mano_yaml_path)
        aligned_joints, wrist_quaternion_xyzw = align_mano_to_object_frame(joints, object_rotation)

        output_path = output_pose_dir / f"{mano_yaml_path.stem}_distilled.yaml"
        write_distilled_yaml(
            output_path=output_path,
            aligned_joints=aligned_joints,
            wrist_quaternion_xyzw=wrist_quaternion_xyzw,
            object_center_xyz=object_center_xyz,
            object_quaternion_xyzw=object_rotation.as_quat(),
        )
        print(f"  ✓ {output_path.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Align HaMeR/MANO poses to the object frame.")
    parser.add_argument(
        "seq_folders",
        nargs="+",
        help="One or more sequence folders produced by MCC-HO.",
    )
    parser.add_argument(
        "--pose-dir",
        dest="pose_dir_name",
        default="pose",
        help="Name of the folder containing HaMeR YAML files inside each sequence.",
    )
    parser.add_argument(
        "--output-pose-dir",
        dest="output_pose_dir_name",
        default="pose_distilled",
        help="Name of the output folder created inside each sequence.",
    )
    parser.add_argument(
        "--full-cloud-name",
        default="output_0.1_hand.ply",
        help="Filename of the full hand+object point cloud.",
    )
    parser.add_argument(
        "--hand-cloud-name",
        default="output_0.1.ply",
        help="Filename of the hand-only point cloud.",
    )
    parser.add_argument(
        "--subtract-threshold",
        type=float,
        default=None,
        help="Distance threshold for removing hand points from the full cloud. If omitted, a threshold is estimated from hand-cloud spacing.",
    )
    parser.add_argument(
        "--threshold-scale",
        type=float,
        default=2.0,
        help="Multiplier applied to the median hand-cloud nearest-neighbor distance when auto-estimating the subtraction threshold.",
    )
    parser.add_argument(
        "--min-threshold",
        type=float,
        default=1e-4,
        help="Lower bound for the auto-estimated subtraction threshold.",
    )
    parser.add_argument(
        "--save-object-cloud",
        action="store_true",
        help="Also save the object-only point cloud to seq_dir/debug/object_only.ply.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    for seq_folder in args.seq_folders:
        seq_dir = Path(seq_folder)
        if not seq_dir.is_dir():
            raise SystemExit(f"[ERR] Path is not a directory: {seq_dir}")

        print(f"[INFO] Processing: {seq_dir}")
        process_sequence(
            seq_dir=seq_dir,
            pose_dir_name=args.pose_dir_name,
            output_pose_dir_name=args.output_pose_dir_name,
            full_cloud_name=args.full_cloud_name,
            hand_cloud_name=args.hand_cloud_name,
            subtract_threshold=args.subtract_threshold,
            threshold_scale=args.threshold_scale,
            min_threshold=args.min_threshold,
            save_object_cloud=args.save_object_cloud,
        )


if __name__ == "__main__":
    main()
