#!/usr/bin/env python3
"""Retarget MANO hand poses stored in YAML files into Leap hand joint values.

Each input YAML file is expected to contain:
    - betas: shape (10,) or (1, 10)
    - global_orient: shape (3, 3) or (1, 3, 3)
    - hand_pose: shape (15, 3, 3), (16, 3, 3), or with a leading singleton
      batch dimension such as (1, 15, 3, 3) / (1, 16, 3, 3)

The script:
    1. loads YAML files from a directory,
    2. converts rotation matrices to MANO axis-angle pose vectors,
    3. runs MANO forward kinematics through manotorch,
    4. extracts Leap-style joint values,
    5. saves the result to a .npy file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as SciPyRotation
from torch.utils.data import DataLoader, Dataset

from manotorch.axislayer import AxisLayerFK
from manotorch.manolayer import MANOOutput, ManoLayer

TensorLike = Union[np.ndarray, torch.Tensor]


def _as_numpy(array: TensorLike) -> np.ndarray:
    """Convert a tensor/array-like object to a detached NumPy array on CPU."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _squeeze_leading_singleton(array: np.ndarray) -> np.ndarray:
    """Remove a leading singleton dimension when present."""
    if array.ndim > 0 and array.shape[0] == 1:
        return array[0]
    return array


def _require_shape(name: str, array: np.ndarray, valid_shapes: Sequence[tuple[int, ...]]) -> None:
    """Raise a readable error when an array shape is not one of the expected shapes."""
    if tuple(array.shape) not in valid_shapes:
        valid = ", ".join(str(shape) for shape in valid_shapes)
        raise ValueError(f"{name} has shape {tuple(array.shape)}, expected one of: {valid}")


class ManoRetargeter:
    """Wrapper around manotorch that converts MANO pose vectors into Leap joint values."""

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        mano_assets_root: Union[str, Path] = "./manotorch/assets/mano",
        side: str = "left",
    ) -> None:
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = resolved_device
        assets_root = str(mano_assets_root)

        self.mano_layer = ManoLayer(
            rot_mode="axisang",
            use_pca=False,
            side=side,
            center_idx=None,
            mano_assets_root=assets_root,
            flat_hand_mean=False,
        )
        self.axis_layer = AxisLayerFK(mano_assets_root=assets_root)

        if hasattr(self.mano_layer, "to"):
            self.mano_layer = self.mano_layer.to(self.device)
        if hasattr(self.axis_layer, "to"):
            self.axis_layer = self.axis_layer.to(self.device)
        if hasattr(self.mano_layer, "eval"):
            self.mano_layer.eval()
        if hasattr(self.axis_layer, "eval"):
            self.axis_layer.eval()

    def mano_retarget(self, joint_pose: TensorLike, shape_params: TensorLike) -> list[float]:
        """Retarget a single MANO pose vector and shape vector."""
        joint_pose_tensor = torch.as_tensor(joint_pose, dtype=torch.float32, device=self.device).reshape(1, -1)
        shape_params_tensor = torch.as_tensor(shape_params, dtype=torch.float32, device=self.device).reshape(1, -1)

        if joint_pose_tensor.shape[1] != 48:
            raise ValueError(f"joint_pose must contain 48 values, got {joint_pose_tensor.shape[1]}")
        if shape_params_tensor.shape[1] != 10:
            raise ValueError(f"shape_params must contain 10 values, got {shape_params_tensor.shape[1]}")

        with torch.no_grad():
            mano_results: MANOOutput = self.mano_layer(joint_pose_tensor, shape_params_tensor)
            transforms_abs = mano_results.transforms_abs
            _global_axis_transforms, _axis_rotations, end_effectors = self.axis_layer(transforms_abs)

        return self._extract_leap_joint_values(end_effectors.squeeze(0).detach().cpu().numpy())

    @staticmethod
    def _extract_leap_joint_values(finger_joints: np.ndarray) -> list[float]:
        """Extract Leap-style joint values from the 16x3 axis output."""
        finger_joints = np.asarray(finger_joints, dtype=np.float32).reshape(16, 3)

        finger_mcp_id = [1, 4, 10]
        finger_pip_id = [2, 5, 11]
        finger_dip_id = [3, 6, 12]

        ee_mcps = finger_joints[finger_mcp_id]
        ee_pips = finger_joints[finger_pip_id]
        ee_dips = finger_joints[finger_dip_id]

        joint_mcp_side = -ee_mcps[:, 1]
        joint_mcp_forward = ee_mcps[:, 2]
        joint_pip = ee_pips[:, 2]
        joint_dip = ee_dips[:, 2]

        thumb_cmc_side = finger_joints[13, 1]
        thumb_cmc_forward = finger_joints[13, 2]
        thumb_mcp = finger_joints[14, 2]
        thumb_ip = finger_joints[15, 2]

        output: list[float] = []
        for i in range(3):
            output.extend(
                [
                    float(joint_mcp_side[i]),
                    float(joint_mcp_forward[i]),
                    float(joint_pip[i]),
                    float(joint_dip[i]),
                ]
            )

        output.extend(
            [
                float(thumb_cmc_side),
                float(thumb_cmc_forward),
                float(thumb_mcp),
                float(thumb_ip),
            ]
        )
        return output


class MultiFileManoDataset(Dataset):
    """Treat each YAML file as one sample."""

    REQUIRED_KEYS = ("betas", "global_orient", "hand_pose")

    def __init__(self, yaml_files: Sequence[Union[str, Path]]) -> None:
        super().__init__()
        self.yaml_files = [Path(path) for path in yaml_files]
        if not self.yaml_files:
            raise ValueError("No YAML files were provided.")

    def __len__(self) -> int:
        return len(self.yaml_files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        yaml_path = self.yaml_files[index]
        with yaml_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

        if not isinstance(data, Mapping):
            raise ValueError(f"{yaml_path} did not contain a YAML mapping/object.")

        missing_keys = [key for key in self.REQUIRED_KEYS if key not in data]
        if missing_keys:
            raise KeyError(f"{yaml_path} is missing required keys: {missing_keys}")

        betas = _squeeze_leading_singleton(np.asarray(data["betas"], dtype=np.float32))
        global_orient = _squeeze_leading_singleton(np.asarray(data["global_orient"], dtype=np.float32))
        hand_pose = _squeeze_leading_singleton(np.asarray(data["hand_pose"], dtype=np.float32))

        _require_shape("betas", betas, [(10,)])
        _require_shape("global_orient", global_orient, [(3, 3)])
        _require_shape("hand_pose", hand_pose, [(15, 3, 3), (16, 3, 3)])

        # Normalize to 15 hand joints so mixed datasets can still be batched.
        # This preserves the original script's behavior, which only used the first 15.
        hand_pose = hand_pose[:15]

        return {
            "betas": torch.from_numpy(betas),
            "global_orient": torch.from_numpy(global_orient),
            "hand_pose": torch.from_numpy(hand_pose),
            "source_path": str(yaml_path),
        }


def collate_samples(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields and keep source paths as a list."""
    return {
        "betas": torch.stack([item["betas"] for item in batch], dim=0),
        "global_orient": torch.stack([item["global_orient"] for item in batch], dim=0),
        "hand_pose": torch.stack([item["hand_pose"] for item in batch], dim=0),
        "source_paths": [item["source_path"] for item in batch],
    }


def rotation_matrices_to_axis_angles(
    global_orient_mat: TensorLike,
    hand_pose_mat: TensorLike,
) -> torch.Tensor:
    """Convert rotation matrices into MANO axis-angle pose vectors.

    Supported input shapes:
        - single sample:
            global_orient_mat -> (3, 3)
            hand_pose_mat     -> (15, 3, 3) or (16, 3, 3)
        - batched:
            global_orient_mat -> (B, 3, 3)
            hand_pose_mat     -> (B, 15, 3, 3) or (B, 16, 3, 3)

    Returns:
        torch.Tensor of shape (48,) for a single sample or (B, 48) for batched input.

    Notes:
        - MANO expects 48 values: 3 global orientation values + 15 joints * 3 values.
        - If 16 hand joints are provided, only the first 15 are used to preserve the
          behavior of the original script.
    """
    global_np = _as_numpy(global_orient_mat).astype(np.float32, copy=False)
    hand_np = _as_numpy(hand_pose_mat).astype(np.float32, copy=False)

    if global_np.ndim == 2 and hand_np.ndim == 3:
        if global_np.shape != (3, 3):
            raise ValueError(f"global_orient_mat must have shape (3, 3), got {global_np.shape}")
        if hand_np.shape not in ((15, 3, 3), (16, 3, 3)):
            raise ValueError(f"hand_pose_mat must have shape (15, 3, 3) or (16, 3, 3), got {hand_np.shape}")

        global_axis_angle = SciPyRotation.from_matrix(global_np).as_rotvec().astype(np.float32)
        hand_axis_angles = SciPyRotation.from_matrix(hand_np[:15]).as_rotvec().astype(np.float32).reshape(-1)
        pose = np.concatenate([global_axis_angle, hand_axis_angles], axis=0)
        return torch.from_numpy(pose)

    if global_np.ndim == 3 and hand_np.ndim == 4:
        if global_np.shape[1:] != (3, 3):
            raise ValueError(f"global_orient_mat must have shape (B, 3, 3), got {global_np.shape}")
        if hand_np.shape[1:] not in ((15, 3, 3), (16, 3, 3)):
            raise ValueError(
                "hand_pose_mat must have shape (B, 15, 3, 3) or (B, 16, 3, 3), "
                f"got {hand_np.shape}"
            )
        if global_np.shape[0] != hand_np.shape[0]:
            raise ValueError(
                "global_orient_mat and hand_pose_mat must have the same batch size, "
                f"got {global_np.shape[0]} and {hand_np.shape[0]}"
            )

        batch_size = global_np.shape[0]
        global_axis_angle = SciPyRotation.from_matrix(global_np).as_rotvec().astype(np.float32)
        hand_axis_angles = (
            SciPyRotation.from_matrix(hand_np[:, :15]).as_rotvec().astype(np.float32).reshape(batch_size, -1)
        )
        pose = np.concatenate([global_axis_angle, hand_axis_angles], axis=1)
        return torch.from_numpy(pose)

    raise ValueError(
        "Unsupported shapes for rotation_matrices_to_axis_angles: "
        f"global_orient_mat={global_np.shape}, hand_pose_mat={hand_np.shape}"
    )


def find_yaml_files(
    input_dir: Union[str, Path],
    glob_pattern: str = "*.yaml",
    filename_contains: Optional[str] = None,
) -> list[Path]:
    """Find YAML files in a directory and optionally filter by substring."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_path}")

    yaml_files = sorted(input_path.glob(glob_pattern))
    if filename_contains:
        yaml_files = [path for path in yaml_files if filename_contains in path.name]
    return yaml_files


def process_files(
    yaml_files: Sequence[Union[str, Path]],
    retargeter: ManoRetargeter,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Process YAML files and return MANO vectors and Leap outputs."""
    dataset = MultiFileManoDataset(yaml_files)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_samples,
    )

    mano_vectors: list[np.ndarray] = []
    leap_outputs: list[list[float]] = []

    for batch_index, batch in enumerate(loader):
        pose_axis_angles = rotation_matrices_to_axis_angles(batch["global_orient"], batch["hand_pose"])
        betas = batch["betas"].float()
        mano_batch = torch.cat([pose_axis_angles, betas], dim=1)

        for source_path, mano_grasp in zip(batch["source_paths"], mano_batch):
            joint_pose = mano_grasp[:48]
            shape_params = mano_grasp[48:]
            leap_hand_output = retargeter.mano_retarget(joint_pose, shape_params)

            mano_vectors.append(mano_grasp.detach().cpu().numpy())
            leap_outputs.append(leap_hand_output)
            logging.debug("Processed %s", source_path)

        logging.info("Processed batch %d containing %d sample(s).", batch_index, len(batch["source_paths"]))

    return np.asarray(mano_vectors, dtype=np.float32), np.asarray(leap_outputs, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Retarget MANO pose YAML files into Leap hand joint values."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input YAML files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the output .npy file containing Leap hand joint values.",
    )
    parser.add_argument(
        "--glob-pattern",
        default="*.yaml",
        help='Glob pattern used to find input files. Default: "*.yaml".',
    )
    parser.add_argument(
        "--filename-contains",
        default="_0mano_output.yaml",
        help='Only process files whose names contain this substring. Default: "_0mano_output.yaml".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of YAML files to load at once. Default: 16.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. Default: 0.",
    )
    parser.add_argument(
        "--mano-assets-root",
        type=Path,
        default=Path("./manotorch/assets/mano"),
        help="Path to the MANO assets directory used by manotorch.",
    )
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default="left",
        help='Hand side to use for MANO. Default: "left".',
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device to use, for example "cpu" or "cuda". Default: auto-detect.',
    )
    parser.add_argument(
        "--save-mano-vectors",
        type=Path,
        default=None,
        help="Optional path to save the intermediate 58D MANO vectors as a .npy file.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help='Logging level. Default: "INFO".',
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    yaml_files = find_yaml_files(
        input_dir=args.input_dir,
        glob_pattern=args.glob_pattern,
        filename_contains=args.filename_contains,
    )

    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML files matched pattern={args.glob_pattern!r} "
            f"and filename_contains={args.filename_contains!r} in {args.input_dir}"
        )

    logging.info("Found %d YAML file(s) to process.", len(yaml_files))

    retargeter = ManoRetargeter(
        device=args.device,
        mano_assets_root=args.mano_assets_root,
        side=args.side,
    )

    mano_vectors, leap_outputs = process_files(
        yaml_files=yaml_files,
        retargeter=retargeter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, leap_outputs)
    logging.info("Saved Leap outputs to %s with shape %s.", args.output, tuple(leap_outputs.shape))

    if args.save_mano_vectors is not None:
        args.save_mano_vectors.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_mano_vectors, mano_vectors)
        logging.info(
            "Saved intermediate MANO vectors to %s with shape %s.",
            args.save_mano_vectors,
            tuple(mano_vectors.shape),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



