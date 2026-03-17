from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import yaml

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import DEFAULT_MEAN, DEFAULT_STD, ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
LOGGER = logging.getLogger("hamer.demo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HaMeR demo with optional MANO parameter export.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--img_folder",
        type=str,
        default="images",
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default="out_demo",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--side_view",
        action="store_true",
        help="Render a side-view visualization for each detected hand.",
    )
    parser.add_argument(
        "--full_frame",
        action="store_true",
        help="Render all detected hands together in the original full-frame image.",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Save OBJ meshes for each detected hand.",
    )
    parser.add_argument(
        "--save_mano_params",
        action="store_true",
        help="Save predicted MANO parameters for each detected hand.",
    )
    parser.add_argument(
        "--mano_dir",
        type=str,
        default="pose",
        help="Subdirectory inside out_folder used for MANO parameter files.",
    )
    parser.add_argument(
        "--mano_format",
        type=str,
        default="yaml",
        choices=["yaml", "json"],
        help="Serialization format for MANO parameter export.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for hand reconstruction.",
    )
    parser.add_argument(
        "--rescale_factor",
        type=float,
        default=2.0,
        help="Factor used to pad each detected hand bounding box.",
    )
    parser.add_argument(
        "--body_detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Body detector backend. regnety usually uses less memory.",
    )
    parser.add_argument(
        "--file_type",
        "--file_patterns",
        nargs="+",
        default=["*.jpg", "*.jpeg", "*.png"],
        help="Glob patterns used to discover input images.",
    )
    parser.add_argument(
        "--person_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for person detections.",
    )
    parser.add_argument(
        "--hand_keypoint_threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for hand keypoints.",
    )
    parser.add_argument(
        "--min_hand_keypoints",
        type=int,
        default=4,
        help="Minimum number of confident hand keypoints required to keep a hand detection.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def to_serializable(obj: Any) -> Any:
    """Recursively convert tensors/arrays into plain Python containers."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_serializable(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_rgb_image(output_path: Path, image_rgb: np.ndarray) -> None:
    """Save an RGB image to disk using OpenCV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_path), image_uint8[:, :, ::-1])


def collect_image_paths(image_dir: Path, patterns: Sequence[str]) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(image_dir.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen and path.is_file():
                unique_paths.append(path)
                seen.add(resolved)
    return unique_paths


def load_detector(detector_name: str):
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if detector_name == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = (
            "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
            "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    from detectron2 import model_zoo

    detectron2_cfg = model_zoo.get_config(
        "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py",
        trained=True,
    )
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    return DefaultPredictor_Lazy(detectron2_cfg)


def detect_hands(
    image_bgr: np.ndarray,
    detector,
    pose_model: ViTPoseModel,
    person_threshold: float,
    hand_keypoint_threshold: float,
    min_hand_keypoints: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect hand boxes and handedness from an RGB image."""
    detections = detector(image_bgr)
    image_rgb = image_bgr[:, :, ::-1]

    instances = detections["instances"]
    valid_idx = (instances.pred_classes == 0) & (instances.scores > person_threshold)

    if int(valid_idx.sum()) == 0:
        return None, None

    pred_bboxes = instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = instances.scores[valid_idx].cpu().numpy()

    if pred_bboxes.size == 0:
        return None, None

    vitposes_out = pose_model.predict_pose(
        image_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes: list[list[float]] = []
    is_right: list[int] = []

    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        for keypoints, hand_side in ((left_hand_keyp, 0), (right_hand_keyp, 1)):
            valid = keypoints[:, 2] > hand_keypoint_threshold
            if int(valid.sum()) < min_hand_keypoints:
                continue

            bbox = [
                float(keypoints[valid, 0].min()),
                float(keypoints[valid, 1].min()),
                float(keypoints[valid, 0].max()),
                float(keypoints[valid, 1].max()),
            ]
            bboxes.append(bbox)
            is_right.append(hand_side)

    if not bboxes:
        return None, None

    return np.asarray(bboxes, dtype=np.float32), np.asarray(is_right, dtype=np.int64)


def build_mano_payload(
    image_name: str,
    person_id: int,
    is_right: int,
    outputs: dict[str, Any],
    batch_index: int,
    cam_t_full: np.ndarray,
) -> dict[str, Any]:
    return {
        "image_name": image_name,
        "person_id": person_id,
        "is_right": int(is_right),
        "global_orient": outputs["global_orient"][batch_index],
        "hand_pose": outputs["hand_pose"][batch_index],
        "betas": outputs["betas"][batch_index],
        "camera_translation_full": cam_t_full,
    }


def save_mano_payload(
    payload: dict[str, Any],
    output_dir: Path,
    stem: str,
    person_id: int,
    file_format: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_payload = to_serializable(payload)
    suffix = ".yaml" if file_format == "yaml" else ".json"
    output_path = output_dir / f"{stem}_{person_id}_mano_output{suffix}"

    with output_path.open("w", encoding="utf-8") as file:
        if file_format == "yaml":
            yaml.safe_dump(serializable_payload, file, sort_keys=False)
        else:
            json.dump(serializable_payload, file, indent=2)


def main() -> None:
    args = parse_args()
    configure_logging()

    out_folder = Path(args.out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading HaMeR model...")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    LOGGER.info("Loading detector and keypoint model...")
    detector = load_detector(args.body_detector)
    pose_model = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    image_paths = collect_image_paths(Path(args.img_folder), args.file_type)
    if not image_paths:
        raise FileNotFoundError(
            f"No images matched patterns {args.file_type!r} in folder: {args.img_folder}"
        )

    LOGGER.info("Found %d image(s) to process.", len(image_paths))

    for image_path in image_paths:
        LOGGER.info("Processing %s", image_path.name)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            LOGGER.warning("Skipping unreadable image: %s", image_path)
            continue

        boxes, right_flags = detect_hands(
            image_bgr=image_bgr,
            detector=detector,
            pose_model=pose_model,
            person_threshold=args.person_threshold,
            hand_keypoint_threshold=args.hand_keypoint_threshold,
            min_hand_keypoints=args.min_hand_keypoints,
        )

        if boxes is None or right_flags is None:
            LOGGER.info("No valid hand detections found for %s", image_path.name)
            continue

        dataset = ViTDetDataset(
            model_cfg,
            image_bgr,
            boxes,
            right_flags,
            rescale_factor=args.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )

        all_verts: list[np.ndarray] = []
        all_cam_t: list[np.ndarray] = []
        all_right: list[int] = []
        render_res: np.ndarray | None = None
        scaled_focal_length_value: float | None = None

        for batch in dataloader:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                outputs = model(batch)

            pred_cam = outputs["pred_cam"].clone()
            multiplier = 2 * batch["right"] - 1
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            scaled_focal_length_value = float(scaled_focal_length.detach().cpu())
            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                scaled_focal_length,
            ).detach().cpu().numpy()
            render_res = img_size[0].detach().cpu().numpy().astype(np.int32)

            batch_image_count = batch["img"].shape[0]
            for batch_index in range(batch_image_count):
                image_stem = image_path.stem
                person_id = int(batch["personid"][batch_index])

                white_img = (
                    torch.ones_like(batch["img"][batch_index]).cpu()
                    - DEFAULT_MEAN[:, None, None] / 255
                ) / (DEFAULT_STD[:, None, None] / 255)
                input_patch = (
                    batch["img"][batch_index].cpu() * (DEFAULT_STD[:, None, None] / 255)
                    + (DEFAULT_MEAN[:, None, None] / 255)
                )
                input_patch = input_patch.permute(1, 2, 0).numpy()

                pred_vertices = outputs["pred_vertices"][batch_index].detach().cpu().numpy()
                pred_cam_t_crop = outputs["pred_cam_t"][batch_index].detach().cpu().numpy()
                regression_img = renderer(
                    pred_vertices,
                    pred_cam_t_crop,
                    batch["img"][batch_index],
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                )

                if args.side_view:
                    side_img = renderer(
                        pred_vertices,
                        pred_cam_t_crop,
                        white_img,
                        mesh_base_color=LIGHT_BLUE,
                        scene_bg_color=(1, 1, 1),
                        side_view=True,
                    )
                    output_rgb = np.concatenate([input_patch, regression_img, side_img], axis=1)
                else:
                    output_rgb = np.concatenate([input_patch, regression_img], axis=1)

                write_rgb_image(out_folder / f"{image_stem}_{person_id}.png", output_rgb)

                verts = pred_vertices.copy()
                is_right = int(batch["right"][batch_index].item())
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t_full = pred_cam_t_full[batch_index]

                all_verts.append(verts)
                all_cam_t.append(cam_t_full)
                all_right.append(is_right)

                if args.save_mesh:
                    tmesh = renderer.vertices_to_trimesh(
                        verts,
                        cam_t_full.copy(),
                        LIGHT_BLUE,
                        is_right=is_right,
                    )
                    tmesh.export(out_folder / f"{image_stem}_{person_id}.obj")

                if args.save_mano_params:
                    payload = build_mano_payload(
                        image_name=image_path.name,
                        person_id=person_id,
                        is_right=is_right,
                        outputs=outputs,
                        batch_index=batch_index,
                        cam_t_full=cam_t_full,
                    )
                    save_mano_payload(
                        payload=payload,
                        output_dir=out_folder / args.mano_dir,
                        stem=image_stem,
                        person_id=person_id,
                        file_format=args.mano_format,
                    )

        if args.full_frame and all_verts and render_res is not None and scaled_focal_length_value is not None:
            full_frame_rgba = renderer.render_rgba_multiple(
                all_verts,
                cam_t=all_cam_t,
                render_res=render_res,
                is_right=all_right,
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length_value,
            )

            input_rgb = image_bgr.astype(np.float32)[:, :, ::-1] / 255.0
            input_rgba = np.concatenate(
                [input_rgb, np.ones_like(input_rgb[:, :, :1])],
                axis=2,
            )
            overlay_rgb = (
                input_rgba[:, :, :3] * (1 - full_frame_rgba[:, :, 3:])
                + full_frame_rgba[:, :, :3] * full_frame_rgba[:, :, 3:]
            )
            write_rgb_image(out_folder / f"{image_path.stem}_all.jpg", overlay_rgb)

    LOGGER.info("Done. Outputs saved to %s", out_folder)


if __name__ == "__main__":
    main()
