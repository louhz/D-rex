from __future__ import annotations

"""Drop-in replacement for a custom multi-object tracker inside FoundationPose.

This version is designed to live inside an existing FoundationPose checkout.
It keeps the same project-level dependencies but makes the tracker code easier to
read and safer to run.

Main changes:
- Removes unsafe multi-threaded tracking over one shared camera/UI/model.
- Uses one shared RGB/depth frame per update for every object.
- Uses the same RGB frame for SAM segmentation and the first pose register step.
- Chooses the best SAM mask using SAM's IoU scores instead of a hard-coded mask index.
- Adds a CLI, logging, validation, and clearer structure.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import pygame
import torch
import trimesh
from PIL import Image
from transformers import SamModel, SamProcessor

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
sys.path.append("/foundationpose")

from estimater import (  # noqa: E402
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    dr,
    o3d,
    toOpen3dCloud,
)
from real.perception.realsense_reader_multi import RealSenseReader  # noqa: E402

DEFAULT_OBJECTS: Dict[str, str] = {
    "ketchup": "assets/ketchup.stl",
}
DEFAULT_CAMERA_SERIAL = "147122073100"
DEFAULT_DEBUG_DIR = "./real/log/foundationpose"


class SamSegmentation:
    """Point-prompted SAM wrapper."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-huge",
        device: Optional[str] = None,
        show_visualization: bool = False,
        mask_threshold: float = 0.5,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.processor = SamProcessor.from_pretrained(model_name)
        self.show_visualization = show_visualization
        self.mask_threshold = mask_threshold

    @staticmethod
    def _normalize_points(input_points: Sequence[Sequence[int]]) -> List[List[List[int]]]:
        points_array = np.asarray(input_points, dtype=int)

        if points_array.ndim == 2 and points_array.shape[1] == 2:
            return [points_array.tolist()]

        if points_array.ndim == 3 and points_array.shape[-1] == 2:
            return points_array.tolist()

        raise ValueError(
            "input_points must look like [[x, y], ...] or [[[x, y], ...]]"
        )

    def predict_mask(
        self,
        image: np.ndarray,
        input_points: Sequence[Sequence[int]],
    ) -> np.ndarray:
        normalized_points = self._normalize_points(input_points)

        if len(normalized_points) != 1:
            raise ValueError(
                "This tracker expects one point-group per object. "
                "Pass points like [[x1, y1], [x2, y2], ...]."
            )

        input_labels = [[1] * len(normalized_points[0])]
        inputs = self.processor(
            image,
            input_points=normalized_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        processed_masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        prompt_masks = processed_masks[0][0]  # shape: [num_masks, H, W]
        mask_scores = outputs.iou_scores[0][0].cpu().numpy()  # shape: [num_masks]
        best_mask_index = int(np.argmax(mask_scores))
        best_mask = prompt_masks[best_mask_index].cpu().numpy() > self.mask_threshold

        if self.show_visualization:
            self.visualize_and_save(image, best_mask, normalized_points)

        return best_mask

    def visualize_and_save(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        input_points: Sequence[Sequence[Sequence[int]]],
        save_path: str = "segmentation_result.png",
    ) -> None:
        image_array = np.asarray(image).copy()
        overlay = np.zeros_like(image_array, dtype=np.uint8)
        overlay[mask] = (0, 255, 0)

        blended = cv2.addWeighted(image_array.astype(np.uint8), 0.65, overlay, 0.35, 0)

        for point_group in input_points:
            for x, y in point_group:
                cv2.circle(blended, (int(x), int(y)), 5, (255, 0, 0), -1)

        result_image = Image.fromarray(blended)
        result_image.save(save_path)
        result_image.show()
        logging.info("Saved segmentation visualization to %s", save_path)


class PoseEstimator:
    """Tracks the pose of one object."""

    def __init__(
        self,
        object_name: str,
        mesh_file: str,
        reader: RealSenseReader,
        segmentation_model: SamSegmentation,
        prompt_points: Sequence[Sequence[int]],
        est_refine_iter: int,
        track_refine_iter: int,
        debug: int,
        debug_dir: str,
    ) -> None:
        self.object_name = object_name
        self.mesh_file = mesh_file
        self.reader = reader
        self.segmentation_model = segmentation_model
        self.prompt_points = prompt_points
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.debug_dir = debug_dir
        self.frame_index = 0

        self.mesh = trimesh.load(self.mesh_file)
        self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3)

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.estimator = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx,
        )

    def _predict_initial_mask(self, color: np.ndarray) -> np.ndarray:
        return self.segmentation_model.predict_mask(color, self.prompt_points)

    def _register(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray:
        mask = self._predict_initial_mask(color)
        pose = self.estimator.register(
            K=self.reader.K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=self.est_refine_iter,
        )

        if self.debug >= 3:
            os.makedirs(self.debug_dir, exist_ok=True)

            mesh_copy = self.mesh.copy()
            mesh_copy.apply_transform(pose)
            mesh_copy.export(f"{self.debug_dir}/{self.object_name}_model_tf.obj")

            xyz_map = depth2xyzmap(depth, self.reader.K)
            valid = depth >= 0.1
            point_cloud = toOpen3dCloud(xyz_map[valid], color[valid])
            o3d.io.write_point_cloud(
                f"{self.debug_dir}/{self.object_name}_scene_complete.ply",
                point_cloud,
            )

        return pose

    def _track(self, color: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return self.estimator.track_one_with_logging(
            rgb=color,
            depth=depth,
            K=self.reader.K,
            iteration=self.track_refine_iter,
            expriment_name=self.object_name,
        )

    def _draw_overlay(self, canvas: np.ndarray, pose: np.ndarray) -> np.ndarray:
        center_pose = pose @ np.linalg.inv(self.to_origin)
        canvas = draw_posed_3d_box(
            self.reader.K,
            img=canvas,
            ob_in_cam=center_pose,
            bbox=self.bbox,
        )
        canvas = draw_xyz_axis(
            canvas,
            ob_in_cam=center_pose,
            scale=0.1,
            K=self.reader.K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        return canvas

    def _save_debug_frame(self, vis_image: np.ndarray) -> None:
        os.makedirs(f"{self.debug_dir}/track_vis", exist_ok=True)

        reader_ids = getattr(self.reader, "id_strs", None)
        if isinstance(reader_ids, (list, tuple)) and 0 <= self.frame_index - 1 < len(reader_ids):
            frame_name = str(reader_ids[self.frame_index - 1])
        else:
            frame_name = f"{self.frame_index:06d}"

        imageio.imwrite(f"{self.debug_dir}/track_vis/{frame_name}.png", vis_image)

    def update(
        self,
        color: np.ndarray,
        depth: np.ndarray,
        canvas: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.frame_index == 0:
            pose = self._register(color, depth)
        else:
            pose = self._track(color, depth)
        self.frame_index += 1

        if canvas is not None and self.debug >= 1:
            canvas = self._draw_overlay(canvas, pose)

        if canvas is not None and self.debug >= 2:
            self._save_debug_frame(canvas)

        return pose, canvas


class MultiObjectPoseEstimator:
    """Tracks all objects using a single shared frame per update."""

    def __init__(
        self,
        objects: Dict[str, str],
        seg_points: Optional[Dict[str, Sequence[Sequence[int]]]],
        camera_serial: str,
        est_refine_iter: int,
        track_refine_iter: int,
        debug: int,
        debug_dir: str,
        image_window: Optional[pygame.Surface],
        show_sam_visualization: bool = False,
        point_selection_exposure: float = 2.0,
    ) -> None:
        if not objects:
            raise ValueError("At least one object must be provided.")

        self.objects = objects
        self.debug = debug
        self.debug_dir = debug_dir
        self.image_window = image_window
        self.reader = RealSenseReader(camera_serial)
        self.results: Dict[str, np.ndarray] = {}

        self.segmentation_model = SamSegmentation(
            show_visualization=show_sam_visualization,
        )

        if not seg_points:
            logging.info(
                "No segmentation points provided. Waiting %.1f seconds before point selection.",
                point_selection_exposure,
            )
            time.sleep(point_selection_exposure)
            color = self.reader.get_color()
            seg_points = self.select_seg_points_on_image(color)

        self.seg_points = seg_points
        self.estimators = {
            object_name: PoseEstimator(
                object_name=object_name,
                mesh_file=mesh_file,
                reader=self.reader,
                segmentation_model=self.segmentation_model,
                prompt_points=self.seg_points[object_name],
                est_refine_iter=est_refine_iter,
                track_refine_iter=track_refine_iter,
                debug=debug,
                debug_dir=debug_dir,
            )
            for object_name, mesh_file in self.objects.items()
        }

    def render_to_window(self, image: np.ndarray) -> None:
        if self.image_window is None:
            return

        resized = cv2.resize(
            image,
            (self.image_window.get_width(), self.image_window.get_height()),
        )
        surface = pygame.surfarray.make_surface(resized.swapaxes(0, 1))
        self.image_window.blit(surface, (0, 0))

    def select_seg_points_on_image(self, color: np.ndarray) -> Dict[str, List[List[int]]]:
        import matplotlib.pyplot as plt

        object_names = list(self.objects.keys())
        seg_points: Dict[str, List[List[int]]] = {name: [] for name in object_names}
        current_object_index = 0

        fig, ax = plt.subplots()
        ax.imshow(color)

        def update_title() -> None:
            current_name = object_names[current_object_index]
            ax.set_title(
                "Left click: add point | Right click: next object | Enter: finish\n"
                f"Current object: {current_name} "
                f"({current_object_index + 1}/{len(object_names)}) | "
                f"selected: {len(seg_points[current_name])}"
            )
            fig.canvas.draw_idle()

        def on_click(event) -> None:
            nonlocal current_object_index

            if event.inaxes != ax or event.xdata is None or event.ydata is None:
                return

            if event.button == 1:
                point = [int(event.xdata), int(event.ydata)]
                current_name = object_names[current_object_index]
                seg_points[current_name].append(point)
                ax.plot(point[0], point[1], "o", color=f"C{current_object_index % 10}")
                logging.info("Selected point %s for %s", point, current_name)
            elif event.button == 3:
                current_object_index = (current_object_index + 1) % len(object_names)

            update_title()

        def on_key(event) -> None:
            if event.key in {"enter", "return"}:
                plt.close(fig)

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        update_title()
        plt.show()

        missing_objects = [name for name, points in seg_points.items() if not points]
        if missing_objects:
            missing_str = ", ".join(missing_objects)
            raise ValueError(
                "Point selection finished, but these objects have no points: "
                f"{missing_str}"
            )

        return seg_points

    def update(self) -> Dict[str, np.ndarray]:
        color, depth = self.reader.get_color_depth()
        canvas = color.copy() if self.debug >= 1 else None

        results: Dict[str, np.ndarray] = {}
        for object_name, estimator in self.estimators.items():
            pose, canvas = estimator.update(color, depth, canvas)
            results[object_name] = pose

        self.results = results

        if canvas is not None:
            self.render_to_window(canvas)

        return dict(self.results)

    def get_results(self) -> Dict[str, np.ndarray]:
        return dict(self.results)

    def close(self) -> None:
        stop_method = getattr(self.reader, "stop", None)
        if callable(stop_method):
            stop_method()


def parse_object_specs(object_specs: Optional[Sequence[str]]) -> Dict[str, str]:
    if not object_specs:
        return dict(DEFAULT_OBJECTS)

    objects: Dict[str, str] = {}
    for spec in object_specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --object value: {spec!r}. Use name=mesh_path format."
            )
        name, mesh_path = spec.split("=", 1)
        name = name.strip()
        mesh_path = mesh_path.strip()
        if not name or not mesh_path:
            raise ValueError(
                f"Invalid --object value: {spec!r}. Use name=mesh_path format."
            )
        objects[name] = mesh_path
    return objects


def parse_seg_point_specs(
    seg_point_specs: Optional[Sequence[str]],
) -> Optional[Dict[str, List[List[int]]]]:
    if not seg_point_specs:
        return None

    seg_points: Dict[str, List[List[int]]] = {}
    for spec in seg_point_specs:
        if ":" not in spec or "," not in spec:
            raise ValueError(
                f"Invalid --seg-point value: {spec!r}. Use object_name:x,y format."
            )
        object_name, coords = spec.split(":", 1)
        x_str, y_str = coords.split(",", 1)
        point = [int(x_str), int(y_str)]
        seg_points.setdefault(object_name.strip(), []).append(point)
    return seg_points


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track multiple objects with FoundationPose using point prompts.",
    )
    parser.add_argument(
        "--camera-serial",
        default=DEFAULT_CAMERA_SERIAL,
        help="RealSense camera serial number.",
    )
    parser.add_argument(
        "--object",
        dest="object_specs",
        action="append",
        help="Object spec in name=mesh_path format. Repeat for multiple objects.",
    )
    parser.add_argument(
        "--seg-point",
        dest="seg_point_specs",
        action="append",
        help="Optional point prompt in object_name:x,y format. Repeat for multiple points.",
    )
    parser.add_argument(
        "--est-refine-iter",
        type=int,
        default=5,
        help="Number of register-stage refinement iterations.",
    )
    parser.add_argument(
        "--track-refine-iter",
        type=int,
        default=2,
        help="Number of track-stage refinement iterations.",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="Debug level. 0 disables drawing; larger values save more debug data.",
    )
    parser.add_argument(
        "--debug-dir",
        default=DEFAULT_DEBUG_DIR,
        help="Directory for debug outputs.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=640,
        help="Display window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=480,
        help="Display window height.",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=60,
        help="Max UI refresh rate when the pygame window is enabled.",
    )
    parser.add_argument(
        "--point-selection-exposure",
        type=float,
        default=2.0,
        help="Seconds to wait before interactive point selection starts.",
    )
    parser.add_argument(
        "--show-sam-vis",
        action="store_true",
        help="Save and display the selected SAM mask for initialization.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Disable the pygame preview window.",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Disable pose publishing over the existing topic utility.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    objects = parse_object_specs(args.object_specs)
    seg_points = parse_seg_point_specs(args.seg_point_specs)

    if seg_points is not None:
        missing_objects = [name for name in objects if name not in seg_points]
        if missing_objects:
            raise ValueError(
                "Segmentation points were provided for only some objects. Missing: "
                + ", ".join(missing_objects)
            )

    image_window: Optional[pygame.Surface] = None
    clock: Optional[pygame.time.Clock] = None

    if not args.no_window:
        pygame.init()
        image_window = pygame.display.set_mode((args.window_width, args.window_height))
        pygame.display.set_caption(
            "FoundationPose Multi-Object Tracker - " + ", ".join(objects.keys())
        )
        clock = pygame.time.Clock()

    tracker = MultiObjectPoseEstimator(
        objects=objects,
        seg_points=seg_points,
        camera_serial=args.camera_serial,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
        debug_dir=args.debug_dir,
        image_window=image_window,
        show_sam_visualization=args.show_sam_vis,
        point_selection_exposure=args.point_selection_exposure,
    )

    publisher = None
    if not args.no_publish:
        from real.utils.topic_util import Publisher, get_port

        publisher = Publisher(get_port("object_pose"))

    running = True
    try:
        while running:
            if image_window is not None:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

            results = tracker.update()
            if publisher is not None and results:
                publisher.send(results)

            if image_window is not None:
                pygame.display.flip()
                if clock is not None:
                    clock.tick(args.fps_limit)
    except KeyboardInterrupt:
        logging.info("Stopping tracker.")
    finally:
        tracker.close()
        if publisher is not None:
            publisher.stop()
        if image_window is not None:
            pygame.quit()


if __name__ == "__main__":
    main()
