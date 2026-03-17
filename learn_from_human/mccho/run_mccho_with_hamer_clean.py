from __future__ import annotations

"""Run MCC-HO from a single image using a HaMeR hand mesh and an object mask.

This refactor keeps the original workflow but reorganizes it into smaller,
reusable functions with clearer error handling.

Run this script from the MCC-HO repository root so that imports such as
`main_mccho`, `mccho_model`, `util.hamer_utils`, and `engine_mccho` resolve.
"""

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.io.obj_io import load_obj
from pytorch3d.renderer import (
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PerspectiveCameras,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

import main_mccho
import mccho_model
import util.hamer_utils as hamer
import util.misc as misc
from engine_mccho import generate_html, generate_objs, prepare_data


DEFAULT_NUM_UNSEEN = 20_000
DEFAULT_COLOR_WEIGHT = 0.01
DEFAULT_OCCUPANCY_WEIGHT = 1.0
DEFAULT_CROP_PADDING = 50
DEFAULT_IMAGE_SIZE = 800
DEFAULT_XYZ_SIZE = 112
DEFAULT_MAX_UNSEEN_FWD = 2_000


class ScriptError(RuntimeError):
    """Raised for user-facing failures in the pipeline."""


def add_argument_if_missing(parser, *flags, **kwargs) -> None:
    existing = getattr(parser, "_option_string_actions", {})
    if any(flag in existing for flag in flags if isinstance(flag, str) and flag.startswith("-")):
        return
    parser.add_argument(*flags, **kwargs)


def ensure_file(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise ScriptError(f"{description} not found: {resolved}")
    return resolved


def ensure_directory(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_rgb_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ScriptError(f"Failed to read image: {image_path}")
    return image


def load_camera_intrinsics(camera_path: Path) -> dict[str, float]:
    with camera_path.open("r", encoding="utf-8") as handle:
        intrinsics = json.load(handle)

    required = {"fx", "fy", "px", "py"}
    missing = required.difference(intrinsics)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ScriptError(f"Camera intrinsics file is missing keys: {missing_str}")

    return {
        "fx": float(intrinsics["fx"]),
        "fy": float(intrinsics["fy"]),
        "px": float(intrinsics["px"]),
        "py": float(intrinsics["py"]),
    }


def get_silhouette_renderer(cameras: PerspectiveCameras, height: int, width: int) -> MeshRenderer:
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=150,
        bin_size=0,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=blend_params),
    )


def get_rasterizer(cameras: PerspectiveCameras, height: int, width: int) -> MeshRasterizer:
    raster_settings = RasterizationSettings(
        image_size=(height, width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


def select_hand_index(hamer_output: dict, use_right_hand: bool) -> int:
    right_flags = hamer_output["batch"]["right"]
    for idx in range(int(right_flags.shape[0])):
        if bool(right_flags[idx]) == use_right_hand:
            return idx
    return 0


def run_hamer_and_load_mesh(image_path: Path, out_dir: Path, use_right_hand: bool) -> tuple[torch.Tensor, torch.Tensor, Path]:
    hamer_out = hamer.run_demo(str(image_path), str(out_dir / "hamer"))
    hand_index = select_hand_index(hamer_out, use_right_hand)

    hand_obj_path = out_dir / "hamer" / f"{image_path.stem}_{hand_index}.obj"
    if not hand_obj_path.exists():
        raise ScriptError(f"Expected HaMeR OBJ was not produced: {hand_obj_path}")

    verts, faces, _ = load_obj(str(hand_obj_path))
    verts = verts.clone()
    verts[:, 0] *= -1
    verts[:, 2] *= -1
    faces_idx = faces.verts_idx
    return verts, faces_idx, hand_obj_path


def save_debug_hand_mesh(verts: torch.Tensor, faces: torch.Tensor, output_path: Path) -> None:
    mesh = trimesh.Trimesh(verts.detach().cpu().numpy(), faces.detach().cpu().numpy())
    mesh.export(str(output_path))


def build_cameras(height: int, width: int, intrinsics: dict[str, float]) -> PerspectiveCameras:
    return PerspectiveCameras(
        focal_length=((intrinsics["fx"], intrinsics["fy"]),),
        principal_point=((intrinsics["px"], intrinsics["py"]),),
        image_size=((height, width),),
        device="cpu",
    )


def build_hand_mesh(verts: torch.Tensor, faces: torch.Tensor) -> Meshes:
    verts_rgb = torch.ones(verts.shape, dtype=torch.float32)
    textures = TexturesVertex(verts_features=[verts_rgb])
    return Meshes(verts=[verts], faces=[faces], textures=textures)


def render_hand_mask(hand_mesh: Meshes, cameras: PerspectiveCameras, height: int, width: int) -> np.ndarray:
    renderer = get_silhouette_renderer(cameras, height, width)
    alpha = renderer(hand_mesh)[0].detach().cpu().numpy()[..., 3]
    return (alpha > 0).astype(np.uint8)


def rasterize_visible_hand_points(
    hand_mesh: Meshes,
    hand_verts: torch.Tensor,
    hand_faces: torch.Tensor,
    cameras: PerspectiveCameras,
    height: int,
    width: int,
) -> torch.Tensor:
    rasterizer = get_rasterizer(cameras, height, width)
    fragments = rasterizer(hand_mesh)

    pix_to_face = fragments.pix_to_face[0, ..., 0]
    bary_coords = fragments.bary_coords[0, ..., 0, :]

    seen_xyz = torch.full((height, width, 3), float("inf"), dtype=hand_verts.dtype)
    valid = pix_to_face >= 0
    if valid.any():
        face_indices = hand_faces[pix_to_face[valid].long()]
        pixel_tris = hand_verts[face_indices.long()]
        pixel_points = torch.einsum("ni,nij->nj", bary_coords[valid], pixel_tris)
        seen_xyz[valid] = pixel_points

    depth = seen_xyz[..., 2]
    depth[(depth < 0) | (~valid)] = float("inf")
    seen_xyz[..., 2] = depth
    return seen_xyz


def normalize_seen_xyz(seen_xyz: torch.Tensor, sd_scale: float = 3.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    finite = torch.isfinite(seen_xyz.sum(dim=-1))
    if not finite.any():
        raise ScriptError("No finite hand surface points were rasterized.")

    finite_xyz = seen_xyz[finite]
    mean = finite_xyz.mean(dim=0)
    sd = finite_xyz.std(dim=0).mean() * sd_scale
    if not torch.isfinite(sd) or sd <= 0:
        raise ScriptError("Invalid normalization scale computed from hand points.")

    normalized = (seen_xyz - mean) / sd
    return normalized, mean, sd


def prepare_seen_rgb(rgb_bgr: np.ndarray) -> torch.Tensor:
    rgb = torch.tensor(rgb_bgr, dtype=torch.float32) / 255.0
    rgb = rgb[..., [2, 1, 0]]
    return torch.nn.functional.interpolate(
        rgb.permute(2, 0, 1)[None],
        size=list(rgb.shape[:2]),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)


def load_binary_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise ScriptError(f"Failed to read object mask: {mask_path}")

    if mask.ndim == 2:
        gray = mask
    elif mask.shape[2] == 4:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
    elif mask.shape[2] == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        raise ScriptError(f"Unsupported mask shape: {mask.shape}")

    return (gray > 0).astype(np.uint8)


def pad_to_square(image: torch.Tensor, fill_value: float) -> torch.Tensor:
    height, width, channels = image.shape
    if height == width:
        return image

    if height > width:
        diff = height - width
        padding = torch.full((height, diff, channels), fill_value, dtype=image.dtype, device=image.device)
        return torch.cat([image, padding], dim=1)

    diff = width - height
    padding = torch.full((diff, width, channels), fill_value, dtype=image.dtype, device=image.device)
    return torch.cat([image, padding], dim=0)


def compute_crop_box(mask: torch.Tensor, padding: int, height: int, width: int) -> tuple[int, int, int, int]:
    if not mask.any():
        raise ScriptError("Combined hand/object mask is empty; cannot compute crop box.")

    coords = mask.nonzero(as_tuple=False)
    top = max(int(coords[:, 0].min().item()) - padding, 0)
    left = max(int(coords[:, 1].min().item()) - padding, 0)
    bottom = min(int(coords[:, 0].max().item()) + padding, height - 1)
    right = min(int(coords[:, 1].max().item()) + padding, width - 1)
    return top, left, bottom, right


def crop_and_resize_inputs(
    seen_xyz: torch.Tensor,
    seen_rgb: torch.Tensor,
    combined_mask: np.ndarray,
    crop_padding: int,
    rgb_size: int,
    xyz_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = combined_mask.shape
    mask_tensor = torch.tensor(combined_mask, dtype=torch.bool)
    top, left, bottom, right = compute_crop_box(mask_tensor, crop_padding, height, width)

    cropped_xyz = seen_xyz[top : bottom + 1, left : right + 1]
    cropped_rgb = seen_rgb[top : bottom + 1, left : right + 1]

    cropped_xyz = pad_to_square(cropped_xyz, float("inf"))
    cropped_rgb = pad_to_square(cropped_rgb, 0.0)

    resized_rgb = torch.nn.functional.interpolate(
        cropped_rgb.permute(2, 0, 1)[None],
        size=[rgb_size, rgb_size],
        mode="bilinear",
        align_corners=False,
    )

    resized_xyz = torch.nn.functional.interpolate(
        cropped_xyz.permute(2, 0, 1)[None],
        size=[xyz_size, xyz_size],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    return resized_xyz, resized_rgb


def build_samples(seen_xyz: torch.Tensor, seen_rgb: torch.Tensor, num_unseen_points: int) -> list:
    zeros = torch.zeros((num_unseen_points, 3), dtype=torch.float32)
    return [
        [seen_xyz, seen_rgb],
        [zeros.clone(), zeros.clone()],
        0,
        [[], [], []],
        [zeros.clone(), zeros.clone()],
    ]


def load_mccho_model(args, device: torch.device):
    model = mccho_model.get_mccho_model(
        occupancy_weight=DEFAULT_OCCUPANCY_WEIGHT,
        rgb_weight=DEFAULT_COLOR_WEIGHT,
        args=args,
    ).to(device)
    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)
    model.eval()
    return model


@torch.no_grad()
def run_visualization(
    model,
    samples: list,
    device: torch.device,
    args,
    seen_mean: torch.Tensor,
    seen_sd: torch.Tensor,
    output_prefix: Path,
) -> None:
    seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images, unseen_seg = prepare_data(
        samples,
        str(device),
        is_train=False,
        args=args,
        is_viz=True,
    )

    pred_occupancies: list[torch.Tensor] = []
    pred_colors: list[torch.Tensor] = []
    pred_segments: list[torch.Tensor] = []

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / DEFAULT_MAX_UNSEEN_FWD))
    for pass_idx in tqdm(range(num_passes), desc="Running MCC-HO"):
        start = pass_idx * DEFAULT_MAX_UNSEEN_FWD
        end = (pass_idx + 1) * DEFAULT_MAX_UNSEEN_FWD

        cur_unseen_xyz = unseen_xyz[:, start:end]
        cur_unseen_rgb = unseen_rgb[:, start:end].zero_()
        cur_labels = labels[:, start:end].zero_()
        cur_unseen_seg = unseen_seg[:, start:end].zero_()

        _, pred = model(
            seen_images=seen_images,
            seen_xyz=seen_xyz,
            unseen_xyz=cur_unseen_xyz,
            unseen_rgb=cur_unseen_rgb,
            unseen_occupy=cur_labels,
            cache_enc=True,
            valid_seen_xyz=valid_seen_xyz,
            unseen_seg=cur_unseen_seg,
        )

        pred_occupancies.append(pred[..., 0].cpu())

        if args.regress_color:
            color_values = 3
            pred_colors.append(pred[..., 1 : color_values + 1].reshape(-1, 3).cpu())
        else:
            color_values = 256 * 3
            color_logits = pred[..., 1 : color_values + 1].reshape(-1, 3, 256)
            color_probs = torch.nn.functional.softmax(color_logits / args.temperature, dim=2)
            color_bins = torch.linspace(0, 1, 256, device=pred.device)
            pred_colors.append((color_probs * color_bins).sum(dim=2).cpu())

        if pred.shape[-1] != 1 + color_values + 3:
            raise ScriptError(
                f"Unexpected model output dimension: got {pred.shape[-1]}, expected {1 + color_values + 3}"
            )

        seg_logits = pred[..., -3:].reshape(-1, 3)
        pred_segments.append(seg_logits.max(dim=1)[1].cpu())

    preview_image = (seen_images[0].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
    all_pred_occupancies = torch.cat(pred_occupancies, dim=1)
    all_pred_colors = torch.cat(pred_colors, dim=0)
    all_pred_segments = torch.cat(pred_segments, dim=0)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    html_path = output_prefix.with_suffix(".html")
    with html_path.open("w", encoding="utf-8") as handle:
        generate_html(
            preview_image,
            seen_xyz,
            seen_images,
            all_pred_occupancies,
            all_pred_colors,
            unseen_xyz,
            handle,
            gt_xyz=None,
            gt_rgb=None,
            mesh_xyz=None,
            pred_seg=all_pred_segments,
            score_thresholds=args.score_thresholds,
        )

    export_thresholds = [float(args.score_thresholds[0])] if args.score_thresholds else [0.1]
    generate_objs(
        all_pred_occupancies,
        all_pred_colors,
        unseen_xyz,
        str(output_prefix),
        pred_seg=all_pred_segments if args.segmentation_label else None,
        score_thresholds=export_thresholds,
        seen_mean=seen_mean,
        seen_sd=seen_sd,
    )


def configure_parser():
    parser = main_mccho.get_args_parser()

    add_argument_if_missing(parser, "--image", type=str, required=True, help="Input RGB image.")
    add_argument_if_missing(
        parser,
        "--obj-seg",
        "--obj_seg",
        dest="obj_seg",
        type=str,
        required=True,
        help="Object segmentation mask. Any non-zero pixels are treated as foreground.",
    )
    add_argument_if_missing(
        parser,
        "--cam",
        type=str,
        required=True,
        help="Camera intrinsics JSON with keys fx, fy, px, py.",
    )
    add_argument_if_missing(
        parser,
        "--hand-side",
        dest="hand_side",
        choices=["right", "left"],
        default="right",
        help="Which hand to use from HaMeR detections.",
    )
    add_argument_if_missing(
        parser,
        "--out-folder",
        "--out_folder",
        dest="out_folder",
        type=str,
        default="out_demo_mccho",
        help="Output directory.",
    )
    add_argument_if_missing(
        parser,
        "--checkpoint",
        type=str,
        default="mccho_best_checkpoint.pth",
        help="Checkpoint to load through MCC-HO's resume path.",
    )
    add_argument_if_missing(
        parser,
        "--granularity",
        type=float,
        default=0.1,
        help="Visualization granularity used by MCC-HO export.",
    )
    add_argument_if_missing(
        parser,
        "--score-thresholds",
        dest="score_thresholds",
        type=float,
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Score thresholds used in the HTML preview.",
    )
    add_argument_if_missing(
        parser,
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for color prediction when color is classified into bins.",
    )
    add_argument_if_missing(
        parser,
        "--crop-padding",
        dest="crop_padding",
        type=int,
        default=DEFAULT_CROP_PADDING,
        help="Padding in pixels added around the combined hand/object mask crop.",
    )
    add_argument_if_missing(
        parser,
        "--rgb-size",
        dest="rgb_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Target size for the RGB crop passed to MCC-HO.",
    )
    add_argument_if_missing(
        parser,
        "--xyz-size",
        dest="xyz_size",
        type=int,
        default=DEFAULT_XYZ_SIZE,
        help="Target size for the XYZ crop passed to MCC-HO.",
    )
    add_argument_if_missing(
        parser,
        "--num-unseen-points",
        dest="num_unseen_points",
        type=int,
        default=DEFAULT_NUM_UNSEEN,
        help="Number of placeholder unseen points allocated for MCC-HO inference.",
    )
    return parser


def main() -> None:
    parser = configure_parser()
    args = parser.parse_args()

    args.resume = args.checkpoint
    args.viz_granularity = args.granularity
    args.segmentation_label = True
    args.eval = True

    image_path = ensure_file(args.image, "Image")
    mask_path = ensure_file(args.obj_seg, "Object mask")
    camera_path = ensure_file(args.cam, "Camera intrinsics")
    out_dir = ensure_directory(args.out_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_bgr = load_rgb_image(image_path)
    height, width = rgb_bgr.shape[:2]

    hand_verts, hand_faces, hand_obj_path = run_hamer_and_load_mesh(
        image_path=image_path,
        out_dir=out_dir,
        use_right_hand=(args.hand_side == "right"),
    )
    print(f"Selected HaMeR hand mesh: {hand_obj_path}")

    save_debug_hand_mesh(hand_verts, hand_faces, out_dir / "input_hand.obj")

    intrinsics = load_camera_intrinsics(camera_path)
    print(
        "Camera intrinsics:",
        intrinsics["fx"],
        intrinsics["fy"],
        intrinsics["px"],
        intrinsics["py"],
    )

    cameras = build_cameras(height, width, intrinsics)
    hand_mesh = build_hand_mesh(hand_verts, hand_faces)

    hand_mask = render_hand_mask(hand_mesh, cameras, height, width)
    cv2.imwrite(str(out_dir / "hand_mask.png"), hand_mask * 255)

    seen_xyz = rasterize_visible_hand_points(hand_mesh, hand_verts, hand_faces, cameras, height, width)
    seen_xyz, seen_mean, seen_sd = normalize_seen_xyz(seen_xyz)

    seen_rgb = prepare_seen_rgb(rgb_bgr)
    object_mask = load_binary_mask(mask_path)
    cv2.imwrite(str(out_dir / "obj_mask.png"), object_mask * 255)

    combined_mask = ((object_mask + hand_mask) > 0).astype(np.uint8)
    cv2.imwrite(str(out_dir / "combined_mask.png"), combined_mask * 255)

    seen_xyz, seen_rgb = crop_and_resize_inputs(
        seen_xyz=seen_xyz,
        seen_rgb=seen_rgb,
        combined_mask=combined_mask,
        crop_padding=args.crop_padding,
        rgb_size=args.rgb_size,
        xyz_size=args.xyz_size,
    )

    model = load_mccho_model(args, device)
    samples = build_samples(seen_xyz, seen_rgb, args.num_unseen_points)
    run_visualization(
        model=model,
        samples=samples,
        device=device,
        args=args,
        seen_mean=seen_mean,
        seen_sd=seen_sd,
        output_prefix=out_dir / "output",
    )
    print(f"Done. Results written to: {out_dir}")


if __name__ == "__main__":
    main()
