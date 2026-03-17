from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from extract_video_frames import destination_folder_for_video, extract_video_to_frames, list_videos

LOGGER = logging.getLogger("hamer.batch_video_inference")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    default_repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Run HaMeR demo.py on every video in a folder, with automatic frame extraction."
    )
    parser.add_argument(
        "--videos_dir",
        required=True,
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--frames_dir",
        required=True,
        help="Directory where per-video extracted frames will be stored.",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory where per-video HaMeR outputs will be stored.",
    )
    parser.add_argument(
        "--repo_root",
        default=str(default_repo_root),
        help="Path to the HaMeR repository root.",
    )
    parser.add_argument(
        "--demo_script",
        default="demo.py",
        help="Demo script path, relative to repo_root or absolute.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch demo.py.",
    )
    parser.add_argument(
        "--frame_image_ext",
        default="jpg",
        choices=["jpg", "png"],
        help="Format used when writing extracted frames.",
    )
    parser.add_argument(
        "--frame_digits",
        type=int,
        default=6,
        help="Zero-padding width for extracted frame filenames.",
    )
    parser.add_argument(
        "--overwrite_frames",
        action="store_true",
        help="Re-extract frames even if a frame folder already exists.",
    )
    parser.add_argument(
        "--skip_existing_frames",
        action="store_true",
        help="Reuse an existing frame folder if it already contains frames.",
    )
    parser.add_argument(
        "--skip_existing_results",
        action="store_true",
        help="Skip demo.py for videos whose result folder already contains files.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Batch size passed to demo.py.",
    )
    parser.add_argument(
        "--rescale_factor",
        type=float,
        default=2.0,
        help="Rescale factor passed to demo.py.",
    )
    parser.add_argument(
        "--body_detector",
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Body detector backend passed to demo.py.",
    )
    parser.add_argument(
        "--side_view",
        action="store_true",
        help="Pass --side_view to demo.py.",
    )
    parser.add_argument(
        "--full_frame",
        action="store_true",
        help="Pass --full_frame to demo.py.",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="Pass --save_mesh to demo.py.",
    )
    parser.add_argument(
        "--save_mano_params",
        action="store_true",
        help="Pass --save_mano_params to demo.py.",
    )
    parser.add_argument(
        "--mano_dir",
        default="pose",
        help="Subdirectory used by demo.py for MANO parameter outputs.",
    )
    parser.add_argument(
        "--mano_format",
        default="yaml",
        choices=["yaml", "json"],
        help="Serialization format used by demo.py for MANO parameters.",
    )
    return parser.parse_args()


def resolve_demo_script(repo_root: Path, demo_script: str) -> Path:
    demo_path = Path(demo_script)
    if not demo_path.is_absolute():
        demo_path = repo_root / demo_path
    if not demo_path.exists():
        raise FileNotFoundError(f"Demo script not found: {demo_path}")
    return demo_path


def has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def build_demo_command(args: argparse.Namespace, demo_script: Path, frame_dir: Path, result_dir: Path) -> list[str]:
    command = [
        args.python,
        str(demo_script),
        "--img_folder",
        str(frame_dir),
        "--out_folder",
        str(result_dir),
        "--batch_size",
        str(args.batch_size),
        "--rescale_factor",
        str(args.rescale_factor),
        "--body_detector",
        args.body_detector,
        "--file_type",
        f"*.{args.frame_image_ext}",
    ]

    if args.side_view:
        command.append("--side_view")
    if args.full_frame:
        command.append("--full_frame")
    if args.save_mesh:
        command.append("--save_mesh")
    if args.save_mano_params:
        command.extend(
            [
                "--save_mano_params",
                "--mano_dir",
                args.mano_dir,
                "--mano_format",
                args.mano_format,
            ]
        )

    return command


def main() -> None:
    args = parse_args()
    configure_logging()

    repo_root = Path(args.repo_root).resolve()
    demo_script = resolve_demo_script(repo_root, args.demo_script)
    videos_dir = Path(args.videos_dir).resolve()
    frames_root = Path(args.frames_dir).resolve()
    results_root = Path(args.results_dir).resolve()

    frames_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_dir)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {videos_dir}")

    LOGGER.info("Found %d video(s) in %s", len(videos), videos_dir)

    for video_path in videos:
        frame_dir = destination_folder_for_video(video_path, frames_root)
        result_dir = results_root / video_path.stem

        extract_video_to_frames(
            video_path=video_path,
            output_dir=frame_dir,
            image_ext=args.frame_image_ext,
            digits=args.frame_digits,
            overwrite=args.overwrite_frames,
            skip_existing=args.skip_existing_frames,
        )

        if args.skip_existing_results and has_files(result_dir):
            LOGGER.info("Skipping %s because result folder already has files: %s", video_path.name, result_dir)
            continue

        result_dir.mkdir(parents=True, exist_ok=True)
        command = build_demo_command(args, demo_script, frame_dir, result_dir)
        LOGGER.info("Running HaMeR on %s", video_path.name)
        LOGGER.info("Command: %s", " ".join(command))
        subprocess.run(command, cwd=repo_root, check=True)

    LOGGER.info("Batch inference complete. Results saved under %s", results_root)


if __name__ == "__main__":
    main()
