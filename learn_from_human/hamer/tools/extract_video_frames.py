from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable

import cv2

LOGGER = logging.getLogger("hamer.extract_video_frames")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".webm"}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from one video or a folder of videos.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a video file or a directory containing videos.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where extracted frame folders will be created.",
    )
    parser.add_argument(
        "--image_ext",
        default="jpg",
        choices=["jpg", "png"],
        help="Image format used for saved frames.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Zero-padding width for frame filenames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing frame directory before extraction.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip extraction if the destination already contains frames.",
    )
    return parser.parse_args()


def list_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path.suffix}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    videos = sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return videos


def destination_folder_for_video(video_path: Path, output_root: Path) -> Path:
    return output_root / video_path.stem


def extract_video_to_frames(
    video_path: Path,
    output_dir: Path,
    image_ext: str = "jpg",
    digits: int = 6,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> int:
    if output_dir.exists() and any(output_dir.iterdir()):
        if skip_existing:
            LOGGER.info("Skipping %s because %s already contains frames.", video_path.name, output_dir)
            return len(list(output_dir.iterdir()))
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Destination already exists and is not empty: {output_dir}. "
                "Use --overwrite or --skip_existing."
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_path}")

    frame_index = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_path = output_dir / f"{frame_index:0{digits}d}.{image_ext}"
            cv2.imwrite(str(frame_path), frame)
            frame_index += 1
    finally:
        cap.release()

    LOGGER.info("Extracted %d frames from %s into %s", frame_index, video_path.name, output_dir)
    return frame_index


def main() -> None:
    args = parse_args()
    configure_logging()

    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    videos = list_videos(input_path)
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {input_path}")

    LOGGER.info("Found %d video(s) to process.", len(videos))
    for video_path in videos:
        extract_video_to_frames(
            video_path=video_path,
            output_dir=destination_folder_for_video(video_path, output_root),
            image_ext=args.image_ext,
            digits=args.digits,
            overwrite=args.overwrite,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
