from __future__ import annotations

import argparse
import logging
import uuid
from pathlib import Path

LOGGER = logging.getLogger("hamer.rename_images")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"}


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely rename all images in a folder to a frame sequence.")
    parser.add_argument("--folder", required=True, help="Folder containing the images to rename.")
    parser.add_argument(
        "--prefix",
        default="frame_",
        help="Filename prefix used for the renamed images.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index for the renamed frame sequence.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=6,
        help="Zero-padding width for the frame number.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show the planned renames without modifying any files.",
    )
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_rename_plan(folder: Path, prefix: str, start_index: int, digits: int) -> list[tuple[Path, Path]]:
    images = list_images(folder)
    if not images:
        raise FileNotFoundError(f"No image files found in: {folder}")

    plan: list[tuple[Path, Path]] = []
    destinations: set[Path] = set()
    for offset, src_path in enumerate(images):
        index = start_index + offset
        dst_path = folder / f"{prefix}{index:0{digits}d}{src_path.suffix.lower()}"
        if dst_path in destinations:
            raise RuntimeError(f"Duplicate destination generated: {dst_path}")
        destinations.add(dst_path)
        plan.append((src_path, dst_path))
    return plan


def apply_rename_plan(plan: list[tuple[Path, Path]], dry_run: bool = False) -> None:
    for src_path, dst_path in plan:
        LOGGER.info("%s -> %s", src_path.name, dst_path.name)

    if dry_run:
        return

    temporary_paths: list[tuple[Path, Path]] = []
    for src_path, dst_path in plan:
        temp_path = src_path.with_name(f".__rename_tmp__{uuid.uuid4().hex}{src_path.suffix}")
        src_path.rename(temp_path)
        temporary_paths.append((temp_path, dst_path))

    for temp_path, dst_path in temporary_paths:
        temp_path.rename(dst_path)


def main() -> None:
    args = parse_args()
    configure_logging()

    folder = Path(args.folder).resolve()
    plan = build_rename_plan(
        folder=folder,
        prefix=args.prefix,
        start_index=args.start_index,
        digits=args.digits,
    )
    apply_rename_plan(plan, dry_run=args.dry_run)
    if args.dry_run:
        LOGGER.info("Dry run complete. No files were renamed.")
    else:
        LOGGER.info("Renamed %d image(s) in %s", len(plan), folder)


if __name__ == "__main__":
    main()
