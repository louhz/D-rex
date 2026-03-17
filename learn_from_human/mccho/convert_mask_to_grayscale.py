from __future__ import annotations

"""Convert a segmentation mask to grayscale.

Any image format supported by Pillow can be used as input. By default the
script writes `<original_stem>_gray<suffix>` next to the source file.
"""

import argparse
from pathlib import Path

from PIL import Image


class ScriptError(RuntimeError):
    """Raised for user-facing failures in the CLI."""


def convert_mask_to_grayscale(input_path: Path, output_path: Path | None = None) -> Path:
    if not input_path.is_file():
        raise ScriptError(f"Input file not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_gray{input_path.suffix}")

    with Image.open(input_path) as mask_image:
        grayscale_mask = mask_image.convert("L")
        grayscale_mask.save(output_path)

    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a mask image to grayscale.")
    parser.add_argument("input", type=str, help="Path to the input mask image.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional output path. Defaults to <input>_gray.<ext>.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    saved_path = convert_mask_to_grayscale(input_path, output_path)
    print(f"Grayscale mask saved to: {saved_path}")


if __name__ == "__main__":
    main()
