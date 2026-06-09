#!/usr/bin/env python3
"""Create a leaderboard-compatible ZIP with images at the archive root."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", default=Path("submission.zip"), type=Path)
    parser.add_argument("--expected-count", default=1000, type=int)
    parser.add_argument("--max-size-mb", default=200, type=int)
    return parser.parse_args()


def validate_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def main() -> None:
    args = parse_args()
    images = sorted(p for p in args.input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images) != args.expected_count:
        raise SystemExit(f"expected {args.expected_count} images, found {len(images)}")

    for path in images:
        width, height = validate_image(path)
        if width < 64 or height < 64:
            raise SystemExit(f"{path} is below 64x64: {width}x{height}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for index, path in enumerate(images):
            suffix = ".jpg" if path.suffix.lower() in {".jpg", ".jpeg"} else ".png"
            zf.write(path, arcname=f"img_{index:04d}{suffix}")

    size_mb = args.output.stat().st_size / (1024 * 1024)
    if size_mb > args.max_size_mb:
        raise SystemExit(f"{args.output} is {size_mb:.1f} MB, above {args.max_size_mb} MB")
    print(f"wrote {args.output} with {len(images)} images, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
