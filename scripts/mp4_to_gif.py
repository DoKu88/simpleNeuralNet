"""
mp4_to_gif.py

Convert one or more .mp4 files to .gif using ffmpeg.

Usage:
    python scripts/mp4_to_gif.py path/to/video.mp4
    python scripts/mp4_to_gif.py path/to/video.mp4 --fps 15 --width 640
    python scripts/mp4_to_gif.py outputs/*.mp4 --out-dir outputs/gifs

Options:
    --fps      Frames per second in the output GIF (default: 10)
    --width    Width in pixels; height scales proportionally (default: 480)
    --out-dir  Directory for output GIFs (default: same dir as input file)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert(src: Path, out_dir: Path, fps: int, width: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.with_suffix(".gif").name

    # Two-pass ffmpeg: generate palette first for better colour quality
    palette = dst.with_suffix(".png")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen",
                str(palette),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src), "-i", str(palette),
                "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[v];[v][1:v]paletteuse",
                str(dst),
            ],
            check=True,
            capture_output=True,
        )
    finally:
        palette.unlink(missing_ok=True)

    return dst


def main() -> None:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found — install it with: brew install ffmpeg")

    parser = argparse.ArgumentParser(description="Convert mp4 files to GIF")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input .mp4 file(s)")
    parser.add_argument("--fps", type=int, default=10, help="Output framerate (default 10)")
    parser.add_argument("--width", type=int, default=480, help="Output width px (default 480)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    for src in args.inputs:
        if not src.exists():
            print(f"skip (not found): {src}")
            continue
        out_dir = args.out_dir if args.out_dir else src.parent
        dst = convert(src, out_dir, fps=args.fps, width=args.width)
        print(f"saved: {dst}")


if __name__ == "__main__":
    main()
