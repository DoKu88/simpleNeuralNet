"""
Convert a list of image files to a single PDF (one image per page).
Usage:
  python ims_to_pdf.py -o output.pdf img1.png img2.jpg ...
  python ims_to_pdf.py img1.png img2.png   # writes images.pdf
"""

import argparse
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Install Pillow: pip install Pillow")


def images_to_pdf(image_paths: list[str | Path], output_path: str | Path) -> None:
    """Write a PDF with one page per image. Images are converted to RGB if necessary."""
    if not image_paths:
        raise ValueError("No image paths provided")

    paths = [Path(p) for p in image_paths]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")

    images = []
    for p in paths:
        img = Image.open(p)
        if img.mode != "RGB":
            img = img.convert("RGB")
        images.append(img)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:])
    print(f"Wrote {len(images)} page(s) to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert images to a single PDF")
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to image files (e.g. .png, .jpg)",
    )
    parser.add_argument(
        "-o", "--output",
        default="images.pdf",
        help="Output PDF path (default: images.pdf)",
    )
    args = parser.parse_args()
    images_to_pdf(args.images, args.output)


if __name__ == "__main__":
    main()
