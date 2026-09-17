"""Tile the gallery thumbnails into the README's header image (§7.2).

The README has shown a hand-made image uploaded to a GitHub user-attachment
URL since 2024: nobody can regenerate it, and it does not change when the
gallery does. This builds the same thing from the thumbnails the docs build
has just produced, so the picture at the top of the README is always the
figures the package currently makes.

The montage is written into the built site (``_static/``) and published with
it, and the README points at that URL. Nothing rendered is committed.

Usage
-----
    python scripts/make_montage.py                       # into docs/_build/html/_static
    python scripts/make_montage.py --out /tmp/g.webp --columns 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GALLERY = Path("docs/auto_examples")
DEFAULT_OUT = Path("docs/_build/html/_static/gallery.webp")

#: Themes in the order the gallery shows them, so the montage is stable
#: between builds rather than filesystem-ordered.
ORDER = (
    "stations",
    "snow",
    "sar",
    "optical",
    "terrain",
    "land",
    "hydro",
    "climate",
)


def _was_executed(thumb: Path) -> bool:
    """Whether the example behind *thumb* actually produced a figure.

    sphinx-gallery gives an example it did not run the "no image" placeholder
    as a thumbnail, resized — so the bytes differ from the placeholder on disk
    and a checksum will not recognise it. What it does *not* write is a
    numbered figure beside it, and that is the reliable test. Without it the
    montage would advertise the seven credentialed products as blank squares.
    """
    name = thumb.name.removeprefix("sphx_glr_").removesuffix("_thumb.png")
    images = thumb.parent.parent
    return any(images.glob(f"sphx_glr_{name}_0*.png"))


def thumbnails(gallery: Path) -> list[Path]:
    """Every executed example's thumbnail, in gallery order."""
    found: list[Path] = []
    themes = [t for t in ORDER if (gallery / t).is_dir()]
    themes += sorted(
        p.name for p in gallery.iterdir() if p.is_dir() and p.name not in ORDER
    )
    for theme in themes:
        thumbs = sorted((gallery / theme / "images" / "thumb").glob("sphx_glr_*.png"))
        found.extend(path for path in thumbs if _was_executed(path))
    return found


def montage(
    paths: list[Path],
    out: Path,
    *,
    columns: int = 6,
    width: int = 1800,
    gap: int = 6,
    aspect: float = 4 / 3,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    """Tile *paths* into a grid *width* pixels wide and save it to *out*.

    Each thumbnail is scaled to cover its cell and centre-cropped rather than
    fitted inside it: sphinx-gallery keeps each figure's own aspect ratio, so
    a fit-inside grid of a wide time series beside a square map is mostly
    white space.
    """
    from PIL import Image, ImageOps  # noqa: PLC0415

    if not paths:
        raise RuntimeError("No executed thumbnails found; build the gallery first.")
    columns = min(columns, len(paths))
    cell_w = (width - gap * (columns + 1)) // columns
    cell_h = int(cell_w / aspect)
    rows = -(-len(paths) // columns)
    canvas = Image.new("RGB", (width, gap + rows * (cell_h + gap)), background)

    for index, path in enumerate(paths):
        with Image.open(path) as image:
            tile = ImageOps.fit(
                image.convert("RGB"),
                (cell_w, cell_h),
                Image.LANCZOS,
                centering=(0.5, 0.5),
            )
            canvas.paste(
                tile,
                (
                    gap + (index % columns) * (cell_w + gap),
                    gap + (index // columns) * (cell_h + gap),
                ),
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".webp":
        canvas.save(out, quality=82, method=6)
    else:
        canvas.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gallery", default=str(GALLERY))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--width", type=int, default=1800)
    args = parser.parse_args(argv)

    gallery = Path(args.gallery)
    if not gallery.is_dir():
        print(
            f"No gallery at {gallery}. Run `pixi run -e docs docs-build-free` first.",
            file=sys.stderr,
        )
        return 1
    paths = thumbnails(gallery)
    out = montage(paths, Path(args.out), columns=args.columns, width=args.width)
    size = out.stat().st_size / 1024
    print(f"{len(paths)} thumbnails → {out} ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
