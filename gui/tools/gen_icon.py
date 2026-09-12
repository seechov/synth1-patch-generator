#!/usr/bin/env python3
"""Render the Synth1GAN app icon.

Draws a simple, flat icon: a rounded dark square with a synth-style waveform
(GAN-inspired) and a subtle glow. Outputs:

  gui/assets/synth1gan.ico   (Windows executable icon, multi-size)
  gui/assets/synth1gan.png   (Linux .desktop icon, 256x256)
  gui/assets/synth1gan.icns  (macOS bundle icon)

Usage: python3 gui/tools/gen_icon.py
"""

import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"

# Palette
BG_TOP = (18, 22, 36)
BG_BOTTOM = (8, 10, 18)
WAVE = (94, 220, 190)      # teal
WAVE_HI = (150, 245, 220)
GLOW = (60, 160, 150)


def draw(size: int = 512) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded background with a vertical gradient.
    radius = size * 0.22
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, fill=255
    )
    grad = Image.new("RGBA", (size, size))
    gd = ImageDraw.Draw(grad)
    for y in range(size):
        t = y / size
        gd.line([(0, y), (size, y)], fill=tuple(
            int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        ) + (255,))
    img.paste(grad, (0, 0), mask)

    # Glow layer under the waveform.
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd2 = ImageDraw.Draw(glow)
    cx, cy = size / 2, size / 2
    amp = size * 0.16
    pts = []
    for i in range(size):
        x = i
        ph = (i / size) * 6.0  # ~3 full cycles
        y = cy + amp * _wave(ph)
        pts.append((x, y))
    gd2.line(pts, fill=(*GLOW, 255), width=int(size * 0.14))
    glow = glow.filter(ImageFilter.GaussianBlur(size * 0.06))
    img.alpha_composite(glow)

    # Foreground waveform.
    d = ImageDraw.Draw(img)
    amp = size * 0.16
    pts = []
    for i in range(size):
        ph = (i / size) * 6.0
        y = cy + amp * _wave(ph)
        pts.append((i, y))
    d.line(pts, fill=(*WAVE, 255), width=max(2, int(size * 0.045)), joint="curve")
    d.line(pts, fill=(*WAVE_HI, 255), width=max(1, int(size * 0.018)), joint="curve")

    return img


def _wave(ph: float) -> float:
    """A pseudo-random but smooth waveform reminiscent of a GAN latent curve."""
    import math
    return (
        0.55 * math.sin(ph)
        + 0.30 * math.sin(2.0 * ph + 0.7)
        + 0.15 * math.sin(3.4 * ph + 1.9)
    )


def _png_bytes(img: Image.Image) -> bytes:
    """Serialize an RGBA image to in-memory PNG bytes."""
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _icns_chunk(tag: bytes, data: bytes) -> bytes:
    """Build a single ICNS chunk: 4-byte type, 4-byte big-endian length."""
    return tag + struct.pack(">I", len(data) + 8) + data


def make_icns(base: Image.Image) -> bytes:
    """Assemble a modern ICNS from PNG-backed chunks.

    Uses the `ic07` (128x128) and `ic13` (256x256) PNG chunks which current
    macOS accepts without needing the legacy bit-masked formats.
    """
    chunks = []
    for tag, size in ((b"ic07", 128), (b"ic13", 256)):
        scaled = base.resize((size, size), Image.Resampling.LANCZOS)
        chunks.append(_icns_chunk(tag, _png_bytes(scaled)))

    body = b"".join(chunks)
    header = b"icns" + struct.pack(">I", len(body) + 8)
    return header + body


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = draw(512)

    # Windows .ico with a range of sizes.
    ico_path = OUT_DIR / "synth1gan.ico"
    base.save(
        ico_path,
        sizes=[
            (16, 16), (24, 24), (32, 32), (48, 48),
            (64, 64), (128, 128), (256, 256),
        ],
    )

    # 256x256 PNG for Linux .desktop and general use.
    png_path = OUT_DIR / "synth1gan.png"
    base.resize((256, 256), Image.Resampling.LANCZOS).save(png_path)

    # macOS bundle icon.
    icns_path = OUT_DIR / "synth1gan.icns"
    icns_path.write_bytes(make_icns(base))

    print(f"wrote {ico_path}")
    print(f"wrote {png_path}")
    print(f"wrote {icns_path}")


if __name__ == "__main__":
    main()
