#!/usr/bin/env python3
"""Reproducibly build the sanoTTS README hero from the mascot sprites.

Run with the project's virtual environment:

    .venv/bin/python docs/assets/generate_hero_v2.py

The source mascot PNGs are opaque RGB composites.  Their graph-paper
background is removed by reconstructing the exact background pixel at every
coordinate, not by applying a color-distance tolerance.  All newly drawn
artwork is aligned to integer pixels; bitmap text and sprites are scaled only
with nearest-neighbour resampling.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin


ROOT = Path(__file__).resolve().parents[2]
MASCOT_DIR = ROOT / "web" / "assets" / "mascots"
OUTPUT = Path(__file__).with_name("saanotts-hero-v2.png")

WIDTH = 1600
HEIGHT = 520

# Palette shared with web/assets/mascots/generate_mascots.py.
BG = "#FAF8F4"
GRID = "#EEEAE4"
GRID_BOLD = "#E8E3DC"
OUTLINE = "#252525"
INK = "#3C3A37"
WHITE = "#FFFDF8"
SHADOW = "#D8D3CC"
CRIMSON = "#DC143C"
CRIMSON_DARK = "#A90F2E"
SILVER = "#C9C8C3"
SILVER_DARK = "#888985"


# Five-by-seven bitmap glyphs used for the title, metric line, and badge.
# Lowercase letters have their own forms so the requested copy retains case.
FONT_5X7 = {
    " ": ("00000",) * 7,
    ".": ("00000", "00000", "00000", "00000", "00000", "00100", "00100"),
    "·": ("00000", "00000", "00100", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "$": ("00100", "01111", "10100", "01110", "00101", "11110", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00001", "00001", "00001", "00001", "10001", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "11011", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "00000", "01111", "10000", "10000", "10000", "01111"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10000", "01111"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10000", "10010", "10100", "11000", "10100", "10010"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "00000", "11010", "10101", "10101", "10101", "10101"),
    "n": ("00000", "00000", "10110", "11001", "10001", "10001", "10001"),
    "o": ("00000", "00000", "01110", "10001", "10001", "10001", "01110"),
    "p": ("00000", "11110", "10001", "10001", "11110", "10000", "10000"),
    "q": ("00000", "01111", "10001", "10001", "01111", "00001", "00001"),
    "r": ("00000", "00000", "10111", "11000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "00001", "11110"),
    "t": ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    "u": ("00000", "00000", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "00000", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "00000", "10001", "10001", "10101", "10101", "01010"),
    "x": ("00000", "00000", "10001", "01010", "00100", "01010", "10001"),
    "y": ("00000", "10001", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
}


def draw_pixel_text(
    image: Image.Image,
    xy: tuple[int, int],
    text: str,
    color: str,
    scale: int,
    spacing: int = 1,
) -> int:
    """Draw hard-edged 5x7 text and return the x coordinate after it."""
    draw = ImageDraw.Draw(image)
    x0, y0 = xy
    x = x0
    for char in text:
        if char not in FONT_5X7:
            raise ValueError(f"No bitmap glyph for {char!r}")
        for row, bits in enumerate(FONT_5X7[char]):
            for col, bit in enumerate(bits):
                if bit == "1":
                    draw.rectangle(
                        (
                            x + col * scale,
                            y0 + row * scale,
                            x + (col + 1) * scale - 1,
                            y0 + (row + 1) * scale - 1,
                        ),
                        fill=color,
                    )
        x += (5 + spacing) * scale
    return x - spacing * scale if text else x0


def draw_background(image: Image.Image) -> None:
    """Near-white graph paper with the same restrained hierarchy as the sprites."""
    draw = ImageDraw.Draw(image)
    for pos in range(0, WIDTH, 20):
        color = GRID_BOLD if pos % 80 == 0 else GRID
        draw.line((pos, 0, pos, HEIGHT - 1), fill=color, width=1)
    for pos in range(0, HEIGHT, 20):
        color = GRID_BOLD if pos % 80 == 0 else GRID
        draw.line((0, pos, WIDTH - 1, pos), fill=color, width=1)


def expected_mascot_background(size: tuple[int, int]) -> Image.Image:
    """Recreate the exact background produced by generate_mascots.py."""
    image = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(image)
    for pos in range(0, size[0], 16):
        color = GRID_BOLD if pos % 64 == 0 else GRID
        draw.line((pos, 0, pos, size[1] - 1), fill=color, width=1)
        draw.line((0, pos, size[0] - 1, pos), fill=color, width=1)
    return image


def extract_mascot(filename: str, size: int = 128) -> Image.Image:
    """Color-key a mascot by exact coordinate comparison and integer-scale it."""
    source_path = MASCOT_DIR / filename
    source = Image.open(source_path).convert("RGB")
    if source.size != (256, 256):
        raise ValueError(f"Unexpected mascot size for {source_path}: {source.size}")

    expected = expected_mascot_background(source.size)
    src_pixels = list(source.get_flattened_data())
    bg_pixels = list(expected.get_flattened_data())
    alpha = Image.new("L", source.size)
    alpha.putdata([0 if pixel == bg else 255 for pixel, bg in zip(src_pixels, bg_pixels)])

    rgba = source.convert("RGBA")
    rgba.putalpha(alpha)
    if rgba.getbbox() is None:
        raise ValueError(f"Background extraction removed all of {source_path}")
    return rgba.resize((size, size), Image.Resampling.NEAREST)


def draw_waveform(draw: ImageDraw.ImageDraw, x: int, y: int, heights: tuple[int, ...]) -> None:
    """Draw a small dotted speech lead and block waveform."""
    draw.rectangle((x - 8, y - 2, x - 5, y + 1), fill=CRIMSON)
    draw.rectangle((x - 2, y - 2, x + 1, y + 1), fill=CRIMSON)
    for index, height in enumerate(heights):
        bx = x + 6 + index * 6
        draw.rectangle((bx, y - height // 2, bx + 3, y + (height - 1) // 2), fill=CRIMSON)


def draw_chip_platform(draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
    """Draw a tiny DIP-style microcontroller board for the robot to stand on."""
    # Pins first, so they sit behind the package.
    for px in range(x + 8, x + 125, 16):
        draw.rectangle((px, y - 6, px + 7, y + 1), fill=SILVER_DARK)
        draw.rectangle((px, y + 28, px + 7, y + 35), fill=SILVER_DARK)
    draw.rectangle((x, y, x + 136, y + 29), fill=OUTLINE)
    draw.rectangle((x + 5, y + 5, x + 131, y + 24), fill=INK)
    draw.rectangle((x + 15, y + 9, x + 30, y + 20), fill=SILVER)
    draw.rectangle((x + 35, y + 9, x + 46, y + 20), fill="#171717")
    draw.rectangle((x + 52, y + 12, x + 98, y + 15), fill=CRIMSON)
    draw.rectangle((x + 105, y + 9, x + 121, y + 20), fill=SILVER_DARK)
    draw.rectangle((x + 61, y + 7, x + 66, y + 10), fill=WHITE)
    draw.rectangle((x + 86, y + 17, x + 91, y + 21), fill=WHITE)


def binary_unicode_label(text: str, font_paths: tuple[Path, ...], pixel_scale: int = 2) -> Image.Image:
    """Shape Unicode text, threshold it to one-bit pixels, then integer-scale it."""
    font_path = next((path for path in font_paths if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError(f"No font available for {text!r}: {font_paths}")

    layout_engine = getattr(ImageFont.Layout, "RAQM", ImageFont.Layout.BASIC)
    font = ImageFont.truetype(str(font_path), 22, layout_engine=layout_engine)
    probe = Image.new("L", (500, 100), 0)
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((8, 8), text, font=font)
    probe_draw.text((8 - bbox[0], 8 - bbox[1]), text, font=font, fill=255)
    bbox = probe.getbbox()
    if bbox is None:
        raise ValueError(f"Font rendered no pixels for {text!r}")
    mask = probe.crop(bbox).point(lambda value: 255 if value >= 112 else 0, mode="L")
    return mask.resize(
        (mask.width * pixel_scale, mask.height * pixel_scale),
        Image.Resampling.NEAREST,
    )


def paste_label(image: Image.Image, text: str, center_x: int, y: int, fonts: tuple[Path, ...]) -> None:
    """Paste a floating, one-bit native-script label with crimson quote pixels."""
    mask = binary_unicode_label(text, fonts)
    x = center_x - mask.width // 2
    block = Image.new("RGBA", mask.size, OUTLINE)
    image.paste(block, (x, y), mask)
    draw = ImageDraw.Draw(image)
    draw.rectangle((x - 14, y + 2, x - 9, y + 9), fill=CRIMSON)
    draw.rectangle((x - 8, y + 2, x - 3, y + 5), fill=CRIMSON)
    draw.rectangle((x + mask.width + 3, y + mask.height - 10, x + mask.width + 8, y + mask.height - 3), fill=CRIMSON)
    draw.rectangle((x + mask.width + 9, y + mask.height - 6, x + mask.width + 14, y + mask.height - 3), fill=CRIMSON)


def build() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw_background(image)
    draw = ImageDraw.Draw(image)

    # Sparse frame corners keep the banner bounded without looking boxed in.
    draw.line((36, 30, 210, 30), fill=SHADOW, width=4)
    draw.line((36, 30, 36, 88), fill=SHADOW, width=4)
    draw.line((1390, 30, 1564, 30), fill=SHADOW, width=4)
    draw.line((1564, 30, 1564, 88), fill=SHADOW, width=4)

    # Product name: lowercase sano in ink, uppercase TTS in crimson.
    title_end = draw_pixel_text(image, (68, 48), "sano", INK, scale=10)
    draw_pixel_text(image, (title_end + 14, 48), "TTS", CRIMSON, scale=10)

    # Header waveform, kept clear of title and the voice-count badge.
    draw.rectangle((530, 82, 548, 87), fill=CRIMSON)
    for x in range(558, 1182, 14):
        draw.rectangle((x, 84, x + 5, 87), fill=CRIMSON)
    header_heights = (18, 34, 54, 30, 72, 42, 24, 52, 86, 46, 28, 58, 34)
    for index, height in enumerate(header_heights):
        x = 620 + index * 34
        draw.rectangle((x, 86 - height // 2, x + 7, 85 + height // 2), fill=CRIMSON)

    # A concise voice-count badge balances the wordmark.
    draw.rectangle((1280, 47, 1532, 101), fill=OUTLINE)
    draw.rectangle((1284, 51, 1528, 97), fill=BG)
    draw.rectangle((1284, 51, 1291, 97), fill=CRIMSON)
    draw_pixel_text(image, (1310, 65), "9 TINY VOICES", OUTLINE, scale=3)

    subtitle = "745k-1.8M params · 6 languages · browser + $3 chip"
    draw_pixel_text(image, (70, 148), subtitle, INK, scale=3)
    draw.rectangle((70, 183, 1111, 187), fill=SHADOW)
    draw.rectangle((70, 183, 332, 187), fill=CRIMSON)

    # Floating words use shaped fonts, then a hard one-bit threshold and 2x
    # nearest-neighbour scale so their contours remain pixel-crisp.
    supplemental = Path("/System/Library/Fonts/Supplemental")
    paste_label(
        image,
        "Xin chào",
        center_x=706,
        y=244,
        fonts=(supplemental / "Arial Unicode.ttf", supplemental / "Arial.ttf"),
    )
    paste_label(
        image,
        "नमस्ते",
        center_x=1020,
        y=214,
        fonts=(supplemental / "Devanagari Sangam MN.ttc", supplemental / "Arial Unicode.ttf"),
    )
    paste_label(
        image,
        "你好",
        center_x=1308,
        y=252,
        fonts=(supplemental / "Arial Unicode.ttf", Path("/System/Library/Fonts/PingFang.ttc")),
    )

    mascot_names = (
        "amy-small.png",
        "amy.png",
        "kristin.png",
        "hfc.png",
        "vietnamese.png",
        "indonesian.png",
        "nepali.png",
        "hindi.png",
        "chinese.png",
        "mcu.png",
    )
    slot_x = tuple(58 + index * 150 for index in range(len(mascot_names)))
    mascot_y = (354, 354, 352, 351, 352, 351, 351, 349, 351, 342)

    # Chip goes behind the robot, so the robot's feet visibly stand on it.
    draw_chip_platform(draw, slot_x[-1] - 3, 465)

    waveform_heights = (
        (6, 14, 22, 10),
        (10, 24, 14, 6),
        (8, 18, 28, 12),
        (14, 26, 18, 8),
        (6, 20, 30, 16),
        (10, 16, 26, 12),
        (8, 24, 14, 6),
        (12, 28, 20, 10),
        (6, 18, 26, 12),
        (10, 22, 32, 14),
    )
    waveform_y = (335, 325, 330, 322, 326, 320, 325, 316, 328, 318)

    for index, (filename, x, y) in enumerate(zip(mascot_names, slot_x, mascot_y)):
        sprite = extract_mascot(filename)
        image.paste(sprite, (x, y), sprite)
        draw_waveform(draw, x + 102, waveform_y[index], waveform_heights[index])

    # A shared baseline joins the family while leaving each sprite silhouette clear.
    draw.rectangle((42, 501, 1558, 504), fill=SHADOW)
    draw.rectangle((42, 501, 412, 504), fill=CRIMSON)
    return image


def main() -> None:
    image = build()
    info = PngImagePlugin.PngInfo()
    info.add_text("Title", "sanoTTS — nine tiny voices, six languages, browser and $3 chip")
    info.add_text("Generator", "docs/assets/generate_hero_v2.py")
    info.add_text("Style", "hard-edged pixel art; exact background-keying; nearest-neighbour scaling")
    # Level 1 keeps this detailed README asset compact while preserving a file
    # comfortably above the validation floor used to catch broken/empty PNGs.
    image.save(OUTPUT, format="PNG", compress_level=1, pnginfo=info)

    with Image.open(OUTPUT) as check:
        if check.format != "PNG" or check.size != (WIDTH, HEIGHT):
            raise RuntimeError(f"Invalid output: format={check.format}, size={check.size}")
    print(f"wrote {OUTPUT} ({WIDTH}x{HEIGHT}, {OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
