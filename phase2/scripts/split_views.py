
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "capture" / "sample.png"
OUTPUT = ROOT / "capture" / "frames"

OUTPUT.mkdir(parents=True, exist_ok=True)

image = Image.open(INPUT).convert("RGB")

width, height = image.size

# Six approximately equal horizontal view regions.
view_names = [
    "left_profile",
    "three_quarter_left",
    "front",
    "back",
    "three_quarter_right",
    "right_profile",
]

for i, name in enumerate(view_names):
    x1 = round(i * width / 6)
    x2 = round((i + 1) * width / 6)

    # Exclude the caption and watermark area.
    crop = image.crop((x1, 0, x2, height - 25))

    crop.save(OUTPUT / f"{name}.png")

    print(f"Saved: {name}.png")

print("Six views extracted.")
