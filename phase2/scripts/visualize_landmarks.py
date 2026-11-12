
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "capture" / "frames"
LANDMARK_DIR = ROOT / "outputs" / "pose"
OUTPUT_DIR = ROOT / "outputs" / "overlays"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Major anatomical landmarks only.
LANDMARKS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

CONNECTIONS = [
    (11, 12),       # Shoulders
    (11, 23),       # Left torso
    (12, 24),       # Right torso
    (23, 24),       # Hips
    (11, 13),       # Left upper arm
    (13, 15),       # Left forearm
    (12, 14),       # Right upper arm
    (14, 16),       # Right forearm
    (23, 25),       # Left thigh
    (25, 27),       # Left lower leg
    (24, 26),       # Right thigh
    (26, 28),       # Right lower leg
]

VISIBILITY_THRESHOLD = 0.5
UPSCALE = 4

def get_visible(landmark):
    return float(landmark.get("visibility", 0.0)) >= VISIBILITY_THRESHOLD


def draw_view(image, landmarks, title):
    height, width = image.shape[:2]

    points = {}

    for index in LANDMARKS:
        point = landmarks[index]

        if not get_visible(point):
            continue

        x = round(point["x"] * width)
        y = round(point["y"] * height)

        if 0 <= x < width and 0 <= y < height:
            points[index] = (x, y)

    # Draw connections.
    for start, end in CONNECTIONS:
        if start not in points or end not in points:
            continue

        cv2.line(
            image,
            points[start],
            points[end],
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    # Draw landmarks.
    for index, (x, y) in points.items():
        cv2.circle(
            image,
            (x, y),
            4,
            (0, 0, 255),
            -1,
            cv2.LINE_AA,
        )

    # Add title.
    cv2.rectangle(image, (0, 0), (width, 25), (245, 245, 245), -1)
    cv2.putText(
        image,
        title,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    return image, len(points)


def main():
    files = sorted(LANDMARK_DIR.glob("*.json"))

    if not files:
        print("No landmark files found.")
        return

    print(f"Processing {len(files)} views...")

    contact_tiles = []

    for path in files:
        image_path = IMAGE_DIR / f"{path.stem}.png"

        if not image_path.exists():
            print(f"[SKIP] Missing image: {image_path.name}")
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[SKIP] Cannot read: {image_path.name}")
            continue

        landmarks = json.loads(path.read_text())

        if len(landmarks) != 33:
            print(f"[SKIP] Invalid landmark count: {path.name}")
            continue

        image, visible_count = draw_view(
            image,
            landmarks,
            path.stem,
        )

        # Enlarge without altering the source coordinates.
        enlarged = cv2.resize(
            image,
            None,
            fx=UPSCALE,
            fy=UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )

        output_path = OUTPUT_DIR / f"{path.stem}.png"
        cv2.imwrite(str(output_path), enlarged)

        print(
            f"[OK] {path.stem}: "
            f"{visible_count}/12 visible -> {output_path.name}"
        )

        # Create a normalized tile for the comparison sheet.
        tile_width = 320
        tile_height = 420

        scale = min(
            tile_width / enlarged.shape[1],
            (tile_height - 30) / enlarged.shape[0],
        )

        resized = cv2.resize(
            enlarged,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

        tile = np.full(
            (tile_height, tile_width, 3),
            245,
            dtype=np.uint8,
        )

        y_offset = 30 + (tile_height - 30 - resized.shape[0]) // 2
        x_offset = (tile_width - resized.shape[1]) // 2

        tile[
            y_offset:y_offset + resized.shape[0],
            x_offset:x_offset + resized.shape[1],
        ] = resized

        cv2.putText(
            tile,
            path.stem,
            (8, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        contact_tiles.append(tile)

    if contact_tiles:
        columns = 3
        rows = (len(contact_tiles) + columns - 1) // columns

        sheet = np.full(
            (rows * 420, columns * 320, 3),
            230,
            dtype=np.uint8,
        )

        for i, tile in enumerate(contact_tiles):
            row = i // columns
            col = i % columns

            sheet[
                row * 420:(row + 1) * 420,
                col * 320:(col + 1) * 320,
            ] = tile

        sheet_path = OUTPUT_DIR / "six_view_comparison.png"
        cv2.imwrite(str(sheet_path), sheet)

        print(f"\nComparison sheet: {sheet_path}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
