
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "pose"
OUTPUT = ROOT / "outputs" / "normalized"

OUTPUT.mkdir(parents=True, exist_ok=True)

# MediaPipe Pose landmark indices
LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

for path in sorted(INPUT.glob("*.json")):
    points = json.loads(path.read_text())

    if len(points) != 33:
        print(f"Skipping {path.name}: expected 33 landmarks")
        continue

    # Use hip midpoint as origin.
    left_hip = np.array([points[23]["x"], points[23]["y"]])
    right_hip = np.array([points[24]["x"], points[24]["y"]])

    hip_center = (left_hip + right_hip) / 2

    # Shoulder midpoint.
    left_shoulder = np.array([points[11]["x"], points[11]["y"]])
    right_shoulder = np.array([points[12]["x"], points[12]["y"]])

    shoulder_center = (left_shoulder + right_shoulder) / 2

    # Scale using torso length.
    torso_length = np.linalg.norm(
        shoulder_center - hip_center
    )

    if torso_length < 1e-6:
        print(f"Skipping {path.name}: invalid torso length")
        continue

    normalized = {}

    for name, index in LANDMARKS.items():
        point = points[index]

        x = (point["x"] - hip_center[0]) / torso_length
        y = (point["y"] - hip_center[1]) / torso_length

        normalized[name] = {
            "x": float(x),
            "y": float(y),
            "z": float(point["z"] / torso_length),
            "visibility": float(point["visibility"]),
        }

    output_file = OUTPUT / path.name
    output_file.write_text(
        json.dumps(normalized, indent=2)
    )

    print(f"Normalized: {path.name}")

print("Done.")
