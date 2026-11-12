
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

MASK_DIR = ROOT / "outputs/silhouettes/masks"
IMAGE_DIR = ROOT / "capture/frames"
POSE_DIR = ROOT / "outputs/pose"

OUTPUT_DIR = ROOT / "outputs/geometry"
CONTOUR_DIR = OUTPUT_DIR / "contours"

CONTOUR_DIR.mkdir(parents=True, exist_ok=True)

VISIBILITY_THRESHOLD = 0.5
MIN_INTERVAL_WIDTH = 2

JOINTS = {
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

ARM_SEGMENTS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]

LEG_LEVELS = {
    "knees": ("left_knee", "right_knee"),
    "lower_legs": ("left_ankle", "right_ankle"),
}

TORSO_LEVELS = {
    "shoulders": 0.0,
    "upper_chest": 0.25,
    "mid_torso": 0.50,
    "waist_estimate": 0.70,
    "hips": 1.0,
}


def get_largest_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    return max(contours, key=cv2.contourArea)


def get_point(pose, name):
    point = pose[JOINTS[name]]

    if float(point.get("visibility", 0)) < VISIBILITY_THRESHOLD:
        return None

    return {
        "x": float(point["x"]),
        "y": float(point["y"]),
        "visibility": float(point["visibility"]),
    }


def midpoint(a, b):
    if a is None or b is None:
        return None

    return {
        "x": (a["x"] + b["x"]) / 2,
        "y": (a["y"] + b["y"]) / 2,
        "visibility": min(a["visibility"], b["visibility"]),
    }


def interpolate_y(a, b, ratio):
    if a is None or b is None:
        return None

    return {
        "x": a["x"] * (1 - ratio) + b["x"] * ratio,
        "y": a["y"] * (1 - ratio) + b["y"] * ratio,
        "visibility": min(a["visibility"], b["visibility"]),
    }


def get_intervals(mask, y):
    row = mask[int(np.clip(y, 0, mask.shape[0] - 1))] > 0

    padded = np.pad(row.astype(np.int8), (1, 1))
    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return [
        (int(start), int(end - 1))
        for start, end in zip(starts, ends)
        if end - start >= MIN_INTERVAL_WIDTH
    ]


def interpolate_arm_x(pose, y_normalized):
    """
    Estimate arm center positions at a given image-space height.
    Returns normalized x positions for visible arm segments.
    """
    arm_positions = []

    for start_name, end_name in ARM_SEGMENTS:
        a = get_point(pose, start_name)
        b = get_point(pose, end_name)

        if a is None or b is None:
            continue

        low = min(a["y"], b["y"])
        high = max(a["y"], b["y"])

        if high - low < 1e-6:
            continue

        if low <= y_normalized <= high:
            ratio = (y_normalized - a["y"]) / (b["y"] - a["y"])

            x = a["x"] + ratio * (b["x"] - a["x"])
            arm_positions.append({
                "x": float(x),
                "side": "left" if "left" in start_name else "right",
            })

    return arm_positions


def select_interval_near(intervals, target_x, width):
    if not intervals:
        return None

    target_px = target_x * width

    def distance(interval):
        left, right = interval

        if left <= target_px <= right:
            return 0

        return min(
            abs(target_px - left),
            abs(target_px - right),
        )

    return min(intervals, key=distance)


def measure_torso(mask, y, center_x, arm_positions):
    height, width = mask.shape
    intervals = get_intervals(mask, y)

    if not intervals:
        return None

    center_px = center_x * width

    interval = select_interval_near(
        intervals,
        center_x,
        width,
    )

    if interval is None:
        return None

    left, right = interval

    # Clip the torso interval before the estimated arm centers.
    left_arms = [
        p["x"] * width
        for p in arm_positions
        if p["side"] == "left" and p["x"] < center_x
    ]

    right_arms = [
        p["x"] * width
        for p in arm_positions
        if p["side"] == "right" and p["x"] > center_x
    ]

    # Image coordinates increase from left to right.
    if left_arms:
        left_limit = max(left_arms)
        left = max(left, int(left_limit))

    if right_arms:
        right_limit = min(right_arms)
        right = min(right, int(right_limit))

    if right < left:
        return None

    return {
        "left_px": int(left),
        "right_px": int(right),
        "width_px": int(right - left + 1),
        "width_normalized": round(
            (right - left + 1) / width,
            6,
        ),
        "method": "torso_interval_clipped_by_arm_estimates",
        "confidence": "approximate",
    }


def measure_leg(mask, y, target_x):
    height, width = mask.shape
    intervals = get_intervals(mask, y)

    if not intervals:
        return None

    interval = select_interval_near(
        intervals,
        target_x,
        width,
    )

    if interval is None:
        return None

    left, right = interval

    return {
        "left_px": left,
        "right_px": right,
        "width_px": right - left + 1,
        "width_normalized": round(
            (right - left + 1) / width,
            6,
        ),
        "method": "nearest_foreground_interval_to_joint",
        "confidence": "approximate",
    }


def process_view(mask_path):
    name = mask_path.stem

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    image = cv2.imread(
        str(IMAGE_DIR / f"{name}.png")
    )

    pose_path = POSE_DIR / f"{name}.json"

    if mask is None or image is None or not pose_path.exists():
        print(f"[SKIP] Missing input: {name}")
        return None

    height, width = image.shape[:2]

    if mask.shape != (height, width):
        mask = cv2.resize(
            mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = np.where(mask > 127, 255, 0).astype(np.uint8)

    contour = get_largest_contour(mask)

    if contour is None:
        print(f"[SKIP] No contour: {name}")
        return None

    x, y, w, h = cv2.boundingRect(contour)

    pose = json.loads(pose_path.read_text())

    points = {
        name: get_point(pose, name)
        for name in JOINTS
    }

    shoulders = midpoint(
        points["left_shoulder"],
        points["right_shoulder"],
    )

    hips = midpoint(
        points["left_hip"],
        points["right_hip"],
    )

    if shoulders is None or hips is None:
        print(f"[SKIP] Missing shoulder/hip landmarks: {name}")
        return None

    measurements = {}
    overlay = image.copy()

    # Torso measurements.
    for label, ratio in TORSO_LEVELS.items():
        level = interpolate_y(shoulders, hips, ratio)

        if level is None:
            measurements[label] = None
            continue

        scan_y = round(level["y"] * height)

        arm_positions = interpolate_arm_x(
            pose,
            level["y"],
        )

        torso = measure_torso(
            mask,
            scan_y,
            level["x"],
            arm_positions,
        )

        measurements[label] = {
            "y_px": scan_y,
            "source": (
                "observed_joint_midpoint"
                if ratio in (0.0, 1.0)
                else "interpolated_between_shoulders_and_hips"
            ),
            "measurement": torso,
        }

        if torso:
            cv2.line(
                overlay,
                (torso["left_px"], scan_y),
                (torso["right_px"], scan_y),
                (0, 200, 255),
                2,
            )

    # Independent leg measurements.
    for label, joint_names in LEG_LEVELS.items():
        left_joint = points[joint_names[0]]
        right_joint = points[joint_names[1]]

        leg_data = {}

        for side, joint in (
            ("left", left_joint),
            ("right", right_joint),
        ):
            if joint is None:
                leg_data[side] = None
                continue

            scan_y = round(joint["y"] * height)

            result = measure_leg(
                mask,
                scan_y,
                joint["x"],
            )

            leg_data[side] = {
                "y_px": scan_y,
                "measurement": result,
            }

            if result:
                cv2.line(
                    overlay,
                    (result["left_px"], scan_y),
                    (result["right_px"], scan_y),
                    (255, 150, 0),
                    2,
                )

        measurements[label] = leg_data

    # Original overall silhouette geometry.
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))

    cv2.drawContours(
        overlay,
        [contour],
        -1,
        (0, 255, 0),
        2,
    )

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2,
    )

    output = {
        "view": name,
        "bounding_box": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
        },
        "silhouette": {
            "area_px": round(area, 2),
            "perimeter_px": round(perimeter, 2),
            "aspect_ratio": round(w / h, 6),
        },
        "measurements": measurements,
        "measurement_warning": (
            "Clothed silhouette estimates, not anatomical "
            "body measurements or circumferences."
        ),
    }

    (OUTPUT_DIR / f"{name}.json").write_text(
        json.dumps(output, indent=2)
    )

    cv2.imwrite(
        str(CONTOUR_DIR / f"{name}.png"),
        overlay,
    )

    print(f"[OK] {name}: measurements extracted")

    return output


def main():
    files = sorted(MASK_DIR.glob("*.png"))

    if not files:
        print("No silhouette masks found.")
        return

    results = []

    for path in files:
        result = process_view(path)

        if result:
            results.append(result)

    (OUTPUT_DIR / "all_views.json").write_text(
        json.dumps(results, indent=2)
    )

    print(f"\nCompleted: {len(results)} views")


if __name__ == "__main__":
    main()
