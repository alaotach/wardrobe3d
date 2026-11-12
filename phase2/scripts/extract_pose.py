
from pathlib import Path
import gc
import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "capture" / "frames"
OUTPUT = ROOT / "outputs" / "pose"

OUTPUT.mkdir(parents=True, exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model = ROOT / "models" / "pose_landmarker_lite.task"

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model)),
    running_mode=RunningMode.IMAGE,
    num_poses=1,
    output_segmentation_masks=False,
)

images = sorted(INPUT.glob("*.png"))

print(f"Found {len(images)} images", flush=True)

with PoseLandmarker.create_from_options(options) as landmarker:
    for path in images:
        print(f"Processing: {path.name}", flush=True)

        image = cv2.imread(str(path))
        if image is None:
            print(f"Could not read {path}", flush=True)
            continue

        height, width = image.shape[:2]
        print(f"Dimensions: {width}x{height}", flush=True)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = landmarker.detect(mp_image)

        print(
            f"Pose detected: {bool(result.pose_landmarks)}",
            flush=True,
        )

        if result.pose_landmarks:
            landmarks = [
                {
                    "x": point.x,
                    "y": point.y,
                    "z": point.z,
                    "visibility": point.visibility,
                }
                for point in result.pose_landmarks[0]
            ]

            import json
            output_file = OUTPUT / f"{path.stem}.json"
            output_file.write_text(
                json.dumps(landmarks, indent=2)
            )

            print(f"Saved: {output_file}", flush=True)

        del result, mp_image, rgb, image
        gc.collect()

print("Finished.", flush=True)
