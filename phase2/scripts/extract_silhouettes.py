
import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "capture" / "frames"
MODEL_PATH = ROOT / "models" / "selfie_segmenter.tflite"

OUTPUT_DIR = ROOT / "outputs" / "silhouettes"

MASK_DIR = OUTPUT_DIR / "masks"
CONFIDENCE_DIR = OUTPUT_DIR / "confidence"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

for directory in (MASK_DIR, CONFIDENCE_DIR, OVERLAY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5

BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image
ImageFormat = mp.ImageFormat


def clean_mask(mask):
    binary = (mask >= THRESHOLD).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5),
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = np.where(labels == largest, 255, 0).astype(np.uint8)

    return binary


def process_image(segmenter, image_path):
    print(f"\nProcessing {image_path.name}", flush=True)

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return

    height, width = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_image = Image(
        image_format=ImageFormat.SRGB,
        data=rgb,
    )

    result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        print("No confidence mask returned.")
        return

    confidence = result.confidence_masks[0].numpy_view()
    confidence = np.squeeze(confidence)

    confidence = cv2.resize(
        confidence,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    confidence = np.clip(confidence, 0.0, 1.0)

    binary = clean_mask(confidence)

    confidence_image = (confidence * 255).astype(np.uint8)

    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :] = (0, 220, 0)

    foreground = binary > 0

    overlay[foreground] = cv2.addWeighted(
        image[foreground],
        0.45,
        green[foreground],
        0.55,
        0,
    )

    cv2.imwrite(
        str(CONFIDENCE_DIR / f"{image_path.stem}.png"),
        confidence_image,
    )

    cv2.imwrite(
        str(MASK_DIR / f"{image_path.stem}.png"),
        binary,
    )

    cv2.imwrite(
        str(OVERLAY_DIR / f"{image_path.stem}.png"),
        overlay,
    )

    foreground_ratio = np.count_nonzero(binary) / binary.size

    print(
        f"Saved | {width}x{height} | "
        f"Foreground: {foreground_ratio:.1%}",
        flush=True,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Segmentation model missing: {MODEL_PATH}"
        )

    images = sorted(IMAGE_DIR.glob("*.png"))

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    options = ImageSegmenterOptions(
        base_options=BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )

    print(f"Found {len(images)} images.", flush=True)

    with ImageSegmenter.create_from_options(options) as segmenter:
        for image_path in images:
            process_image(segmenter, image_path)

    print("\nSegmentation complete.")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
