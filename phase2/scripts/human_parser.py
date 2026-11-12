
import cv2
import numpy as np
import onnxruntime as ort

from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "capture" / "frames"
MODEL_DIR = ROOT / "models" / "schp_pascal"

OUTPUT_DIR = ROOT / "outputs" / "human_parsing"
MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

for directory in (MASK_DIR, OVERLAY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

REPO = "pirocheto/schp-pascal-7"

MODEL_FILE = "onnx/schp-pascal-7-int8-static.onnx"

INPUT_SIZE = 512

MEAN = np.array(
    [0.406, 0.456, 0.485],
    dtype=np.float32,
).reshape(1, 1, 3)

STD = np.array(
    [0.225, 0.224, 0.229],
    dtype=np.float32,
).reshape(1, 1, 3)

LABELS = {
    0: "background",
    1: "head",
    2: "torso",
    3: "upper_arms",
    4: "lower_arms",
    5: "upper_legs",
    6: "lower_legs",
}

COLORS = {
    0: (0, 0, 0),
    1: (255, 180, 0),
    2: (0, 200, 0),
    3: (0, 100, 255),
    4: (255, 0, 180),
    5: (255, 200, 0),
    6: (0, 220, 255),
}


def load_model():
    print("Loading SCHP ONNX model...", flush=True)

    model_path = MODEL_DIR / "schp-pascal-7-int8-static.onnx"

    if not model_path.exists():
        model_path = Path(
            hf_hub_download(
                repo_id=REPO,
                filename=MODEL_FILE,
                local_dir=MODEL_DIR,
            )
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]

    print("Model loaded.", flush=True)
    print("Input:", input_info.name, input_info.shape, input_info.type)

    for output in session.get_outputs():
        print("Output:", output.name, output.shape, output.type)

    return session


def preprocess(image):
    # Original image is BGR. Convert to RGB.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # SCHP processor resizes directly to 512x512.
    resized = cv2.resize(
        rgb,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Convert to float32 [0, 1].
    normalized = resized.astype(np.float32) / 255.0

    # Apply SCHP normalization.
    normalized = (normalized - MEAN) / STD

    # HWC -> CHW -> NCHW.
    tensor = np.transpose(normalized, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor, dtype=np.float32)


def infer(session, image):
    tensor = preprocess(image)

    input_name = session.get_inputs()[0].name

    output_names = [
        output.name for output in session.get_outputs()
    ]

    outputs = session.run(
        output_names,
        {input_name: tensor},
    )

    # Prefer raw segmentation logits.
    logits_index = next(
        (
            i for i, name in enumerate(output_names)
            if "logits" == name.lower()
            or name.lower().endswith("/logits")
        ),
        0,
    )

    logits = outputs[logits_index]

    if logits.ndim != 4:
        raise RuntimeError(
            f"Unexpected output shape: {logits.shape}"
        )

    if logits.shape[1] != 7:
        raise RuntimeError(
            f"Expected 7 classes, got output shape {logits.shape}"
        )

    class_map = np.argmax(logits, axis=1)[0].astype(np.uint8)

    return class_map


def create_overlay(image, class_map):
    color_map = np.zeros_like(image)

    for class_id, color in COLORS.items():
        color_map[class_map == class_id] = color

    overlay = cv2.addWeighted(
        image,
        0.45,
        color_map,
        0.55,
        0,
    )

    return overlay


def process_image(session, path):
    print(f"\nProcessing {path.name}", flush=True)

    image = cv2.imread(str(path))

    if image is None:
        print("Could not read image.")
        return

    height, width = image.shape[:2]

    class_map = infer(session, image)

    # Restore segmentation to original image dimensions.
    class_map = cv2.resize(
        class_map,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = create_overlay(image, class_map)

    # Save full semantic label map.
    cv2.imwrite(
        str(MASK_DIR / f"{path.stem}_labels.png"),
        class_map,
    )

    # Save individual body-part masks.
    for class_id, label in LABELS.items():
        if class_id == 0:
            continue

        mask = (
            class_map == class_id
        ).astype(np.uint8) * 255

        cv2.imwrite(
            str(MASK_DIR / f"{path.stem}_{label}.png"),
            mask,
        )

    cv2.imwrite(
        str(OVERLAY_DIR / f"{path.stem}.png"),
        overlay,
    )

    total = class_map.size

    print(f"Image dimensions: {width}x{height}")

    for class_id, label in LABELS.items():
        ratio = np.count_nonzero(
            class_map == class_id
        ) / total

        print(f"{label:15}: {ratio:.2%}")

    print("Saved successfully.", flush=True)


def main():
    images = sorted(IMAGE_DIR.glob("*.png"))

    if not images:
        print(f"No images found in {IMAGE_DIR}")
        return

    session = load_model()

    print(f"\nFound {len(images)} images.", flush=True)

    for path in images:
        process_image(session, path)

    print("\nHuman parsing completed.")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
