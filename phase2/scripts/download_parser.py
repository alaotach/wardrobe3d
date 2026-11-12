
from huggingface_hub import hf_hub_download
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "schp_pascal"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

model = hf_hub_download(
    repo_id="pirocheto/schp-pascal-7",
    filename="onnx/schp-pascal-7-int8-static.onnx",
    local_dir=MODEL_DIR,
)

print("Model downloaded:", model)
