
from pathlib import Path
import json
import torch

root = Path("artifacts")

metrics_path = root / "test_metrics.json"
model_path = root / "leaf_disease_mobilenet.pt"
classes_path = root / "class_names.json"

print("Metrics file exists:", metrics_path.exists())
print("Model file exists:", model_path.exists())
print("Class names file exists:", classes_path.exists())

if classes_path.exists():
    print("\nClass names file:")
    print(json.dumps(json.loads(classes_path.read_text()), indent=2))

if model_path.exists():
    payload = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )
    print("\nModel checkpoint type:", type(payload).__name__)

    if isinstance(payload, dict):
        print("Checkpoint keys:", list(payload.keys()))

        for key in ("classes", "architecture"):
            if key in payload:
                print(f"{key}:", payload[key])

        if "state_dict" in payload:
            print("Number of model tensors:", len(payload["state_dict"]))
