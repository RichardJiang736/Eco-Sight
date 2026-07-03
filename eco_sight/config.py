import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.environ.get("ECO_SIGHT_MODEL_DIR", PROJECT_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("ECO_SIGHT_RESULTS_DIR", PROJECT_ROOT / "results"))

MODEL_VARIANTS = {
    "nano": {
        "yolo_name": "yolov10n",
        "weight_file": "best_nano.pt",
        "description": "Low Energy",
        "fallback_weight": "yolov10n.pt",
    },
    "small": {
        "yolo_name": "yolov10s",
        "weight_file": "best_small.pt",
        "description": "Medium Energy",
        "fallback_weight": "yolov10s.pt",
    },
    "medium": {
        "yolo_name": "yolov10m",
        "weight_file": "best_medium.pt",
        "description": "High Energy",
        "fallback_weight": "yolov10m.pt",
    },
}

SWITCHING_POLICY = {
    "empty_threshold": 0,
    "few_threshold": 3,
    "idle_fps": 5,
    "active_fps": 30,
}

CONFIDENCE_THRESHOLD = 0.5

# nuScenes-derived class IDs (used by fine-tuned models):
#   person=7, bus=4, car=5, minibus=6, pickup=15, suv=16, truck=18
# COCO class IDs (used by base YOLOv10 weights):
#   person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7

import os as _os
_use_coco = _os.environ.get("ECO_SIGHT_COCO_CLASSES", "1").lower() in ("1", "true", "yes")

if _use_coco:
    PERSON_CLASS_ID = 0
    VEHICLE_CLASS_IDS = [2, 5, 7]  # car, bus, truck
else:
    PERSON_CLASS_ID = 7
    VEHICLE_CLASS_IDS = [4, 5, 6, 15, 16, 18]

RELEVANT_CLASS_IDS = [PERSON_CLASS_ID] + VEHICLE_CLASS_IDS
