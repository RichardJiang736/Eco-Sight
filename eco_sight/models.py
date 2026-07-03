from pathlib import Path
from ultralytics import YOLO

from .config import MODEL_DIR, MODEL_VARIANTS


def _resolve_weight_path(variant: str) -> Path:
    info = MODEL_VARIANTS[variant]
    primary = MODEL_DIR / info["weight_file"]
    if primary.exists():
        return primary
    fallback = MODEL_DIR / info["fallback_weight"]
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No weight file found for variant '{variant}'. "
        f"Tried: {primary} and {fallback}. "
        f"Place trained weights in {MODEL_DIR}/ or set ECO_SIGHT_MODEL_DIR."
    )


def load_model(variant: str) -> YOLO:
    info = MODEL_VARIANTS[variant]
    weight_path = _resolve_weight_path(variant)
    model = YOLO(str(weight_path))
    model.name = info["description"]
    model.variant = variant
    return model


def load_all_models() -> dict:
    return {v: load_model(v) for v in MODEL_VARIANTS}
