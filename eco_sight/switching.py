from .config import SWITCHING_POLICY


def select_model(detection_count: int) -> tuple[str, int]:
    p = SWITCHING_POLICY
    if detection_count <= p["empty_threshold"]:
        return ("nano", p["idle_fps"])
    elif detection_count <= p["few_threshold"]:
        return ("medium", p["active_fps"])
    else:
        return ("small", p["idle_fps"])
