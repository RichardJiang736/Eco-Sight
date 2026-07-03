from ultralytics.engine.results import Results

from .config import RELEVANT_CLASS_IDS, CONFIDENCE_THRESHOLD


def count_relevant_detections(results: list[Results]) -> int:
    count = 0
    boxes = results[0].boxes
    if boxes is None:
        return 0
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        if confidence > CONFIDENCE_THRESHOLD and class_id in RELEVANT_CLASS_IDS:
            count += 1
    return count


def extract_relevant_boxes(results: list[Results]) -> list[tuple[float, float, float, float, int, float]]:
    boxes = results[0].boxes
    if boxes is None:
        return []
    relevant = []
    for box in boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        if confidence > CONFIDENCE_THRESHOLD and class_id in RELEVANT_CLASS_IDS:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            relevant.append((x1, y1, x2, y2, class_id, confidence))
    return relevant
