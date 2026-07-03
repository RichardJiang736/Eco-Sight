import pytest
from eco_sight.detector import count_relevant_detections


class MockBox:
    def __init__(self, cls, conf):
        self.cls = cls
        self.conf = conf


class MockBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self):
        return len(self._boxes)


class MockResults:
    def __init__(self, boxes):
        self.boxes = boxes


def test_count_relevant_detections_empty():
    results = [MockResults(None)]
    assert count_relevant_detections(results) == 0


def test_count_relevant_detections_no_boxes():
    results = [MockResults(MockBoxes([]))]
    assert count_relevant_detections(results) == 0


def test_count_relevant_person():
    results = [MockResults(MockBoxes([MockBox(cls=7, conf=0.9)]))]
    assert count_relevant_detections(results) == 1


def test_count_relevant_vehicle():
    results = [MockResults(MockBoxes([MockBox(cls=5, conf=0.8)]))]
    assert count_relevant_detections(results) == 1


def test_ignore_low_confidence():
    results = [MockResults(MockBoxes([MockBox(cls=7, conf=0.3)]))]
    assert count_relevant_detections(results) == 0


def test_ignore_irrelevant_class():
    results = [MockResults(MockBoxes([MockBox(cls=99, conf=0.9)]))]
    assert count_relevant_detections(results) == 0


def test_count_multiple():
    boxes = MockBoxes([
        MockBox(cls=7, conf=0.9),
        MockBox(cls=5, conf=0.8),
        MockBox(cls=6, conf=0.7),
        MockBox(cls=99, conf=0.9),
        MockBox(cls=4, conf=0.2),
    ])
    results = [MockResults(boxes)]
    assert count_relevant_detections(results) == 3
