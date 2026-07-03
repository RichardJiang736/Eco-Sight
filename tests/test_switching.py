import pytest
from eco_sight.switching import select_model


def test_select_model_no_detections():
    variant, fps = select_model(0)
    assert variant == "nano"
    assert fps == 5


def test_select_model_one_detection():
    variant, fps = select_model(1)
    assert variant == "medium"
    assert fps == 30


def test_select_model_three_detections():
    variant, fps = select_model(3)
    assert variant == "medium"
    assert fps == 30


def test_select_model_four_detections():
    variant, fps = select_model(4)
    assert variant == "small"
    assert fps == 5


def test_select_model_many_detections():
    variant, fps = select_model(50)
    assert variant == "small"
    assert fps == 5
