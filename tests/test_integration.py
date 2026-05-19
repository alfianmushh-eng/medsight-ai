Integration tests for medsight pipeline.
from pathlib import Path
import numpy as np
from PIL import Image

def test_end_to_end_preprocess():
    from medsight import preprocess
    from medsight import augment
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    gray = preprocess.to_grayscale(img)
    assert gray.shape == (64, 64)
    norm = preprocess.min_max_normalize(gray)
    assert 0.0 <= norm.min() and norm.max() <= 1.0
    flipped = augment.horizontal_flip(gray)
    assert flipped.shape == gray.shape

def test_end_to_end_metrics():
    from medsight import metrics
    y = np.array([0, 1, 1, 0, 1])
    m = metrics.binary_classification(y, y)
    assert abs(m.f1 - 1.0) < 1e-6
    dice = metrics.dice_score(y.astype(bool), y.astype(bool))
    assert abs(dice - 1.0) < 1e-6

def test_models_import():
    from medsight.models import SimpleCNN, UNet
    assert SimpleCNN is not None
    assert UNet is not None
