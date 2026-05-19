import numpy as np
from medsight import visualize

def test_overlay_mask():
    img = np.ones((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    out = visualize.overlay_mask(img, mask)
    assert out.shape == (32, 32, 3)
