import os
import numpy as np
from PIL import Image, ImageChops
import pytest

#: Directory where test image snapshots are stored.
SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
#: Directory where failed snapshot comparisons are stored.
FAILED_SNAPSHOT_DIR = os.path.join(SNAPSHOT_DIR, "failed")

def compare_images(img1_path, img2_path, tolerance=5):
    """
    Compare two images pixel-by-pixel. Returns True if they are the same within the given tolerance.
    Args:
        img1_path (str): Path to the first image.
        img2_path (str): Path to the second image.
        tolerance (int): Maximum allowed difference per channel (0-255).
    Returns:
        bool: True if images are the same within tolerance, False otherwise.
    """
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    diff = ImageChops.difference(img1, img2)
    np_diff = np.array(diff)
    max_diff = np.max(np_diff)
    return max_diff <= tolerance 

@pytest.fixture(scope="session", autouse=True)
def clean_failed_snapshots():
    """Empty the 'snapshots/failed' directory before any tests run."""
    if os.path.exists(FAILED_SNAPSHOT_DIR):
        for filename in os.listdir(FAILED_SNAPSHOT_DIR):
            file_path = os.path.join(FAILED_SNAPSHOT_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(FAILED_SNAPSHOT_DIR, exist_ok=True) 