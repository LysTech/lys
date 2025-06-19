# tests/test_plot3d.py

import os
import numpy as np
from PIL import Image, ImageChops
import pytest

from lys.objects.mesh import from_mat
from lys.visualization.plot3d import VTKScene
from lys.utils.paths import lys_data_dir

"""
A snapshot test: it plots a mesh, saves it as a file the first time the test is run, in future tests it compares the
new image to the saved file pixel by pixel.
"""

SNAPSHOT_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def test_mesh_snapshot(tmp_path):
    """Test that rendering a mesh produces the expected image."""
    # Arrange
    mesh_path = os.path.join(lys_data_dir(), "P03/anat/meshes/P03_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_D32k.mat")
    mesh = from_mat(mesh_path, show=False)
    scene = VTKScene()
    scene.add(mesh)
    out_path = tmp_path / "mesh.png"
    snapshot_path = os.path.join(SNAPSHOT_DIR, "mesh.png")

    # Act
    scene.screenshot(str(out_path), size=(400, 400))

    # Assert
    if not os.path.exists(snapshot_path):
        # First run: save the snapshot
        os.rename(out_path, snapshot_path)
        pytest.skip("Snapshot created, rerun the test to compare.")
    else:
        assert compare_images(out_path, snapshot_path, tolerance=5), "Rendered image does not match snapshot."


def compare_images(img1_path, img2_path, tolerance=0):
    """Compare two images pixel-by-pixel. Returns True if they are the same within tolerance."""
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    diff = ImageChops.difference(img1, img2)
    np_diff = np.array(diff)
    max_diff = np.max(np_diff)
    return max_diff <= tolerance