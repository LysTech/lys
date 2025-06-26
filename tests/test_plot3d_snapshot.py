import os
import numpy as np
from PIL import Image, ImageChops
import pytest
import shutil

from lys.objects.mesh import from_mat
from lys.visualization.plot3d import VTKScene
from lys.utils.paths import lys_data_dir
from lys.objects.segmentation import load_charm_segmentation
from lys.objects.optodes import Points
from lys.objects.mesh import TimeSeriesMeshData
from lys.tests.testing_utils import SNAPSHOT_DIR, FAILED_SNAPSHOT_DIR, compare_images

"""
Snapshot tests: plot an object, take screenshot on first test run, then compare to saved image pixel by pixel in future tests.
"""

#TODO: a lot of this stuff is slow, should be possible to speec up with mocking etc.



def test_atlas_snapshot(tmp_path):
    """Test that rendering an atlas segmentation produces the expected image."""
    seg = load_charm_segmentation("P03")
    scene = VTKScene()
    scene.add(seg)
    out_path = tmp_path / "atlas.png"
    snapshot_path = os.path.join(SNAPSHOT_DIR, "atlas.png")
    scene.screenshot(str(out_path), size=(600, 600))
    if not os.path.exists(snapshot_path):
        os.rename(out_path, snapshot_path)
        pytest.skip("Snapshot created, rerun the test to compare.")
    else:
        assert compare_images(out_path, snapshot_path), "Rendered image does not match snapshot."


def test_optodes_snapshot(tmp_path):
    """Test that rendering a set of optodes (Points) produces the expected image."""
    coordinates = [
        (10.0, 20.0, 30.0),
        (40.0, 50.0, 60.0),
        (70.0, 80.0, 90.0),
        (25.0, 35.0, 45.0),
        (55.0, 65.0, 75.0)
    ]
    points = Points(coordinates)
    scene = VTKScene()
    scene.add(points)
    out_path = tmp_path / "optodes.png"
    snapshot_path = os.path.join(SNAPSHOT_DIR, "optodes.png")
    scene.screenshot(str(out_path), size=(400, 400))
    if not os.path.exists(snapshot_path):
        os.rename(out_path, snapshot_path)
        pytest.skip("Snapshot created, rerun the test to compare.")
    else:
        assert compare_images(out_path, snapshot_path), "Rendered image does not match snapshot."


def test_atlas_dynamic_legend(tmp_path):
    """
    Test dynamic legend interaction for Atlas:
    - Initial state: all labels visible (compare to atlas.png)
    - Hide a label: compare to new snapshot
    - Show the label again: should match initial snapshot
    """
    # Arrange
    seg = load_charm_segmentation("P03")
    scene = VTKScene()
    scene.add(seg)
    out_path = tmp_path / "atlas.png"
    snapshot_path = os.path.join(SNAPSHOT_DIR, "atlas.png")
    label_to_toggle = 5  # This is the outer layer, so it should be visible when we un-tick

    # --- Initial state ---
    scene.screenshot(str(out_path), size=(600, 600))
    assert compare_images(out_path, snapshot_path), "Initial atlas render does not match snapshot."

    # --- Hide label ---
    seg.toggle_label_visibility(label_to_toggle)
    out_path_hidden = tmp_path / f"atlas_label{label_to_toggle}_hidden.png"
    snapshot_path_hidden = os.path.join(SNAPSHOT_DIR, f"atlas_label{label_to_toggle}_hidden.png")
    scene.screenshot(str(out_path_hidden), size=(600, 600))
    if not os.path.exists(snapshot_path_hidden):
        os.rename(out_path_hidden, snapshot_path_hidden)
        pytest.skip("Dynamic snapshot created, rerun the test to compare.")
    else:
        assert compare_images(out_path_hidden, snapshot_path_hidden), "Atlas with label hidden does not match snapshot."

    # --- Show label again ---
    seg.toggle_label_visibility(label_to_toggle)
    out_path_restored = tmp_path / f"atlas_label{label_to_toggle}_restored.png"
    scene.screenshot(str(out_path_restored), size=(600, 600))
    # Should match the original snapshot
    assert compare_images(out_path_restored, snapshot_path), "Atlas after re-showing label does not match original snapshot."
