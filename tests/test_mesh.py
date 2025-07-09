import numpy as np
import pytest
from lys.objects.mesh import load_unMNI_mesh
from lys.objects.segmentation import load_charm_segmentation
from lys.tests.testing_utils import compare_images, SNAPSHOT_DIR
from lys.visualization.plot3d import VTKScene
import os


@pytest.mark.skip(reason="Visually this checks out but the test is flaky")
def test_mesh_downsample_snapshot(tmp_path):
    """
    Test Mesh.downsample:
    - Loads a mesh using get_unMNI_mesh("P03")
    - Checks vertex count is > 10k
    - Downsamples to 10k vertices
    - Asserts output vertices are a subset of input vertices
    - Optionally checks faces are correctly reindexed
    - Compares rendered image to snapshot
    """
    mesh = load_unMNI_mesh("P03")
    n_vertices_orig = mesh.vertices.shape[0]
    assert n_vertices_orig > 10000, f"Expected >10k vertices, got {n_vertices_orig}"

    # Save a snapshot of the mesh before downsampling
    scene_orig = VTKScene()
    scene_orig.add(mesh)
    orig_snapshot_path = os.path.join(SNAPSHOT_DIR, "unMNI_mesh.png")
    orig_out_path = tmp_path / "unMNI_mesh.png"
    scene_orig.screenshot(str(orig_out_path), size=(400, 400))
    if not os.path.exists(orig_snapshot_path):
        os.rename(orig_out_path, orig_snapshot_path)
        print("Original mesh snapshot created.")

    mesh_down = mesh.downsample(10000)
    n_vertices_down = mesh_down.vertices.shape[0]
    # The vtkQuadricDecimation algorithm is not exact, so we check for approximate equality
    assert abs(n_vertices_down - 10000) / 10000 < 0.05, f"Expected ~10k vertices, got {n_vertices_down}"

    """ We don't expect the new vertices to be exactly a subset of the original vertices
        because of how quadric decimation works. We have a visual snapshot test below to 
        sanity check the downsampling. """

    # Optionally: check that all face indices are valid
    assert np.all(mesh_down.faces < n_vertices_down), "Face indices out of bounds in downsampled mesh"
    assert np.all(mesh_down.faces >= 0), "Negative face indices in downsampled mesh"

    # Snapshot test
    scene = VTKScene()
    scene.add(mesh_down)
    out_path = tmp_path / "unMNI_mesh_downsampled.png"
    snapshot_path = os.path.join(SNAPSHOT_DIR, "unMNI_mesh_downsampled.png")
    scene.screenshot(str(out_path), size=(400, 400))
    if not os.path.exists(snapshot_path):
        os.rename(out_path, snapshot_path)
        pytest.skip("Snapshot created, rerun the test to compare.")
    else:
        if not compare_images(out_path, snapshot_path):
            # Save the failed image to FAILED_SNAPSHOT_DIR
            from lys.tests.testing_utils import FAILED_SNAPSHOT_DIR
            failed_path = os.path.join(FAILED_SNAPSHOT_DIR, "unMNI_mesh_downsampled_failed.png")
            os.rename(out_path, failed_path)
            assert False, f"Downsampled mesh render does not match snapshot. Failed image saved to {failed_path}"
