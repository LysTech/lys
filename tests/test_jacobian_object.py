import numpy as np
import pytest
from unittest.mock import patch
import os
import h5py
from pathlib import Path

from lys.objects.jacobian import Jacobian, JacobianFactory


def test_jacobian_initialization():
    """Tests that the Jacobian can be initialized correctly."""
    data = np.random.rand(2, 3, 4, 5, 6)  # Shape (S, D, X, Y, Z)
    j = Jacobian(data, wavelength='wl1')
    np.testing.assert_array_equal(j.data, data)


def test_sample_at_vertices():
    """Tests the sample_at_vertices method of the Jacobian class."""
    # Create a simple 2x2x3x3x3 jacobian where the value is the sum of spatial indices
    # Shape is (S=2, D=2, X=3, Y=3, Z=3)
    s, d, x, y, z = np.mgrid[0:2, 0:2, 0:3, 0:3, 0:3]
    data = x + y + z  # Value depends only on spatial coordinates 
    j = Jacobian(data, wavelength='wl1')

    # Test sampling at integer coordinates (on grid points)
    vertices_on_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    # Expected shape is (N=3) where N is number of vertices
    # For vertex [0,0,0]: all values are 0, Frobenius norm = sqrt(0² * 2 * 2) = 0
    # For vertex [1,1,1]: all values are 3, Frobenius norm = sqrt(3² * 2 * 2) = sqrt(36) = 6
    # For vertex [2,2,2]: all values are 6, Frobenius norm = sqrt(6² * 2 * 2) = sqrt(144) = 12
    expected_on_grid = np.array([0., 6., 12.])
    sampled_on_grid = j.sample_at_vertices(vertices_on_grid)
    np.testing.assert_allclose(sampled_on_grid, expected_on_grid)
    assert sampled_on_grid.shape == (3,)  # (N_vertices,)

    # Test sampling between grid points (discretization to nearest integer)
    vertices_between_grid = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    # [0.5,0.5,0.5] -> [0, 0, 0] = 0, [1.5,1.5,1.5] -> [2,2,2] = 12
    expected_between_grid = np.array([0., 12.])
    sampled_between_grid = j.sample_at_vertices(vertices_between_grid)
    np.testing.assert_allclose(sampled_between_grid, expected_between_grid)
    assert sampled_between_grid.shape == (2,)  # (N_vertices,)

    # Test sampling outside the bounds (should raise AssertionError)
    vertices_outside = np.array([[-1, 0, 0], [10, 10, 10]])
    with pytest.raises(AssertionError):
        j.sample_at_vertices(vertices_outside)


def test_sample_at_vertices_unsorted():
    """Tests that sample_at_vertices works with unsorted vertex indices."""
    s, d, x, y, z = np.mgrid[0:2, 0:2, 0:3, 0:3, 0:3]
    data = x + y + z  # Value is sum of spatial indices
    j = Jacobian(data, wavelength='wl1')

    # Unsorted vertices that would fail h5py's monotonicity requirement
    vertices_unsorted = np.array([[2, 2, 2], [0, 0, 0], [1, 1, 1]])
    
    # Expected values: 
    # vertex [2,2,2]: all values are 6, Frobenius norm = sqrt(6² * 2 * 2) = sqrt(144) = 12
    # vertex [0,0,0]: all values are 0, Frobenius norm = sqrt(0² * 2 * 2) = 0
    # vertex [1,1,1]: all values are 3, Frobenius norm = sqrt(3² * 2 * 2) = sqrt(36) = 6
    # The output should be in the same order as the input vertices.
    expected_values = np.array([12., 0., 6.])
    
    sampled_unsorted = j.sample_at_vertices(vertices_unsorted)
    np.testing.assert_allclose(sampled_unsorted, expected_values)
    assert sampled_unsorted.shape == (3,)  # (N_vertices,)


def test_sample_at_vertices_modes():
    """Tests that sample_at_vertices works with different collapse modes."""
    # Create a simple 2x2x2x2x2 jacobian with known values
    # Shape is (S=2, D=2, X=2, Y=2, Z=2)
    data = np.zeros((2, 2, 2, 2, 2))
    # Set specific values for testing
    data[0, 0, 0, 0, 0] = 3  # One non-zero value at [0,0,0,0,0]
    data[1, 1, 1, 1, 1] = 4  # One non-zero value at [1,1,1,1,1]
    j = Jacobian(data, wavelength='wl1')

    vertices = np.array([[0, 0, 0], [1, 1, 1]])
    
    # Test 'fro' mode (Frobenius norm)
    # For vertex [0,0,0]: only one non-zero value (3), Frobenius norm = sqrt(3²) = 3
    # For vertex [1,1,1]: only one non-zero value (4), Frobenius norm = sqrt(4²) = 4
    expected_fro = np.array([3., 4.])
    sampled_fro = j.sample_at_vertices(vertices, mode='fro')
    np.testing.assert_allclose(sampled_fro, expected_fro)
    
    # Test 'max' mode (maximum absolute value)
    # For vertex [0,0,0]: max absolute value = 3
    # For vertex [1,1,1]: max absolute value = 4
    expected_max = np.array([3., 4.])
    sampled_max = j.sample_at_vertices(vertices, mode='max')
    np.testing.assert_allclose(sampled_max, expected_max)
    
    # Test invalid mode
    with pytest.raises(ValueError, match="Invalid mode 'invalid'. Must be 'fro' or 'max'."):
        j.sample_at_vertices(vertices, mode='invalid')


def test_load_jacobian_from_mat(tmp_path):
    """Test that load_jacobian_from_mat loads the data without transposition."""
    # Create a small test array with shape (2, 3, 4, 5, 6)
    arr = np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6)
    # Save as 'Jacobian' in a .mat file with shape (6, 5, 4, 3, 2) to simulate MATLAB order
    mat_path = tmp_path / "test_jacobian_wl2.mat"
    with h5py.File(mat_path, "w") as f:
        f.create_dataset("Jacobian", data=arr)
    # Load using our function
    jac = JacobianFactory().get(mat_path)
    # The loaded data should match the MATLAB-ordered data (no transposition performed)
    np.testing.assert_array_equal(jac.data, arr)


def test_jacobian_wavelength_extraction(tmp_path):
    """Test that the Jacobian object sets the wavelength attribute based on the file path."""
    import h5py
    arr = np.zeros((2, 2, 2, 2, 2))
    arr_matlab = np.transpose(arr, (4, 3, 2, 1, 0))
    mat_path_wl1 = tmp_path / "test_jacobian_wl1.mat"
    with h5py.File(mat_path_wl1, "w") as f:
        f.create_dataset("Jacobian", data=arr_matlab)
    jac1 = JacobianFactory().get(mat_path_wl1)
    assert jac1.wavelength == "wl1"
    mat_path_wl2 = tmp_path / "test_jacobian_wl2.mat"
    with h5py.File(mat_path_wl2, "w") as f:
        f.create_dataset("Jacobian", data=arr_matlab)
    jac2 = JacobianFactory().get(mat_path_wl2)
    assert jac2.wavelength == "wl2"


def test_find_jacobian_files(tmp_path):
    # Create plausible Jacobian files
    jacobian_file1 = tmp_path / "Jacobian_test1.mat"
    jacobian_file1.touch()
    jacobian_file2 = tmp_path / "jacobian_test2.mat"
    jacobian_file2.touch()
    unrelated_file = tmp_path / "randomfile.mat"
    unrelated_file.touch()

    from lys.objects.jacobian import _find_jacobian_files
    paths = _find_jacobian_files(tmp_path)
    assert set(p.name for p in paths) == {"Jacobian_test1.mat", "jacobian_test2.mat"}

    # Should raise if no Jacobian files
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        _find_jacobian_files(empty_dir)



