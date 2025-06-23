import numpy as np
import pytest
from unittest.mock import patch
import os
import h5py

from lys.objects.jacobian import Jacobian, _jacobian_path, load_jacobian_from_mat


def test_jacobian_initialization():
    """Tests that the Jacobian can be initialized correctly."""
    data = np.random.rand(10, 10, 10)
    j = Jacobian(data)
    np.testing.assert_array_equal(j.data, data)


def test_sample_at_vertices():
    """Tests the sample_at_vertices method of the Jacobian class."""
    # Create a simple 3x3x3 jacobian where the value is the sum of indices
    x, y, z = np.mgrid[0:3, 0:3, 0:3]
    data = x + y + z
    j = Jacobian(data)

    # Test sampling at integer coordinates (on grid points)
    vertices_on_grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    expected_on_grid = np.array([0, 3, 6])
    sampled_on_grid = j.sample_at_vertices(vertices_on_grid)
    np.testing.assert_allclose(sampled_on_grid, expected_on_grid)

    # Test sampling between grid points (interpolation)
    vertices_between_grid = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]])
    expected_between_grid = np.array([1.5, 4.5])
    sampled_between_grid = j.sample_at_vertices(vertices_between_grid)
    np.testing.assert_allclose(sampled_between_grid, expected_between_grid)

    # Test sampling outside the bounds (should return fill_value, which is 0)
    vertices_outside = np.array([[-1, 0, 0], [10, 10, 10]])
    expected_outside = np.array([0, 0])
    sampled_outside = j.sample_at_vertices(vertices_outside)
    np.testing.assert_allclose(sampled_outside, expected_outside)


def test_jacobian_path_finds_jacobian_file():
    path = _jacobian_path('P03', 'fnirs_8classes', 'session-01')
    assert "Jacobian" in path.split("/")[-1]

    with pytest.raises(FileNotFoundError):
        path = _jacobian_path('P03', 'Not-An-Experiment', 'session-01')


def test_load_jacobian_from_mat(tmp_path):
    """Test that load_jacobian_from_mat loads and transposes the data correctly."""
    # Create a small test array with shape (2, 3, 4, 5, 6)
    arr = np.arange(2*3*4*5*6).reshape(2, 3, 4, 5, 6)
    # Save as 'Jacobian' in a .mat file with shape (6, 5, 4, 3, 2) to simulate MATLAB order
    arr_matlab = np.transpose(arr, (4, 3, 2, 1, 0))
    mat_path = tmp_path / "test_jacobian.mat"
    with h5py.File(mat_path, "w") as f:
        f.create_dataset("Jacobian", data=arr_matlab)
    # Load using our function
    jac = load_jacobian_from_mat(str(mat_path))
    # The loaded data should match the original arr
    np.testing.assert_array_equal(jac.data, arr)



