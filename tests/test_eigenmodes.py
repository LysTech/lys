import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from lys.objects.eigenmodes import Eigenmode, load_eigenmodes


def test_eigenmode_creation():
    """Test creating an Eigenmode object."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    eigenvalue = 0.5
    
    eigenmode = Eigenmode(values, eigenvalue)
    
    assert eigenmode.values is values
    assert eigenmode.eigenvalue == eigenvalue
    assert len(eigenmode) == 4


def test_eigenmode_indexing():
    """Test indexing into eigenmode values."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    eigenvalue = 0.5
    
    eigenmode = Eigenmode(values, eigenvalue)
    
    assert eigenmode[0] == 1.0
    assert eigenmode[1] == 2.0
    assert eigenmode[-1] == 4.0


def test_load_eigenmodes():
    """Test loading eigenmodes from a mock MATLAB file."""
    # Mock data - eigenvalues stored as 2D array [[v1, v2, ...]]
    # eigenmodes_array shape: (N_vertices, N_eigenmodes) = (3, 2) - 3 vertices, 2 eigenmodes
    mock_eigenmodes_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)
    mock_eigenvalues = np.array([[0.1, 0.2]])  # 2D array as stored in MATLAB file
    
    # Mock the loadmat function
    with patch('lys.objects.eigenmodes.loadmat') as mock_loadmat:
        mock_mdata = {
            'eigenmodes': mock_eigenmodes_array,
            'eigenvalues': mock_eigenvalues
        }
        mock_loadmat.return_value = mock_mdata
        
        # Mock the path to avoid file system dependencies
        with patch('lys.objects.eigenmodes.lys_data_dir') as mock_lys_data_dir:
            mock_lys_data_dir.return_value = MagicMock()
            
            eigenmodes = load_eigenmodes("P03")
            
            assert len(eigenmodes) == 2
            assert eigenmodes[0].eigenvalue == 0.1
            assert eigenmodes[1].eigenvalue == 0.2
            # Each eigenmode.values should be a 1D array of length N_vertices (3)
            np.testing.assert_array_equal(eigenmodes[0].values, np.array([1.0, 3.0, 5.0]))
            np.testing.assert_array_equal(eigenmodes[1].values, np.array([2.0, 4.0, 6.0]))


def test_load_eigenmodes_assertion_error():
    """Test that load_eigenmodes raises an assertion error when eigenvalues don't match eigenmodes."""
    # Mock data with mismatched dimensions
    # eigenmodes_array shape: (N_vertices, N_eigenmodes) = (3, 2) - 3 vertices, 2 eigenmodes
    mock_eigenmodes_array = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)
    # eigenvalues shape: (1, N_eigenvalues) = (1, 1) - only 1 eigenvalue instead of 2
    mock_eigenvalues = np.array([[0.1]])  # Only one eigenvalue for two eigenmodes, stored as 2D
    
    # Mock the loadmat function
    with patch('lys.objects.eigenmodes.loadmat') as mock_loadmat:
        mock_mdata = {
            'eigenmodes': mock_eigenmodes_array,
            'eigenvalues': mock_eigenvalues
        }
        mock_loadmat.return_value = mock_mdata
        
        # Mock the path to avoid file system dependencies
        with patch('lys.objects.eigenmodes.lys_data_dir') as mock_lys_data_dir:
            mock_lys_data_dir.return_value = MagicMock()
            
            with pytest.raises(AssertionError, match="Number of eigenvalues must match number of eigenmodes"):
                load_eigenmodes("P03") 