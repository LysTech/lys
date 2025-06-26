import numpy as np
from scipy.io import loadmat
from typing import List

from lys.utils.paths import lys_data_dir


class Eigenmode:
    """Represents an eigenmode with its associated eigenvalue."""
    
    def __init__(self, values: np.ndarray, eigenvalue: float):
        """
        Initialize an eigenmode.
        
        Args:
            values: Array of eigenmode values, one per vertex
            eigenvalue: The corresponding eigenvalue
        """
        self.values = values
        self.eigenvalue = eigenvalue
        
    def __len__(self) -> int:
        """Return the number of vertices in this eigenmode."""
        return len(self.values)
    
    def __getitem__(self, index):
        """Allow indexing into the eigenmode values."""
        return self.values[index]


def load_eigenmodes(patient: str) -> List[Eigenmode]:
    """
    Load eigenmodes from a MATLAB file for a given patient.
    
    Args:
        patient: Patient identifier (e.g., "P03")
        
    Returns:
        List of Eigenmode objects, each containing values and eigenvalue
    """
    path = lys_data_dir() / patient / "anat" / "meshes" / f"{patient}_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_unMNI_D32k_eigenmodes.mat"
    mdata = loadmat(path)
    eigenmodes_array = mdata["eigenmodes"]
    eigenvals = mdata["eigenvalues"][0]
    assert len(eigenvals) == eigenmodes_array.shape[1], "Number of eigenvalues must match number of eigenmodes"
    
    # Create Eigenmode objects
    eigenmodes = []
    for i in range(eigenmodes_array.shape[1]):
        eigenmode = Eigenmode(eigenmodes_array[:, i], eigenvals[i])
        eigenmodes.append(eigenmode)
    
    return eigenmodes 