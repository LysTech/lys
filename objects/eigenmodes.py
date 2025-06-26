import numpy as np
from scipy.io import loadmat
from typing import List

from lys.utils.paths import lys_data_dir

#TODO: think about whether it should really be a subclass of np.ndarray,
# np.array( of a list of Eigenmode objects) will not allow arr[0].eigenvalue...

class Eigenmode(np.ndarray):
    """Represents an eigenmode as a numpy array with an associated eigenvalue."""
    
    def __new__(cls, values: np.ndarray, eigenvalue: float):
        """
        Create a new Eigenmode array.
        
        Args:
            values: Array of eigenmode values, one per vertex
            eigenvalue: The corresponding eigenvalue
            
        Returns:
            Eigenmode array with eigenvalue attribute
        """
        # Create the array using numpy's __new__ method
        obj = np.asarray(values).view(cls)
        # Add the eigenvalue as an attribute
        obj.eigenvalue = eigenvalue
        return obj
    
    def __array_finalize__(self, obj):
        """Called when creating new arrays from this one."""
        if obj is None:
            return
        # Copy the eigenvalue attribute when creating new arrays
        self.eigenvalue = getattr(obj, 'eigenvalue', None)


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

