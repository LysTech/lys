import warnings
from dataclasses import dataclass
import numpy as np
import h5py
from pathlib import Path

from lys.utils import lys_data_dir
import os

#TODO: reflect on the multi-wavelength thing

@dataclass
class Jacobian:
    """Represents a Jacobian matrix with lazy-loaded data from an HDF5 dataset and its associated wavelength.
    
    The internal HDF5 data has shape (D, S, X, Y, Z) but is accessed as (X, Y, Z, D, S)
    to match the expected coordinate system.
    """

    data: h5py.Dataset
    wavelength: str  # e.g. 'wl1' or 'wl2'
    
    def __post_init__(self):
        """Validate that the dataset has the expected 5D shape."""
        if len(self.data.shape) != 5:
            raise ValueError(f"Expected 5D Jacobian dataset, got shape {self.data.shape}")
    
    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        """
        Returns the transposed shape: (X, Y, Z, S, D) instead of (D, S, X, Y, Z).
        
        Returns:
            Tuple representing the transposed dimensions (X, Y, Z, S, D).
        """
        # Original shape is (D, S, X, Y, Z), we want (X, Y, Z, S, D)
        D, S, X, Y, Z = self.data.shape
        return (X, Y, Z, S, D)
    
    def _transpose_indices(self, indices):
        """
        Transpose indices from (x,y,z,s,d) to (d,s,x,y,z) for accessing the raw dataset.
        
        Args:
            indices: Tuple of indices in (x,y,z,s,d) order
            
        Returns:
            Tuple of indices in (d,s,x,y,z) order for accessing the raw dataset
        """
        if len(indices) != 5:
            raise ValueError(f"Expected 5 indices for 5D Jacobian, got {len(indices)}")
        
        x, y, z, s, d = indices
        return (d, s, x, y, z)
    
    def sample_at_vertices(self, vertices: np.ndarray, mode: str = 'fro') -> np.ndarray:
        """
        Sample Jacobian at arbitrary vertices.

        • Uses one single HDF5 point-selection when legal (no dupes in z, y, x).
        • Falls back to a per-vertex loop otherwise.
        Returns Jacobian blocks for each vertex in the original vertex order.

        Args:
            vertices: A numpy array of shape (N, 3) containing vertex coordinates in (z, y, x) order.
            mode: Collapse mode for the Jacobian blocks. Options: 'fro' (Frobenius norm) or 'max' (maximum absolute value).
                Note: Currently not used as the method returns raw Jacobian blocks.

        Returns:
            A numpy array of shape (N, S, D) where:
            - N is the number of vertices
            - S is the number of sources  
            - D is the number of detectors
            Each (S, D) block represents the Jacobian matrix at the corresponding vertex.
        """
        # ── 0. Preliminaries ───────────────────────────────────────────────────────
        idx = np.rint(vertices).astype(np.int64)          # (N, 3) in (z, y, x) order
        # self._assert_dimensions_are_correct(idx)  # Removed as requested
        D, S = self.data.shape[:2]  # D = detectors, S = sources
        N     = idx.shape[0]

        # ── 1. Remove exact duplicate vertices (keeps first occurrence) ────────────
        uniq_idx, uniq_pos, inverse = np.unique(
            idx, axis=0, return_index=True, return_inverse=True
        )

        z_u, y_u, x_u = uniq_idx.T
        # ── 2. Can we use the fast path? ───────────────────────────────────────────
        duplicates_in_any_dim = (
            len(z_u) != len(np.unique(z_u))
            or len(y_u) != len(np.unique(y_u))
            or len(x_u) != len(np.unique(x_u))
        )

        #TODO: can this be sped up?
        print("Using slow path")
        # ── 3. Fallback: loop over unique vertices ────────────────────────────────
        out = np.empty((len(uniq_idx), S, D), dtype=self.data.dtype)
        for k, (z, y, x) in enumerate(uniq_idx):
            out[k, :, :] = self.data[:, :, x, y, z].T    # transpose to (S, D) and assign to (k, S, D)
        jacobian_blocks = out[inverse]

        return jacobian_blocks
        # ── 4. Collapse Jacobian blocks to single values per vertex ─────────────────
        #return self._jacobian_to_vertex_val(jacobian_blocks, mode)

    def _assert_dimensions_are_correct(self, indices: np.ndarray) -> None:
        """
        Asserts that the given indices are within bounds for the Jacobian's spatial dimensions.
        
        Args:
            indices: A numpy array of shape (N, 3) containing integer indices for (x, y, z) coordinates.
            
        Raises:
            AssertionError: If any indices are out of bounds.
        """
        # Jacobian shape is (D, S, X, Y, Z), so we check against dimensions 2,3,4
        assert np.all(indices >= 0), \
            f"Vertex coordinates must be non-negative, got indices with min {indices.min()}"
        assert np.all(indices[:, 0] < self.data.shape[2]), \
            f"X coordinates out of bounds. Expected range [0, {self.data.shape[2]}), got max {indices[:, 0].max()}"
        assert np.all(indices[:, 1] < self.data.shape[3]), \
            f"Y coordinates out of bounds. Expected range [0, {self.data.shape[3]}), got max {indices[:, 1].max()}"
        assert np.all(indices[:, 2] < self.data.shape[4]), \
            f"Z coordinates out of bounds. Expected range [0, {self.data.shape[4]}), got max {indices[:, 2].max()}"

    def get_slice(self, idx):
        """
        Returns a slice of the Jacobian data in the transposed orientation.
        
        This method allows accessing the Jacobian as if it were stored in (X, Y, Z, D, S) order,
        while internally handling the transposition to access the raw (D, S, X, Y, Z) dataset.

        Args:
            idx: A tuple of indices or slices in (x,y,z,d,s) order, e.g. (x_slice, y_slice, z_slice, d_slice, s_slice).
                 The number of elements in the tuple should be 5 to match the 5D structure.

        Returns:
            A numpy array containing the requested slice, loaded into memory in (X, Y, Z, D, S) order.
        """
        if len(idx) != 5:
            raise ValueError(f"Expected 5 indices for 5D Jacobian, got {len(idx)}")
        
        # Transpose indices from (x,y,z,d,s) to (d,s,x,y,z)
        transposed_idx = self._transpose_indices(idx)
        
        # Get the data in raw (D, S, X, Y, Z) order
        data = self.data[transposed_idx]
        return data

    def __getitem__(self, idx):
        """
        Convenience method that delegates to get_slice for cleaner syntax.
        
        Args:
            idx: Same as get_slice method.
            
        Returns:
            Same as get_slice method.
        """
        return self.get_slice(idx)


class JacobianFactory:
    """
    tl;dr: we don't wanna load the same Jacobian file N-times if it's shared across sessions!
    this violates OCP (sorta?) because we might have to add _load_from_npy or something like that.

    GPT's docstring: 

    A factory for creating and caching Jacobian objects.

    This factory ensures that a Jacobian from a specific file is loaded only
    once. It uses the canonical path of the file as a cache key, so it
    correctly handles different paths (e.g., via symbolic links) that point to
    the same underlying file.
    """
    def __init__(self):
        self._cache = {}

    def get(self, path: Path) -> Jacobian:
        """Gets a Jacobian, using a cache to avoid redundant loads. Expects a Path object."""
        real_path = path.resolve()
        if real_path in self._cache:
            print(f"Using cached Jacobian from {real_path}")
            return self._cache[real_path]

        jacobian = self._load_from_file(path)
        self._cache[real_path] = jacobian
        return jacobian

    def _load_from_file(self, path: Path) -> Jacobian:
        """Loads a Jacobian from a file, dispatching on extension."""
        wavelength = _extract_wavelength_from_path(str(path))
        if str(path).endswith(".mat"):
            return self._load_from_mat(path, wavelength)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

    def _load_from_mat(self, path: Path, wavelength: str) -> Jacobian:
        """Loads a Jacobian from a MATLAB .mat file."""
        print(f"Lazy loading Jacobian from {path}")
        f_jac = h5py.File(str(path), "r")
        J_dataset = f_jac["Jacobian"]
        return Jacobian(J_dataset, wavelength)


_jacobian_factory = JacobianFactory()


def load_jacobians_from_session_dir(session_dir: Path) -> list[Jacobian]:
    """
    Loads all Jacobian files in a given session directory.

    Args:
        session_dir: A Path object pointing to the session directory.

    Returns:
        A list of Jacobian objects loaded from the corresponding files.
    """
    jacobian_files = _find_jacobian_files(session_dir)
    print(f"Found {len(jacobian_files)} Jacobian file(s) in {session_dir}")
    jacobian_files = _find_jacobian_files(session_dir)
    #TODO: I think it may be bad code, ordering by wavelength is sorta implicit, BAD!
    return [_jacobian_factory.get(path) for path in jacobian_files]


def _extract_wavelength_from_path(path: str) -> str:
    """
    Extracts the wavelength identifier ('wl1' or 'wl2') from the file path.

    Args:
        path: The file path to extract the wavelength from.

    Returns:
        The wavelength string ('wl1' or 'wl2').

    Raises:
        ValueError: If neither 'wl1' nor 'wl2' is found in the path.
    """
    basename = os.path.basename(path).lower()
    if 'wl1' in basename:
        return 'wl1'
    elif 'wl2' in basename:
        return 'wl2'
    else:
        raise ValueError(f"Could not determine wavelength from path: {path}")


def _find_jacobian_files(session_dir: Path) -> list[Path]:
    """
    Finds all files in the session directory with 'jacobian' in their name.
    Files are sorted by filename for consistent ordering.

    Args:
        session_dir: A Path object pointing to the session directory.

    Returns:
        A list of Path objects for Jacobian files, sorted by filename.
    """
    jacobian_files = [f for f in session_dir.iterdir() if 'jacobian' in f.name.lower()]
    if not jacobian_files:
        raise FileNotFoundError(f"No Jacobian file found in {session_dir}")
    
    return sorted(jacobian_files)

