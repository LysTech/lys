import warnings
from dataclasses import dataclass
import numpy as np
import h5py
from pathlib import Path

from lys.utils import lys_data_dir
import os

#TODO: reflect on the multi-wavelength thing
#TODO: what about tranposing? rn we transpose nothing, seems clean?

@dataclass
class Jacobian:
    """Represents a Jacobian matrix with lazy-loaded data from an HDF5 dataset and its associated wavelength."""

    data: h5py.Dataset
    wavelength: str  # e.g. 'wl1' or 'wl2'

    
    def sample_at_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """
        Samples the Jacobian at the given vertices by discretizing coordinates to the nearest integer indices.

        Args:
            vertices: A numpy array of shape (N, 3) where N is the number of
                      vertices, and each row represents the (x, y, z)
                      coordinates of a vertex. Coordinates should be in the index space
                      of the Jacobian data (i.e., between 0 and shape-1 for each axis).

        Returns:
            A numpy array of shape (N,) containing the sampled values at the nearest indices.
        """
        # Round coordinates to nearest integer indices
        indices = np.rint(vertices).astype(int)
        # Ensure indices are within bounds
        assert np.all(indices >= 0) and np.all(indices < np.array(self.data.shape)), \
            f"Vertex coordinates out of bounds. Expected range [0, {self.data.shape}), got indices with min {indices.min()} and max {indices.max()}"
        return self.data[indices[:, 0], indices[:, 1], indices[:, 2]]

    def get_slice(self, idx):
        """
        Returns a slice of the Jacobian data.

        Args:
            idx: A tuple of indices or slices, e.g. (i, j, :, :, :) for a 3D block.
                 The number of elements in the tuple should match the number of dimensions
                 of the underlying data (e.g., 5 for a 5D array).

        Returns:
            A numpy array containing the requested slice, loaded into memory.
        """
        return self.data[idx]  # h5py.Dataset supports numpy-style slicing


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
        warnings.warn(
            "No transpose is done here because we do lazy loading; might need to correct orientation when accessing the data.",
            UserWarning
        )
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

    Args:
        session_dir: A Path object pointing to the session directory.

    Returns:
        A list of Path objects for Jacobian files.
    """
    jacobian_files = [f for f in session_dir.iterdir() if 'jacobian' in f.name.lower()]
    if not jacobian_files:
        raise FileNotFoundError(f"No Jacobian file found in {session_dir}")
    return jacobian_files

