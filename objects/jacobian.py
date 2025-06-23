import warnings
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interpn
import h5py

from lys.utils import lys_data_dir
import os

#TODO: implement lazy loading of the jacobian
#TODO: reflect on the multi-wavelength thing

@dataclass
class Jacobian:
    """Represents a Jacobian matrix, which is a 3D numpy array and its associated wavelength."""

    data: np.ndarray
    wavelength: str  # e.g. 'wl1' or 'wl2'

    
    def sample_at_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Samples the Jacobian at the given vertices using linear interpolation.

        Args:
            vertices: A numpy array of shape (N, 3) where N is the number of
                      vertices, and each row represents the (x, y, z)
                      coordinates of a vertex.

        Returns:
            A numpy array of shape (N,) containing the sampled values.
        """
        nx, ny, nz = self.data.shape
        points = (np.arange(nx), np.arange(ny), np.arange(nz))

        return interpn(
            points,
            self.data,
            vertices,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )


class JacobianFactory:
    """
    tl;dr: we don't wanna load the same Jacobian file N-times if it's shared across sessions!

    GPT's docstring: 

    A factory for creating and caching Jacobian objects.

    This factory ensures that a Jacobian from a specific file is loaded only
    once. It uses the canonical path of the file as a cache key, so it
    correctly handles different paths (e.g., via symbolic links) that point to
    the same underlying file.
    """
    def __init__(self):
        self._cache = {}

    def get(self, path: str) -> Jacobian:
        """Gets a Jacobian, using a cache to avoid redundant loads."""
        real_path = os.path.realpath(path)
        if real_path in self._cache:
            print(f"Using cached Jacobian from {real_path}")
            return self._cache[real_path]

        print(f"Loading Jacobian from {path}")
        jacobian = self._load_from_file(path)
        self._cache[real_path] = jacobian
        return jacobian

    def _load_from_file(self, path: str) -> Jacobian:
        """Loads a Jacobian from a file, dispatching on extension."""
        wavelength = _extract_wavelength_from_path(path)
        if path.endswith(".mat"):
            return self._load_from_mat(path, wavelength)
        else:
            raise ValueError(f"Unsupported file extension: {path}")

    def _load_from_mat(self, path: str, wavelength: str) -> Jacobian:
        """Loads a Jacobian from a MATLAB .mat file."""
        with h5py.File(path, "r") as f_jac:
            J_dataset = f_jac["Jacobian"]
            J_full = J_dataset[()]
            # Transpose to (192, 256, 256, 16, 24) if original is (24, 16, 256, 256, 192)
            J = np.transpose(J_full, (4, 3, 2, 1, 0))
            warnings.warn(
                "Transposing the raw Jacobian file from MATLAB order to (N,M, 256,256,192). NOT SURE!! "
            )

            return Jacobian(J, wavelength)


_jacobian_factory = JacobianFactory()


def load_jacobians(patient: str, experiment: str, session: str) -> list[Jacobian]:
    """
    Loads all Jacobian files for a given patient, experiment, and session.

    This function uses a factory that caches Jacobians, so if multiple sessions
    use symlinks to the same Jacobian file, the file will only be loaded once.

    Args:
        patient: The patient identifier (e.g., 'P03').
        experiment: The experiment name (e.g.  'fnirs_8classes').
        session: The session name (e.g. 'session-01').

    Returns:
        A list of Jacobian objects loaded from the corresponding files.
    """
    paths = _jacobian_paths(patient, experiment, session)
    return [_jacobian_factory.get(path) for path in paths]


def load_jacobian_from(path: str) -> Jacobian:
    """Loads a Jacobian from a file, using a cache to avoid redundant loads."""
    return _jacobian_factory.get(path)


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


def _jacobian_paths(patient: str, experiment: str, session: str) -> list[str]:
    """
    Constructs a list of paths to all Jacobian files for a given patient, experiment, and session.

    Args:
        patient: The patient identifier (e.g., 'sub-001').
        experiment: The experiment name.
        session: The session name.

    Returns:
        A list of absolute paths to Jacobian files (with 'jacobian' in their names) in the session directory.
    """
    session_dir = os.path.join(lys_data_dir(), patient, 'nirs', experiment, session)
    jacobian_files = [os.path.join(session_dir, fname)
                      for fname in os.listdir(session_dir)
                      if 'jacobian' in fname.lower()]
    if not jacobian_files:
        raise FileNotFoundError(f"No Jacobian file found in {session_dir}")
    return jacobian_files

