from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interpn
import h5py
import warnings

from lys.utils import lys_data_dir
import os

#TODO: implement lazy loading of the jacobian

@dataclass
class Jacobian:
    """Represents a Jacobian matrix, which is a 3D numpy array."""

    data: np.ndarray

    
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


def load_jacobians(patient: str, experiment: str, session: str) -> list[Jacobian]:
    """
    Loads all Jacobian files for a given patient, experiment, and session.

    Args:
        patient: The patient identifier (e.g., 'P03').
        experiment: The experiment name (e.g.  'fnirs_8classes').
        session: The session name (e.g. 'session-01').

    Returns:
        A list of Jacobian objects loaded from the corresponding files.
    """
    paths = _jacobian_paths(patient, experiment, session)
    out = []
    for path in paths:
        print(f"Loading Jacobian from {path}")
        jac = load_jacobian_from(path)
        out.append(jac)
    return out


def load_jacobian_from(path: str) -> Jacobian:
    """ Violates OCP but we'll add other file types later, realistically it's only one or two more types"""
    if path.endswith(".mat"):
        return load_jacobian_from_mat(path)
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def load_jacobian_from_mat(path: str) -> Jacobian:
    """
    Loads a Jacobian from a MATLAB .mat file produced by, e.g., adjoint.m.

    Args:
        path: Path to the .mat file containing the Jacobian.

    Returns:
        Jacobian: An instance of the Jacobian class with the loaded data.
    """
    with h5py.File(path, "r") as f_jac:
        J_dataset = f_jac["Jacobian"]
        J_full = J_dataset[()]
        # Transpose to (192, 256, 256, 16, 24) if original is (24, 16, 256, 256, 192)
        J = np.transpose(J_full, (4, 3, 2, 1, 0))
        warnings.warn(
            "Transposing the raw Jacobian file from MATLAB order to (N,M, 256,256,192). NOT SURE!! "
        )

        return Jacobian(J)


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

