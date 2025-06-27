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
        Sample Jacobian at arbitrary vertices.

        • Uses one single HDF5 point-selection when legal (no dupes in x, y, z).
        • Falls back to a per-vertex loop otherwise.
        Both paths return an array shaped (S, D, N) in the *original* vertex order.
        """
        # ── 0. Preliminaries ───────────────────────────────────────────────────────
        idx = np.rint(vertices).astype(np.int64)          # (N, 3)
        self._assert_dimensions_are_correct(idx)
        S, D = self.data.shape[:2]
        N     = idx.shape[0]

        # ── 1. Remove exact duplicate vertices (keeps first occurrence) ────────────
        uniq_idx, uniq_pos, inverse = np.unique(
            idx, axis=0, return_index=True, return_inverse=True
        )

        x_u, y_u, z_u = uniq_idx.T
        # ── 2. Can we use the fast path? ───────────────────────────────────────────
        duplicates_in_any_dim = (
            len(x_u) != len(np.unique(x_u))
            or len(y_u) != len(np.unique(y_u))
            or len(z_u) != len(np.unique(z_u))
        )

        if not duplicates_in_any_dim:
            # ---- FAST PATH -------------------------------------------------------
            #   Sort so every array is strictly increasing.
            sort = np.lexsort((z_u, y_u, x_u))        # z fastest (C-order)
            vals_sorted = self.data[:, :, x_u[sort], y_u[sort], z_u[sort]]
            vals_unique = vals_sorted[..., np.argsort(sort)]      # undo sort
            return vals_unique[..., inverse]                      # re-expand dupes

        # ── 3. Fallback: loop over unique vertices ────────────────────────────────
        out = np.empty((S, D, len(uniq_idx)), dtype=self.data.dtype)
        for k, (x, y, z) in enumerate(uniq_idx):
            out[:, :, k] = self.data[:, :, x, y, z]    # broadcast over S & D

        return out[..., inverse]

    def _assert_dimensions_are_correct(self, indices: np.ndarray) -> None:
        """
        Asserts that the given indices are within bounds for the Jacobian's spatial dimensions.
        
        Args:
            indices: A numpy array of shape (N, 3) containing integer indices for (x, y, z) coordinates.
            
        Raises:
            AssertionError: If any indices are out of bounds.
        """
        # Jacobian shape is (S, D, X, Y, Z), so we check against dimensions 2,3,4
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

