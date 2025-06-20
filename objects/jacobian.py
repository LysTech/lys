from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interpn


@dataclass
class Jacobian:
    """Represents a Jacobian matrix, which is a 3D numpy array."""

    data: np.ndarray

    def __post_init__(self):
        """Validates the Jacobian data after initialization."""
        if self.data.ndim != 3:
            raise ValueError("Jacobian data must be a 3D numpy array.")

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
