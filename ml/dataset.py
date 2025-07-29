from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MLDataset:
    """Simple container for ML data - just X and y arrays."""
    X: np.ndarray
    y: np.ndarray
    metadata: dict = None
    
    def __post_init__(self):
        """Validate that X and y have compatible shapes."""
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y must have same length. Got X: {len(self.X)}, y: {len(self.y)}")
        
        # Initialize empty metadata if None
        if self.metadata is None:
            self.metadata = {}


class SessionDatasetCollection:
    """Collection of MLDataset objects, one per session."""
    
    def __init__(self, datasets: List[MLDataset]):
        """
        Args:
            datasets: List of MLDataset objects, each with their own metadata
        """
        if not datasets:
            raise ValueError("SessionDatasetCollection cannot be empty")
        
        self.datasets = datasets
        self._validate_consistent_metadata()
    
    def _validate_consistent_metadata(self):
        """Assert that all datasets have the same metadata."""
        if len(self.datasets) <= 1:
            return
        
        first_metadata = self.datasets[0].metadata
        for i, dataset in enumerate(self.datasets[1:], start=1):
            if dataset.metadata != first_metadata:
                raise ValueError(
                    f"All datasets must have the same metadata. "
                    f"Dataset 0 has metadata: {first_metadata}, "
                    f"but dataset {i} has metadata: {dataset.metadata}"
                )
    
    @property
    def metadata(self) -> dict:
        """Get the shared metadata from the collection."""
        return self.datasets[0].metadata if self.datasets else {}

