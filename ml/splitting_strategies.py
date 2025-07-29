from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from lys.ml.dataset import SessionDatasetCollection


class DatasetSplitter(ABC):
    """Abstract base class for different strategies to split datasets for training/testing."""
    
    @abstractmethod
    def split(self, dataset_collection: SessionDatasetCollection) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split a collection of session datasets into train/test sets.
        
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        pass