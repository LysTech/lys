from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import numpy as np

from lys.ml.dataset import MLDataset


@dataclass
class TrainValTestSplit:
    """A container for train, validation, and test datasets."""
    train: MLDataset
    val: MLDataset
    test: MLDataset


class DatasetSplitter(ABC):
    """
    Abstract base class for dataset splitting strategies.
    """
    @abstractmethod
    def split(self, datasets: List[MLDataset]) -> TrainValTestSplit:
        """
        Splits a list of datasets into training, validation and testing sets.

        Args:
            datasets (List[MLDataset]): The datasets to split.

        Returns:
            TrainValTestSplit: An object containing the
            training, validation, and testing datasets.
        """
        pass


class TemporalSplitter(DatasetSplitter):
    """
    Splits time series data chronologically to respect causality.
    
    Each session is split temporally: first portion goes to training,
    middle portion to validation, last portion to test. This ensures
    that training data always comes before validation data, which
    comes before test data, preserving temporal order.
    """
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Initialize the temporal splitter.
        
        Args:
            train_ratio: Proportion of data for training (default 0.6)
            val_ratio: Proportion of data for validation (default 0.2)  
            test_ratio: Proportion of data for testing (default 0.2)
            
        Raises:
            AssertionError: If ratios don't sum to 1.0
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, (
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, datasets: List[MLDataset]) -> TrainValTestSplit:
        """
        Splits datasets temporally within each session.
        
        For each session, takes the first train_ratio proportion for training,
        the next val_ratio proportion for validation, and the final test_ratio
        proportion for testing. All sessions contribute to all three splits.
        
        Args:
            datasets: List of MLDataset objects, one per session
            
        Returns:
            TrainValTestSplit: Combined train, validation, and test datasets
        """
        train_X_list, train_y_list = [], []
        val_X_list, val_y_list = [], []
        test_X_list, test_y_list = [], []
        
        for dataset in datasets:
            train_data, val_data, test_data = self._split_single_session(dataset)
            
            train_X_list.append(train_data.X)
            train_y_list.append(train_data.y)
            
            val_X_list.append(val_data.X)
            val_y_list.append(val_data.y)
            
            test_X_list.append(test_data.X)
            test_y_list.append(test_data.y)
        
        # Concatenate all sessions for each split
        train_dataset = MLDataset(
            X=np.concatenate(train_X_list, axis=0),
            y=np.concatenate(train_y_list, axis=0),
            metadata=datasets[0].metadata  # All sessions should have same metadata
        )
        
        val_dataset = MLDataset(
            X=np.concatenate(val_X_list, axis=0),
            y=np.concatenate(val_y_list, axis=0),
            metadata=datasets[0].metadata
        )
        
        test_dataset = MLDataset(
            X=np.concatenate(test_X_list, axis=0),
            y=np.concatenate(test_y_list, axis=0),
            metadata=datasets[0].metadata
        )
        
        return TrainValTestSplit(train=train_dataset, val=val_dataset, test=test_dataset)
    
    def _split_single_session(self, dataset: MLDataset) -> tuple[MLDataset, MLDataset, MLDataset]:
        """
        Split a single session's data temporally.
        
        Args:
            dataset: MLDataset for a single session
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        n_samples = len(dataset.X)
        
        # Calculate split indices
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))
        
        # Ensure we don't lose samples due to rounding
        if val_end >= n_samples:
            val_end = n_samples - 1
        
        # Split the data temporally
        train_dataset = MLDataset(
            X=dataset.X[:train_end],
            y=dataset.y[:train_end],
            metadata=dataset.metadata
        )
        
        val_dataset = MLDataset(
            X=dataset.X[train_end:val_end],
            y=dataset.y[train_end:val_end],
            metadata=dataset.metadata
        )
        
        test_dataset = MLDataset(
            X=dataset.X[val_end:],
            y=dataset.y[val_end:],
            metadata=dataset.metadata
        )
        
        return train_dataset, val_dataset, test_dataset
