from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

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
