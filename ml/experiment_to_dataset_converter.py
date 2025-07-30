from typing import List
import numpy as np

from lys.objects.experiment import Experiment
from lys.objects.session import Session
from lys.ml.dataset import MLDataset
from lys.ml.splitting_strategies import DatasetSplitter, TrainValTestSplit


class ExperimentToDatasetConverter:
    """
    Converts an Experiment into a set of ML datasets for training, validation, and testing.
    """

    def __init__(self, splitter: DatasetSplitter):
        """
        Initializes the converter with a dataset splitting strategy.

        Args:
            splitter (DatasetSplitter): The strategy for splitting datasets.
        """
        self.splitter = splitter

    def convert(self, experiment: Experiment) -> TrainValTestSplit:
        """
        Converts an experiment into training, validation, and testing datasets.

        Args:
            experiment (Experiment): The experiment to convert.

        Returns:
            TrainValTestSplit: The split datasets.
        """
        session_datasets = self._create_session_datasets(experiment.sessions)
        self._validate_consistent_metadata(session_datasets)
        return self.splitter.split(session_datasets)

    def _validate_consistent_metadata(self, datasets: List[MLDataset]) -> None:
        """
        Validates that all datasets have consistent metadata.
        
        Args:
            datasets: List of MLDataset objects to validate.
            
        Raises:
            AssertionError: If datasets have inconsistent metadata.
        """
        if len(datasets) <= 1:
            return
            
        first_metadata = datasets[0].metadata
        for i, dataset in enumerate(datasets[1:], start=1):
            assert dataset.metadata == first_metadata, (
                f"All datasets must have consistent metadata. "
                f"Dataset 0 has metadata: {first_metadata}, "
                f"but dataset {i} has metadata: {dataset.metadata}"
            )

    def _create_session_datasets(self, sessions: List[Session]) -> List[MLDataset]:
        """
        Converts a list of Sessions into a list of MLDataloader.
        """
        return [self._session_to_mldataset(session) for session in sessions]

    def _session_to_mldataset(self, session: Session) -> MLDataset:
        """
        Converts a single Session to an MLDataset.

        This method extracts the features (X) from the session's processed_data
        and the target labels (y) from the session's protocol based on timing.
        For each time point, it finds the most recent protocol interval that 
        started before that time point to respect causality.

        Args:
            session (Session): The session to convert.

        Returns:
            MLDataset: The resulting dataset.

        Raises:
            AssertionError: If required data is not found or has wrong format.
        """
        data_for_ml = session.processed_data.get("data_for_ml")
        assert data_for_ml is not None, "Session's processed_data is missing 'data_for_ml' key."
        assert isinstance(data_for_ml, np.ndarray), "'data_for_ml' must be a numpy array."

        time_vector = session.raw_data.get("time")
        assert time_vector is not None, "Session's raw_data is missing 'time' key."
        assert isinstance(time_vector, np.ndarray), "'time' must be a numpy array."
        assert len(time_vector) == len(data_for_ml), "Time vector and data_for_ml must have the same length."

        # X is the entire data_for_ml array
        x = data_for_ml

        # y comes from protocol based on timing
        y = self._extract_labels_from_protocol(time_vector, session.protocol)

        return MLDataset(X=x, y=y, metadata=session.metadata)

    def _extract_labels_from_protocol(self, time_vector: np.ndarray, protocol) -> np.ndarray:
        """
        Extract target labels for each time point based on protocol intervals.
        
        For each time point, finds the most recent protocol interval that started
        before that time point to respect causality (the "last thing that was done").
        
        Args:
            time_vector (np.ndarray): Array of time points.
            protocol: Protocol object with intervals attribute.
            
        Returns:
            np.ndarray: Array of labels for each time point.
        """
        labels = []
        
        for time_point in time_vector:
            # Find all intervals that started before or at this time point
            valid_intervals = [
                (t_start, t_end, label) 
                for t_start, t_end, label in protocol.intervals
                if t_start <= time_point
            ]
            
            if valid_intervals:
                # Among valid intervals, pick the one with the largest start time
                # (the most recent one that started)
                most_recent_interval = max(valid_intervals, key=lambda interval: interval[0])
                label = most_recent_interval[2]  # The third element is the label
            else:
                # If no interval has started yet, use a default label
                label = "<BASELINE>"  # Time before any experimental task begins
                
            labels.append(label)
        
        return np.array(labels)


if __name__ == "__main__":
    from lys.objects.experiment import create_experiment
    from lys.ml.splitting_strategies import TemporalSplitter
    from lys.processing.pipeline import ProcessingPipeline

    experiment = create_experiment("perceived_speech", "flow2")
    
    # Apply processing pipeline (even with no steps, this ensures MLDataPreparer runs)
    pipeline = ProcessingPipeline([])
    experiment = pipeline.apply(experiment)
    
    splitter = TemporalSplitter()
    converter = ExperimentToDatasetConverter(splitter)
    datasets = converter.convert(experiment) #the "extra dimension" comes from stacking: here we just have 'data', but if
    #we have 'wl1', 'wl2', 'HbO', 'HbR', we stack them all together and have a few extra dimensions.