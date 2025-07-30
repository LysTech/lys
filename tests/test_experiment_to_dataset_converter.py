import pytest
from unittest.mock import Mock, MagicMock
import numpy as np
from typing import List

from lys.ml.experiment_to_dataset_converter import ExperimentToDatasetConverter
from lys.ml.dataset import MLDataset
from lys.ml.splitting_strategies import DatasetSplitter, TrainValTestSplit
from lys.objects.experiment import Experiment
from lys.objects.session import Session
from lys.objects.protocol import Protocol
from lys.objects.patient import Patient


class TestExperimentToDatasetConverter:
    """Test suite for ExperimentToDatasetConverter."""

    @pytest.fixture
    def mock_splitter(self) -> Mock:
        """Create a mock dataset splitter."""
        splitter = Mock(spec=DatasetSplitter)
        mock_datasets = TrainValTestSplit(
            train=Mock(spec=MLDataset),
            val=Mock(spec=MLDataset),
            test=Mock(spec=MLDataset)
        )
        splitter.split.return_value = mock_datasets
        return splitter

    @pytest.fixture
    def converter(self, mock_splitter: Mock) -> ExperimentToDatasetConverter:
        """Create a converter with mocked splitter."""
        return ExperimentToDatasetConverter(mock_splitter)

    @pytest.fixture
    def mock_patient(self) -> Mock:
        """Create a mock patient."""
        patient = Mock(spec=Patient)
        patient.name = "P01"
        return patient

    @pytest.fixture
    def mock_protocol(self) -> Mock:
        """Create a mock protocol with realistic intervals."""
        protocol = Mock(spec=Protocol)
        # Intervals: (start_time, end_time, label)
        protocol.intervals = [
            (0.0, 10.0, "REST"),
            (10.0, 20.0, "TASK_A"),
            (20.0, 30.0, "TASK_B"),
            (30.0, 40.0, "REST")
        ]
        return protocol

    @pytest.fixture
    def mock_session(self, mock_patient: Mock, mock_protocol: Mock) -> Mock:
        """Create a mock session with realistic data."""
        session = Mock(spec=Session)
        session.patient = mock_patient
        session.protocol = mock_protocol
        
        # Create realistic test data
        time_points = 100
        session.raw_data = {
            "time": np.linspace(0, 40, time_points)  # 40 seconds, 100 time points
        }
        session.processed_data = {
            "data_for_ml": np.random.randn(time_points, 10)  # 10 features
        }
        
        # Add consistent dummy processing steps metadata
        session.metadata = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "BandpassFilter"
            ]
        }
        
        return session

    @pytest.fixture
    def mock_experiment(self, mock_session: Mock) -> Mock:
        """Create a mock experiment with multiple sessions."""
        experiment = Mock(spec=Experiment)
        
        # Create two sessions with different patient names but same structure and metadata
        session1 = mock_session
        session2 = Mock(spec=Session)
        session2.patient = Mock(spec=Patient)
        session2.patient.name = "P02"
        session2.protocol = mock_session.protocol
        session2.raw_data = mock_session.raw_data.copy()
        session2.processed_data = mock_session.processed_data.copy()
        
        # Ensure both sessions have the same processing steps metadata
        session2.metadata = mock_session.metadata.copy()
        
        experiment.sessions = [session1, session2]
        return experiment

    def test_convert_calls_all_expected_methods(self, converter: ExperimentToDatasetConverter, 
                                               mock_experiment: Mock) -> None:
        """Test that convert method orchestrates the conversion process correctly."""
        result = converter.convert(mock_experiment)
        
        # Verify splitter was called
        converter.splitter.split.assert_called_once()
        
        # Verify result is the expected type
        assert isinstance(result, TrainValTestSplit)

    def test_session_to_mldataset_extracts_correct_data(self, converter: ExperimentToDatasetConverter,
                                                       mock_session: Mock) -> None:
        """Test that session conversion extracts features and labels correctly."""
        result = converter._session_to_mldataset(mock_session)
        
        # Verify the dataset structure
        assert isinstance(result, MLDataset)
        expected_metadata = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "BandpassFilter"
            ]
        }
        assert result.metadata == expected_metadata
        
        # Verify data shapes match
        expected_X = mock_session.processed_data["data_for_ml"]
        expected_time_length = len(mock_session.raw_data["time"])
        
        assert np.array_equal(result.X, expected_X)
        assert len(result.y) == expected_time_length
        assert len(result.X) == len(result.y)

    def test_extract_labels_from_protocol_assigns_correct_labels(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that labels are correctly assigned based on protocol timing."""
        time_vector = np.array([0.0, 5.0, 15.0, 25.0, 35.0])
        
        protocol = Mock()
        protocol.intervals = [
            (0.0, 10.0, "REST"),
            (10.0, 20.0, "TASK_A"), 
            (20.0, 30.0, "TASK_B"),
            (30.0, 40.0, "REST")
        ]
        
        labels = converter._extract_labels_from_protocol(time_vector, protocol)
        
        expected_labels = ["REST", "REST", "TASK_A", "TASK_B", "REST"]
        assert np.array_equal(labels, expected_labels)

    def test_extract_labels_handles_baseline_before_first_interval(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that times before any interval get baseline label."""
        time_vector = np.array([0.0, 5.0, 8.0, 15.0])
        
        protocol = Mock()
        protocol.intervals = [(10.0, 20.0, "TASK_A")]
        
        labels = converter._extract_labels_from_protocol(time_vector, protocol)
        
        expected_labels = ["<BASELINE>", "<BASELINE>", "<BASELINE>", "TASK_A"]
        assert np.array_equal(labels, expected_labels)

    def test_validate_consistent_metadata_passes_with_identical_metadata(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that validation passes when all datasets have identical metadata."""
        consistent_metadata = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "BandpassFilter"
            ]
        }
        datasets = [
            MLDataset(X=np.array([[1]]), y=np.array([1]), metadata=consistent_metadata),
            MLDataset(X=np.array([[2]]), y=np.array([2]), metadata=consistent_metadata)
        ]
        
        # Should not raise an exception
        converter._validate_consistent_metadata(datasets)

    def test_validate_consistent_metadata_handles_empty_list(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that validation handles empty dataset list gracefully."""
        converter._validate_consistent_metadata([])

    def test_validate_consistent_metadata_handles_single_dataset(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that validation handles single dataset gracefully."""
        metadata = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "BandpassFilter"
            ]
        }
        dataset = MLDataset(X=np.array([[1]]), y=np.array([1]), metadata=metadata)
        converter._validate_consistent_metadata([dataset])

    def test_validate_consistent_metadata_raises_on_inconsistent_metadata(self, converter: ExperimentToDatasetConverter) -> None:
        """Test that validation raises assertion error for inconsistent metadata."""
        metadata1 = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "BandpassFilter"
            ]
        }
        metadata2 = {
            "processing_steps": [
                "ConvertWavelengthsToOD",
                "ConvertODtoHbOandHbR", 
                "RemoveScalpEffect"  # Different processing step
            ]
        }
        datasets = [
            MLDataset(X=np.array([[1]]), y=np.array([1]), metadata=metadata1),
            MLDataset(X=np.array([[2]]), y=np.array([2]), metadata=metadata2)
        ]
        
        with pytest.raises(AssertionError) as exc_info:
            converter._validate_consistent_metadata(datasets)
        
        assert_error_message_contains_dataset_info(str(exc_info.value))

    def test_session_to_mldataset_raises_when_data_for_ml_missing(self, converter: ExperimentToDatasetConverter,
                                                                 mock_session: Mock) -> None:
        """Test that conversion raises assertion error when data_for_ml is missing."""
        mock_session.processed_data = {}  # Missing 'data_for_ml' key
        
        with pytest.raises(AssertionError) as exc_info:
            converter._session_to_mldataset(mock_session)
        
        assert "data_for_ml" in str(exc_info.value)

    def test_session_to_mldataset_raises_when_data_for_ml_wrong_type(self, converter: ExperimentToDatasetConverter,
                                                                    mock_session: Mock) -> None:
        """Test that conversion raises assertion error when data_for_ml is not numpy array."""
        mock_session.processed_data = {"data_for_ml": [[1, 2, 3]]}  # Not numpy array
        
        with pytest.raises(AssertionError) as exc_info:
            converter._session_to_mldataset(mock_session)
        
        assert "numpy array" in str(exc_info.value)

    def test_session_to_mldataset_raises_when_time_missing(self, converter: ExperimentToDatasetConverter,
                                                          mock_session: Mock) -> None:
        """Test that conversion raises assertion error when time vector is missing."""
        mock_session.raw_data = {}  # Missing 'time' key
        
        with pytest.raises(AssertionError) as exc_info:
            converter._session_to_mldataset(mock_session)
        
        assert "time" in str(exc_info.value)

    def test_session_to_mldataset_raises_when_time_wrong_type(self, converter: ExperimentToDatasetConverter,
                                                             mock_session: Mock) -> None:
        """Test that conversion raises assertion error when time is not numpy array."""
        mock_session.raw_data = {"time": [0, 1, 2, 3]}  # Not numpy array
        
        with pytest.raises(AssertionError) as exc_info:
            converter._session_to_mldataset(mock_session)
        
        assert "numpy array" in str(exc_info.value)

    def test_session_to_mldataset_raises_when_lengths_mismatch(self, converter: ExperimentToDatasetConverter,
                                                              mock_session: Mock) -> None:
        """Test that conversion raises assertion error when time and data lengths don't match."""
        mock_session.raw_data = {"time": np.array([0, 1, 2])}  # Length 3
        mock_session.processed_data = {"data_for_ml": np.array([[1], [2]])}  # Length 2
        
        with pytest.raises(AssertionError) as exc_info:
            converter._session_to_mldataset(mock_session)
        
        assert "same length" in str(exc_info.value)

    def test_create_session_datasets_processes_all_sessions(self, converter: ExperimentToDatasetConverter,
                                                           mock_experiment: Mock) -> None:
        """Test that all sessions are converted to datasets."""
        result = converter._create_session_datasets(mock_experiment.sessions)
        
        assert len(result) == len(mock_experiment.sessions)
        assert all(isinstance(dataset, MLDataset) for dataset in result)


def assert_error_message_contains_dataset_info(error_message: str) -> None:
    """Assert that error message contains information about dataset differences."""
    assert "metadata" in error_message
    assert "Dataset 0" in error_message
    assert "dataset 1" in error_message 