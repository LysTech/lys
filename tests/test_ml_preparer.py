import pytest
import numpy as np
from unittest.mock import Mock
from lys.ml.preparer import MLDataPreparer
from lys.objects.session import Session

class TestMLDataPreparer:
    """Test cases for the MLDataPreparer class."""

    def setup_method(self):
        """Set up a mock session for each test."""
        self.mock_session = Mock(spec=Session)
        self.mock_session.patient = Mock()
        self.mock_session.patient.name = "test_patient"
        self.mock_session.processed_data = {}

    def test_prepare_with_allowed_keys(self):
        """Test that prepare() correctly stacks data from allowed keys."""
        self.mock_session.processed_data = {
            "HbO": np.array([[1, 2], [3, 4]]),
            "HbR": np.array([[5, 6], [7, 8]]),
            "other_key": "some_value"
        }
        
        preparer = MLDataPreparer()
        preparer.prepare(self.mock_session)
        
        assert "data_for_ml" in self.mock_session.processed_data
        
        keys_to_stack = ["HbO", "HbR"]
        expected_data = np.stack([
            self.mock_session.processed_data[key] for key in keys_to_stack
        ], axis=-1)
        
        np.testing.assert_array_equal(self.mock_session.processed_data["data_for_ml"], expected_data)

    def test_prepare_with_no_allowed_keys(self):
        """Test that prepare() creates an empty array when no allowed keys are present."""
        self.mock_session.processed_data = {"other_key": "some_value"}
        
        preparer = MLDataPreparer()
        preparer.prepare(self.mock_session)
        
        assert "data_for_ml" in self.mock_session.processed_data
        assert len(self.mock_session.processed_data['data_for_ml']) == 0
        

    def test_prepare_with_mismatched_shapes(self, capsys):
        """Test that prepare() handles mismatched shapes and issues a warning."""
        self.mock_session.processed_data = {
            "HbO": np.array([[1, 2], [3, 4]]),
            "HbR": np.array([5, 6, 7, 8])
        }
        
        preparer = MLDataPreparer()
        preparer.prepare(self.mock_session)
        
        captured = capsys.readouterr()
        assert "Warning: Could not stack data for ML" in captured.out
        assert "data_for_ml" in self.mock_session.processed_data
        assert self.mock_session.processed_data["data_for_ml"].size == 0

    def test_prepare_warns_about_extra_keys(self, capsys):
        """Test that prepare() issues a warning for keys not included in the ML data."""
        self.mock_session.processed_data = {
            "HbO": np.array([[1, 2]]),
            "extra_key_1": 123,
            "extra_key_2": "abc"
        }
        
        preparer = MLDataPreparer()
        preparer.prepare(self.mock_session)
        
        captured = capsys.readouterr()
        
        assert "Warning: The following keys from session.processed_data were not included" in captured.out
        assert "extra_key_1" in captured.out
        assert "extra_key_2" in captured.out
        assert "HbO" not in captured.out

    def test_prepare_with_custom_allowed_keys(self):
        """Test that prepare() works correctly with custom allowed keys."""
        self.mock_session.processed_data = {
            "custom1": np.array([1, 2]),
            "custom2": np.array([3, 4]),
            "HbO": np.array([5, 6])
        }
        
        preparer = MLDataPreparer(allowed_keys={"custom1", "custom2"})
        preparer.prepare(self.mock_session)
        
        assert "data_for_ml" in self.mock_session.processed_data
        
        keys_to_stack = ["custom1", "custom2"]
        expected_data = np.stack([
            self.mock_session.processed_data[key] for key in keys_to_stack
        ], axis=-1)
        
        np.testing.assert_array_equal(self.mock_session.processed_data["data_for_ml"], expected_data) 