import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import numpy as np
from typing import Dict, List, Sequence

from lys.objects.session import (
    Session, 
    create_session, 
    create_session_with_common_channels,
    _load_npz_or_error,
    _load_numpy_formatted_data,
    _make_session
)
from lys.utils.paths import get_session_paths
from lys.objects import Patient, Protocol, Jacobian


@pytest.fixture
def mock_patient():
    """Create a mock Patient object."""
    return Mock(spec=Patient)


@pytest.fixture
def mock_protocol():
    """Create a mock Protocol object."""
    return Mock(spec=Protocol)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data for testing."""
    return {'wl1': np.array([1, 2, 3])}


@pytest.fixture
def sample_processed_data():
    """Create sample processed data for testing."""
    return {'wl1': np.array([10, 20, 30])}


@pytest.fixture
def sample_physio_data():
    """Create sample physiological data for testing."""
    return {'heart_rate': np.array([60, 65, 70])}


@pytest.fixture
def mock_jacobians():
    """Create mock Jacobian objects."""
    return [Mock(spec=Jacobian), Mock(spec=Jacobian)]


@pytest.fixture
def session_path():
    """Create a sample session path for testing."""
    return Path('/test/P01/nirs/experiment_name/session_name')


@pytest.fixture
def create_session_helper(mock_patient, mock_protocol, sample_raw_data):
    """Helper fixture to create Session objects with common defaults."""
    def _create_session(**kwargs):
        """Create a Session with default parameters that can be overridden."""
        defaults = {
            'patient': mock_patient,
            'protocol': mock_protocol,
            'raw_data': sample_raw_data
        }
        defaults.update(kwargs)
        return Session(**defaults)
    return _create_session


def assert_session_attributes(session, expected_attrs):
    """Assert that session has the expected attributes."""
    for attr_name, expected_value in expected_attrs.items():
        assert getattr(session, attr_name) is expected_value


def verify_mock_calls(mocks, expected_calls):
    """Verify that mocks were called with expected parameters."""
    for mock_name, call_info in expected_calls.items():
        mock = mocks[mock_name]
        if 'args' in call_info:
            mock.assert_called_once_with(*call_info['args'])
        else:
            mock.assert_called_once()


class TestSessionInitialization:
    """Tests for Session object initialization and default values."""

    def test_session_metadata_defaults_to_empty_processing_steps(self, create_session_helper):
        """Test that metadata['processing_steps'] is properly initialized to []."""
        session = create_session_helper()
        
        assert session.metadata['processing_steps'] == []

    def test_session_metadata_can_be_overridden(self, create_session_helper):
        """Test that metadata can be overridden with custom values."""
        custom_metadata = {'processing_steps': ['step1'], 'custom_key': 'value'}
        
        session = create_session_helper(metadata=custom_metadata)
        
        assert session.metadata['processing_steps'] == ['step1']
        assert session.metadata['custom_key'] == 'value'

    def test_processed_data_defaults_to_raw_data(self, mock_patient, mock_protocol):
        """Test that processed_data defaults to a copy of raw_data when not provided."""
        raw_data = {'wl1': np.array([1, 2, 3]), 'wl2': np.array([4, 5, 6])}
        
        session = Session(patient=mock_patient, protocol=mock_protocol, raw_data=raw_data)
        
        # processed_data should not be the same object
        assert session.processed_data is not session.raw_data
        # but the values should be the same
        assert session.processed_data is not None
        assert session.processed_data.keys() == session.raw_data.keys()
        for key in session.processed_data:
            np.testing.assert_array_equal(session.processed_data[key], session.raw_data[key])

    def test_processed_data_can_be_overridden(self, create_session_helper, sample_processed_data):
        """Test that processed_data can be overridden with custom values."""
        session = create_session_helper(processed_data=sample_processed_data)
        
        assert session.processed_data is sample_processed_data
        assert session.processed_data is not session.raw_data

    def test_jacobians_defaults_to_none(self, create_session_helper):
        """Test that jacobians defaults to None."""
        session = create_session_helper()
        
        assert session.jacobians is None

    def test_jacobians_can_be_provided(self, create_session_helper, mock_jacobians):
        """Test that jacobians can be provided."""
        session = create_session_helper(jacobians=mock_jacobians)
        
        assert session.jacobians == mock_jacobians

    def test_physio_data_defaults_to_none(self, create_session_helper):
        """Test that physio_data defaults to None."""
        session = create_session_helper()
        
        assert session.physio_data is None

    def test_physio_data_can_be_provided(self, create_session_helper, sample_physio_data):
        """Test that physio_data can be provided."""
        session = create_session_helper(physio_data=sample_physio_data)
        
        assert session.physio_data == sample_physio_data

    def test_session_attributes_are_set_correctly(self, create_session_helper, mock_patient, mock_protocol, sample_raw_data):
        """Test that all session attributes are set correctly."""
        session = create_session_helper()
        
        expected_attrs = {
            'patient': mock_patient,
            'protocol': mock_protocol,
            'raw_data': sample_raw_data
        }
        assert_session_attributes(session, expected_attrs)


class TestCreateSession:
    """Tests for the create_session function."""

    @pytest.fixture
    def mock_session_dependencies(self, mock_patient, mock_protocol, mock_jacobians, sample_raw_data, sample_processed_data, sample_physio_data):
        """Setup all mocks needed for create_session tests."""
        with patch('lys.objects.session.extract_patient_from_path') as mock_extract_patient, \
             patch('lys.objects.session.Patient.from_name') as mock_patient_from_name, \
             patch('lys.objects.session.Protocol.from_session_path') as mock_protocol_from_path, \
             patch('lys.objects.session.load_jacobians_from_session_dir') as mock_load_jacobians, \
             patch('lys.objects.session._load_npz_or_error') as mock_load_npz:
            
            mock_extract_patient.return_value = 'P01'
            mock_patient_from_name.return_value = mock_patient
            mock_protocol_from_path.return_value = mock_protocol
            mock_load_jacobians.return_value = mock_jacobians
            
            yield {
                'extract_patient': mock_extract_patient,
                'patient_from_name': mock_patient_from_name,
                'protocol_from_path': mock_protocol_from_path,
                'load_jacobians': mock_load_jacobians,
                'load_npz': mock_load_npz,
                'patient': mock_patient,
                'protocol': mock_protocol,
                'jacobians': mock_jacobians,
                'raw_data': sample_raw_data,
                'processed_data': sample_processed_data,
                'physio_data': sample_physio_data
            }

    def test_create_session_uses_standard_channels_file(self, mock_session_dependencies, session_path):
        """Test that create_session loads the standard raw_channel_data.npz file."""
        mocks = mock_session_dependencies
        mocks['load_npz'].side_effect = [mocks['raw_data'], mocks['processed_data'], mocks['physio_data']]
        
        session = create_session(session_path)
        
        # Verify the correct raw data file was requested
        calls = mocks['load_npz'].call_args_list
        assert calls[0][0] == (session_path, 'raw_channel_data.npz')  # First call for raw data
        assert calls[0][1]['required'] == True
        
        # Verify session attributes
        assert session.patient == mocks['patient']
        assert session.raw_data == mocks['raw_data']

    def test_create_session_with_common_channels_uses_common_channels_file(self, mock_session_dependencies, session_path):
        """Test that create_session_with_common_channels loads the common channels file."""
        mocks = mock_session_dependencies
        mocks['load_npz'].side_effect = [mocks['raw_data'], mocks['processed_data'], mocks['physio_data']]
        
        session = create_session_with_common_channels(session_path)
        
        # Verify the correct raw data file was requested
        calls = mocks['load_npz'].call_args_list
        assert calls[0][0] == (session_path, 'raw_channel_data_common_channels.npz')  # First call for raw data
        assert calls[0][1]['required'] == True
        
        # Verify session attributes
        assert session.patient == mocks['patient']
        assert session.raw_data == mocks['raw_data']


class TestLoadNpzOrError:
    """Tests for the _load_npz_or_error helper function."""

    @patch('numpy.load')
    @patch('pathlib.Path.exists')
    def test_load_npz_or_error_file_exists(self, mock_exists, mock_np_load):
        """Test _load_npz_or_error when file exists."""
        mock_exists.return_value = True
        mock_data = {'data': np.array([1, 2, 3])}
        mock_np_load.return_value = mock_data
        
        path = Path('/test/path')
        result = _load_npz_or_error(path, 'test.npz')
        
        assert result == mock_data
        mock_np_load.assert_called_once_with(path / 'test.npz', allow_pickle=True)

    @patch('pathlib.Path.exists')
    def test_load_npz_or_error_file_missing_required(self, mock_exists):
        """Test _load_npz_or_error raises when required file is missing."""
        mock_exists.return_value = False
        
        path = Path('/test/path')
        
        with pytest.raises(FileNotFoundError, match="Required file 'test.npz' not found"):
            _load_npz_or_error(path, 'test.npz', required=True)

    @patch('pathlib.Path.exists')
    def test_load_npz_or_error_file_missing_optional(self, mock_exists):
        """Test _load_npz_or_error returns None when optional file is missing (only Error when required)."""
        mock_exists.return_value = False
        
        path = Path('/test/path')
        result = _load_npz_or_error(path, 'test.npz', required=False)
        
        assert result is None


def test_get_session_paths_fnirs_8classes():
    """Test that 8classes experiment returns all 8 sessions."""
    session_paths = get_session_paths('8classes', 'nirs')
    assert len(session_paths) == 8, f"Expected 8 sessions, got {len(session_paths)}"


class TestSessionPostInit:
    """Tests for Session's __post_init__ method."""

    def test_post_init_sets_processed_data_to_copy_of_raw_data_when_none(self, create_session_helper):
        """Test that __post_init__ sets processed_data to a copy of raw_data when processed_data is None in __init__."""
        # Create session with processed_data=None (default)
        session = create_session_helper()
        
        # Verify processed_data was set to a copy of raw_data in __post_init__
        assert session.processed_data is not session.raw_data
        # but the values should be the same
        assert session.processed_data is not None
        assert session.processed_data.keys() == session.raw_data.keys()
        for key in session.processed_data:
            np.testing.assert_array_equal(session.processed_data[key], session.raw_data[key])

    def test_post_init_does_not_override_existing_processed_data(self, create_session_helper, sample_processed_data):
        """Test that __post_init__ doesn't override existing processed_data."""
        # Create session with explicit processed_data
        session = create_session_helper(processed_data=sample_processed_data)
        
        # Verify processed_data was not overridden
        assert session.processed_data is sample_processed_data
        assert session.processed_data is not session.raw_data 


class TestHelperFunctions:
    """Tests for the refactored helper functions."""
    
    def test_load_numpy_formatted_data_loads_all_files(self):
        """Test that _load_numpy_formatted_data loads all three npz files."""
        with patch('lys.objects.session._load_npz_or_error') as mock_load:
            mock_raw = {'wl1': np.array([1, 2, 3])}
            mock_processed = {'wl1': np.array([10, 20, 30])}
            mock_physio = {'heart_rate': np.array([60, 65, 70])}
            mock_load.side_effect = [mock_raw, mock_processed, mock_physio]
            
            path = Path('/test/path')
            raw, processed, physio = _load_numpy_formatted_data(path, 'test_raw.npz')
            
            # Verify all files were requested in the correct order
            calls = mock_load.call_args_list
            assert len(calls) == 3
            assert calls[0] == ((path, 'test_raw.npz'), {'required': True})
            assert calls[1] == ((path, 'processed_channel_data.npz'), {'required': False})
            assert calls[2] == ((path, 'physio_data.npz'), {'required': False})
            
            # Verify return values
            assert raw == mock_raw
            assert processed == mock_processed
            assert physio == mock_physio
    
    def test_make_session_creates_session_correctly(self, mock_patient, mock_protocol, mock_jacobians):
        """Test that _make_session correctly assembles a Session object."""
        raw_data = {'wl1': np.array([1, 2, 3])}
        processed_data = {'wl1': np.array([10, 20, 30]), 'metadata': {'processing_steps': ['step1']}}
        physio_data = {'heart_rate': np.array([60, 65, 70])}
        
        session = _make_session(
            patient=mock_patient,
            protocol=mock_protocol,
            jacobians=mock_jacobians,
            raw_npz=raw_data,
            processed_npz=processed_data,
            physio_npz=physio_data
        )
        
        # Verify all attributes are set correctly
        assert session.patient == mock_patient
        assert session.protocol == mock_protocol
        assert session.jacobians == mock_jacobians
        assert session.raw_data == raw_data
        assert session.processed_data == processed_data
        assert session.physio_data == physio_data
        assert session.metadata == {'processing_steps': ['step1']}  # From processed_data metadata 