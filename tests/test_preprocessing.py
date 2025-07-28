import numpy as np
from pathlib import Path
from processing.preprocessing import BettinaSessionAdapter, RawSessionPreProcessor
import pytest
from unittest.mock import MagicMock, patch
from processing.preprocessing import Flow2MomentsSessionAdapter


def create_mock_snirf_with_stimulus(start_timestamp: float, n_timepoints: int = 5, 
                                   n_channels: int = 2, n_wavelengths: int = 2, 
                                   n_moments: int = 3, fs: float = 10.0) -> MagicMock:
    """
    Create a mock SNIRF object with proper stimulus events for testing.
    
    Args:
        start_timestamp: Absolute timestamp to put in stimulus event
        n_timepoints: Number of time points in the data
        n_channels: Number of channels (source-detector pairs)
        n_wavelengths: Number of wavelengths
        n_moments: Number of moments (amplitude, mean time, variance)
        fs: Sampling frequency in Hz
    
    Returns:
        Mock SNIRF object with stimulus events and data
    """
    # Build measurement list
    class FakeM:
        def __init__(self, src, det, wav, dtype_idx, dunit):
            self.sourceIndex = src
            self.detectorIndex = det
            self.wavelengthIndex = wav
            self.dataTypeIndex = dtype_idx
            self.dataUnit = dunit

    mlist = []
    for ch in range(1, n_channels + 1):
        for wav in range(1, n_wavelengths + 1):
            for moment_idx, (dtype, unit) in enumerate([(2, ''), (1, 'ps'), (3, 'ps^2')]):
                if moment_idx < n_moments:
                    mlist.append(FakeM(ch, ch, wav, dtype, unit))

    # Create fake data
    fake_data = np.arange(n_timepoints * len(mlist)).reshape(n_timepoints, len(mlist))
    relative_time = np.linspace(0, (n_timepoints - 1) / fs, n_timepoints)

    # Create data block
    fake_data_block = MagicMock()
    fake_data_block.measurementList = mlist
    fake_data_block.dataTimeSeries = fake_data
    fake_data_block.time = relative_time

    # Create stimulus event with "StartExperiment" 
    fake_stim = MagicMock()
    fake_stim.name = "StartExperiment"
    # SNIRF stimulus format: [onset, duration, amplitude, value]
    # Put absolute timestamp in value field (4th column, index 3)
    fake_stim.data = np.array([[0.0, 1.0, 1.0, start_timestamp]])

    # Create NIRS object
    fake_nirs = MagicMock()
    fake_nirs.data = [fake_data_block]
    fake_nirs.stim = [fake_stim]

    # Create SNIRF object
    fake_snirf = MagicMock()
    fake_snirf.nirs = [fake_nirs]

    return fake_snirf


def write_dummy_file(path: Path, data: np.ndarray):
    """Write a 2D numpy array to a text file, row by row."""
    np.savetxt(path, data)


def test_can_handle_true(tmp_path):
    """BettinaSessionAdapter.can_handle returns True if both .wl1 and .wl2 files exist."""
    (tmp_path / 'test.wl1').touch()
    (tmp_path / 'test.wl2').touch()
    adapter = BettinaSessionAdapter()
    assert adapter.can_handle(tmp_path)


def test_can_handle_false(tmp_path):
    """BettinaSessionAdapter.can_handle returns False if one or both files are missing."""
    (tmp_path / 'test.wl1').touch()
    adapter = BettinaSessionAdapter()
    assert not adapter.can_handle(tmp_path)
    (tmp_path / 'test.wl2').touch()
    (tmp_path / 'test.wl1').unlink()
    assert not adapter.can_handle(tmp_path)


def test_process_saves_npz(tmp_path):
    """BettinaSessionAdapter.process saves correct arrays to raw_channel_data.npz."""
    wl1_data = np.array([[1, 2, 3], [4, 5, 6]])
    wl2_data = np.array([[7, 8, 9], [10, 11, 12]])
    write_dummy_file(tmp_path / 'foo.wl1', wl1_data)
    write_dummy_file(tmp_path / 'bar.wl2', wl2_data)
    adapter = BettinaSessionAdapter()
    adapter.process(tmp_path)
    out = np.load(tmp_path / 'raw_channel_data.npz')
    np.testing.assert_array_equal(out['wl1'], wl1_data)
    np.testing.assert_array_equal(out['wl2'], wl2_data)


def test_bettina_adapter_extract_data_returns_correct_dict(tmp_path):
    """BettinaSessionAdapter.extract_data returns correct dictionary with wl1 and wl2 arrays."""
    wl1_data = np.array([[1, 2, 3], [4, 5, 6]])
    wl2_data = np.array([[7, 8, 9], [10, 11, 12]])
    write_dummy_file(tmp_path / 'foo.wl1', wl1_data)
    write_dummy_file(tmp_path / 'bar.wl2', wl2_data)
    adapter = BettinaSessionAdapter()
    data = adapter.extract_data(tmp_path)
    assert 'wl1' in data
    assert 'wl2' in data
    np.testing.assert_array_equal(data['wl1'], wl1_data)
    np.testing.assert_array_equal(data['wl2'], wl2_data)


def test_raw_session_processor_selects_bettina_adapter(tmp_path):
    """RawSessionProcessor selects BettinaSessionAdapter when .wl1 and .wl2 files are present."""
    (tmp_path / 'test.wl1').touch()
    (tmp_path / 'test.wl2').touch()
    processor = RawSessionPreProcessor(tmp_path)
    assert isinstance(processor.session_adapter, BettinaSessionAdapter)


def test_raw_session_processor_processes_bettina_session(tmp_path):
    """RawSessionProcessor.process correctly processes a Bettina session."""
    wl1_data = np.array([[1, 2], [3, 4]])
    wl2_data = np.array([[5, 6], [7, 8]])
    write_dummy_file(tmp_path / 'data.wl1', wl1_data)
    write_dummy_file(tmp_path / 'data.wl2', wl2_data)
    
    RawSessionPreProcessor.preprocess(tmp_path)
    
    output_file = tmp_path / 'raw_channel_data.npz'
    assert output_file.exists()
    loaded_data = np.load(output_file)
    np.testing.assert_array_equal(loaded_data['wl1'], wl1_data)
    np.testing.assert_array_equal(loaded_data['wl2'], wl2_data)


def test_raw_session_processor_raises_error_for_unsupported_session(tmp_path):
    """RawSessionProcessor raises ValueError when no adapter can handle the session."""
    (tmp_path / 'random.txt').touch()
    
    with pytest.raises(ValueError, match="No suitable adapter found"):
        RawSessionPreProcessor.preprocess(tmp_path)


def test_raw_session_processor_raises_error_for_empty_session(tmp_path):
    """RawSessionProcessor raises ValueError for empty sessions."""
    with pytest.raises(ValueError, match="No suitable adapter found"):
        RawSessionPreProcessor.preprocess(tmp_path) 


def test_flow2_can_handle_true(tmp_path):
    """Flow2MomentsSessionAdapter.can_handle returns True if a *_MOMENTS.snirf file exists."""
    (tmp_path / 'foo_MOMENTS.snirf').touch()
    adapter = Flow2MomentsSessionAdapter()
    assert adapter.can_handle(tmp_path)

def test_flow2_can_handle_false(tmp_path):
    """Flow2MomentsSessionAdapter.can_handle returns False if no *_MOMENTS.snirf file exists."""
    (tmp_path / 'foo.snirf').touch()
    adapter = Flow2MomentsSessionAdapter()
    assert not adapter.can_handle(tmp_path)


def test_flow2_extract_data_shape_and_metadata(monkeypatch, tmp_path):
    """extract_data returns correct shape and metadata for a mocked SNIRF file."""
    import json
    from datetime import datetime, timezone

    # --- Setup for alignment logic ---
    # Create a dummy .jsonl file. We'll set the NIRS start time to match this
    # exactly, so no trimming should occur, keeping the test focused on reshaping.
    start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    protocol_start_timestamp = start_time.timestamp()
    log_file = tmp_path / "protocol.jsonl"
    with open(log_file, 'w') as f:
        json.dump({'timestamp': protocol_start_timestamp, 'event': 'start'}, f)

        # Create a fake snirf file
    fake_file = tmp_path / 'bar_MOMENTS.snirf'
    fake_file.touch()

    # Define test parameters
    n_timepoints = 5
    n_channels = 2
    n_wavelengths = 2
    n_moments = 3
    fs = 10.0

    # Calculate relative time (needed for assertions)
    relative_time = np.linspace(0, (n_timepoints - 1) / fs, n_timepoints)

    # Create mock SNIRF object using our helper
    fake_snirf = create_mock_snirf_with_stimulus(
        start_timestamp=protocol_start_timestamp,
        n_timepoints=n_timepoints,
        n_channels=n_channels,
        n_wavelengths=n_wavelengths,
        n_moments=n_moments,
        fs=fs
    )

    # Patch snirf.Snirf to return our fake object
    with patch('snirf.Snirf', return_value=fake_snirf):
        adapter = Flow2MomentsSessionAdapter()
        result = adapter.extract_data(tmp_path)
        
    # --- Assertions ---
    # The main purpose of this test: check the reshaping and metadata extraction
    arr = result['data']
    assert arr.shape == (n_timepoints, n_channels, n_wavelengths, n_moments)
    assert set(result['moment_names']) == {'amplitude', 'mean_time_of_flight', 'variance'}
    assert len(result['channels']) == n_channels
    assert len(result['wavelengths']) == n_wavelengths
    
    # The time vector should now be absolute, and since we aligned it perfectly,
    # it should be untrimmed and match the expected absolute timestamps.
    expected_absolute_time = relative_time + protocol_start_timestamp
    assert result['time'].shape[0] == n_timepoints
    np.testing.assert_allclose(result['time'], expected_absolute_time)


def test_flow2_adapter_aligns_with_protocol(monkeypatch, tmp_path):
    """
    Test that Flow2MomentsSessionAdapter correctly trims NIRS data to align
    with the protocol's start time.
    """
    import json
    from datetime import datetime, timezone

    # 1. Define timing parameters for the test
    nirs_start_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    protocol_start_offset_s = 5
    protocol_start_time = nirs_start_time.timestamp() + protocol_start_offset_s
    fs = 10  # Hz
    n_timepoints = 100  # 10 seconds of data

    # 2. Create a fake .jsonl protocol file
    log_file = tmp_path / "protocol.jsonl"
    with open(log_file, 'w') as f:
        json.dump({'timestamp': protocol_start_time, 'event': 'start'}, f)

    # 3. Create a fake SNIRF file on disk (needed for the adapter to find it)
    fake_snirf_file = tmp_path / 'test_MOMENTS.snirf'
    fake_snirf_file.touch()

    # 4. Create mock SNIRF object with stimulus event containing NIRS start timestamp
    # The stimulus event should contain the NIRS start time, not the protocol start time
    fake_snirf_obj = create_mock_snirf_with_stimulus(
        start_timestamp=nirs_start_time.timestamp(),
        n_timepoints=n_timepoints,
        n_channels=1,  # Minimal for this test
        n_wavelengths=1,
        n_moments=1,
        fs=fs
    )
    
    # 5. Run the adapter with the mocked snirf object
    adapter = Flow2MomentsSessionAdapter()
    with patch('snirf.Snirf', return_value=fake_snirf_obj):
        result = adapter.extract_data(tmp_path)

    # 6. Assert the results
    # The data should be trimmed by `protocol_start_offset_s * fs` samples
    expected_trimmed_samples = protocol_start_offset_s * fs
    expected_remaining_samples = n_timepoints - expected_trimmed_samples
    
    assert result['data'].shape[0] == expected_remaining_samples
    assert result['time'].shape[0] == expected_remaining_samples

    # The new start time should be very close to the protocol start time
    actual_start_time = result['time'][0]
    time_difference = abs(actual_start_time - protocol_start_time)
    
    # Assert that the difference is less than half a sample period
    assert time_difference < (1 / fs) / 2 