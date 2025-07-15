import numpy as np
from pathlib import Path
from processing.preprocessing import BettinaSessionAdapter, RawSessionPreProcessor
import pytest
from unittest.mock import MagicMock, patch
from processing.preprocessing import Flow2MomentsSessionAdapter


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
    # Create a fake snirf file
    fake_file = tmp_path / 'bar_MOMENTS.snirf'
    fake_file.touch()

    # Mock the snirf.Snirf object and its structure
    n_timepoints = 5
    n_channels = 2
    n_wavelengths = 2
    n_moments = 3
    # Build fake measurementList
    class FakeM:
        def __init__(self, src, det, wav, dtype_idx, dunit):
            self.sourceIndex = src
            self.detectorIndex = det
            self.wavelengthIndex = wav
            self.dataTypeIndex = dtype_idx
            self.dataUnit = dunit
    mlist = [
        FakeM(1, 1, 1, 2, ''),   # amplitude
        FakeM(1, 1, 1, 1, 'ps'), # mean time
        FakeM(1, 1, 1, 3, 'ps^2'), # variance
        FakeM(1, 1, 2, 2, ''),
        FakeM(1, 1, 2, 1, 'ps'),
        FakeM(1, 1, 2, 3, 'ps^2'),
        FakeM(2, 2, 1, 2, ''),
        FakeM(2, 2, 1, 1, 'ps'),
        FakeM(2, 2, 1, 3, 'ps^2'),
        FakeM(2, 2, 2, 2, ''),
        FakeM(2, 2, 2, 1, 'ps'),
        FakeM(2, 2, 2, 3, 'ps^2'),
    ]
    # Fake data: (n_timepoints, n_measurements)
    fake_data = np.arange(n_timepoints * len(mlist)).reshape(n_timepoints, len(mlist))
    fake_time = np.linspace(0, 1, n_timepoints)
    fake_data_block = MagicMock()
    fake_data_block.measurementList = mlist
    fake_data_block.dataTimeSeries = fake_data
    fake_data_block.time = fake_time
    fake_nirs = MagicMock()
    fake_nirs.data = [fake_data_block]
    fake_snirf = MagicMock()
    fake_snirf.nirs = [fake_nirs]
    # Patch snirf.Snirf to return our fake object
    with patch('snirf.Snirf', return_value=fake_snirf):
        adapter = Flow2MomentsSessionAdapter()
        result = adapter.extract_data(tmp_path)
    arr = result['data']
    assert arr.shape == (n_timepoints, n_channels, n_wavelengths, n_moments)
    assert set(result['moment_names']) == {'amplitude', 'mean_time_of_flight', 'variance'}
    assert len(result['channels']) == n_channels
    assert len(result['wavelengths']) == n_wavelengths
    np.testing.assert_array_equal(result['time'], fake_time) 