import numpy as np
from pathlib import Path
from processing.preprocessing import BettinaSessionAdapter, RawSessionPreProcessor
import pytest


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