import pytest
from pathlib import Path
import json
from lys.objects.protocol import Protocol

@pytest.fixture
def temp_session_dir(tmp_path: Path) -> Path:
    """Create a temporary directory to simulate a session folder."""
    session_dir = tmp_path / "session-01"
    session_dir.mkdir()
    return session_dir

@pytest.fixture
def perceived_speech_log_file(temp_session_dir: Path) -> Path:
    """Create a dummy perceived speech log file in the temp session directory."""
    log_path = temp_session_dir / "test_log.jsonl"
    log_entries = [
        {"timestamp": 1700000000.0, "event": {"event_type": "perceived_word", "word": "Hello", "start_ms": 100, "end_ms": 500}},
        {"timestamp": 1700000001.0, "event": {"event_type": "pause"}},
        {"timestamp": 1700000002.0, "event": {"event_type": "perceived_word", "word": "World", "start_ms": 600, "end_ms": 1000}},
    ]
    with open(log_path, 'w') as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + '\n')
    return log_path

def test_protocol_from_perceived_speech_log(perceived_speech_log_file: Path):
    """
    Tests that a Protocol can be created from a .jsonl log file.
    """
    session_path = perceived_speech_log_file.parent
    protocol = Protocol.from_session_path(session_path)

    # Check that it correctly parsed only the 'perceived_word' events
    assert len(protocol.intervals) == 2

    # Check the content of the first interval
    interval1 = protocol.intervals[0]
    assert interval1[0] == 0.0  # t_start (timestamp)
    assert interval1[1] == pytest.approx(0.0 + 0.4)  # t_end (timestamp + duration)
    assert interval1[2] == "Hello"  # label (word)

    # Check the content of the second interval
    interval2 = protocol.intervals[1]
    assert interval2[0] == 2.0  # t_start
    assert interval2[1] == pytest.approx(2.0 + 0.4) # t_end
    assert interval2[2] == "World"  # label
    
    # Check the unique tasks
    assert protocol.tasks == {"Hello", "World"}

def test_protocol_from_session_path_no_file(temp_session_dir: Path):
    """
    Tests that FileNotFoundError is raised when no protocol file is found.
    """
    with pytest.raises(FileNotFoundError):
        Protocol.from_session_path(temp_session_dir)

def test_from_session_path_raises_error_on_ambiguity(temp_session_dir: Path):
    """
    Tests that a ValueError is raised if both .jsonl and .prt files are present.
    """
    # Create both a .jsonl and a .prt file
    (temp_session_dir / "protocol.jsonl").touch()
    (temp_session_dir / "protocol.prt").touch()

    with pytest.raises(ValueError, match="Ambiguous protocol"):
        Protocol.from_session_path(temp_session_dir)

def test_from_session_path_raises_error_on_multiple_jsonl(temp_session_dir: Path):
    """
    Tests that a ValueError is raised if multiple .jsonl files are found.
    """
    # Create multiple .jsonl files
    (temp_session_dir / "protocol1.jsonl").touch()
    (temp_session_dir / "protocol2.jsonl").touch()

    with pytest.raises(ValueError, match="Multiple .jsonl files found"):
        Protocol.from_session_path(temp_session_dir) 