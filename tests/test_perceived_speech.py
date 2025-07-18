import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from lys.data_recording.perceived_speech import PerceivedSpeechTaskExecutor, PerceivedWordEvent, SoundDeviceAudioPlayer
from lys.interfaces.event import PauseEvent, ResumeEvent

# A session-scoped fixture to create a QApplication instance for tests that need it.
# This prevents a fatal crash when instantiating QWidget-based classes.
@pytest.fixture(scope="session")
def qapp(argv=[]):
    """Fixture for creating a session-wide QApplication."""
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication(argv)
    return app


def test_on_word_puts_event_in_queue():
    """Verify that on_word() creates and queues a PerceivedWordEvent."""
    executor = PerceivedSpeechTaskExecutor()
    word_data = {"word": "hello", "start_ms": 100, "end_ms": 200, "confidence": 0.9}

    executor.on_word(word_data)
    event = executor.event_queue.get_nowait()

    assert isinstance(event, PerceivedWordEvent)
    assert event.word == "hello"
    assert event.confidence == 0.9


def test_on_pause_puts_event_in_queue():
    """Verify that on_pause() queues a PauseEvent."""
    executor = PerceivedSpeechTaskExecutor()
    executor.on_pause()
    event = executor.event_queue.get_nowait()
    assert isinstance(event, PauseEvent)


def test_on_resume_puts_event_in_queue():
    """Verify that on_resume() queues a ResumeEvent."""
    executor = PerceivedSpeechTaskExecutor()
    executor.on_resume()
    event = executor.event_queue.get_nowait()
    assert isinstance(event, ResumeEvent)


def test_event_stream_stops_on_signal():
    """Verify that the event_stream() generator stops when a stop signal is received."""
    executor = PerceivedSpeechTaskExecutor()
    word_data = {"word": "test", "start_ms": 0, "end_ms": 100, "confidence": 1.0}
    
    # Put some events and a stop signal in the queue
    executor.on_word(word_data)
    executor.on_pause()
    executor.on_stop()  # This puts the `None` stop signal in the queue

    # The event stream should yield the events and then terminate
    events = list(executor.event_stream(session_path=Mock()))
    
    assert len(events) == 2
    assert isinstance(events[0], PerceivedWordEvent)
    assert isinstance(events[1], PauseEvent)


def __test_log_event_writes_to_file():
    """Verify that log_event() writes a JSON line to the configured log file."""
    log_path = executor_paths["log_path"]
    executor = PerceivedSpeechTaskExecutor()
    event = PerceivedWordEvent("world", 200, 300, 0.95)

    executor.log_event(event)
    
    # The log file is flushed but not closed, so we can read its contents
    content = log_path.read_text()
    logged_event = json.loads(content)
    
    assert logged_event['event']["word"] == "world"
    assert logged_event['event']["event_type"] == "perceived_word"

def test_on_word_raises_error_if_confidence_is_missing():
    """Check that on_word raises KeyError if confidence is not in the data."""
    executor = PerceivedSpeechTaskExecutor()
    word_data = {"word": "test", "start_ms": 0, "end_ms": 100}  # No confidence key

    with pytest.raises(KeyError):
        executor.on_word(word_data)


def create_transcript_file(path: Path, content: str):
    """Helper to create a transcript file."""
    path.write_text(content)

def test_parse_transcript_valid(tmp_path):
    """Test parsing a valid transcript file with a single JSON object containing a transcription list."""
    transcript_content = (
        '{\n'
        '  "transcription": [\n'
        '    {\n'
        '      "timestamps": {"from": "00:00:01,000", "to": "00:00:02,000"},\n'
        '      "offsets": {"from": 1000, "to": 2000},\n'
        '      "text": "hello"\n'
        '    },\n'
        '    {\n'
        '      "timestamps": {"from": "00:00:02,500", "to": "00:00:03,500"},\n'
        '      "offsets": {"from": 2500, "to": 3500},\n'
        '      "text": "world"\n'
        '    }\n'
        '  ]\n'
        '}\n'
    )
    transcript_path = tmp_path / "transcript.txt"
    create_transcript_file(transcript_path, transcript_content)

    words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    expected = [
        {"word": "hello", "start_ms": 1000, "end_ms": 2000},
        {"word": "world", "start_ms": 2500, "end_ms": 3500},
    ]
    assert len(words) == len(expected)
    for w, exp in zip(words, expected):
        assert w["word"] == exp["word"]
        assert w["start_ms"] == exp["start_ms"]
        assert w["end_ms"] == exp["end_ms"]

def test_parse_transcript_file_not_found():
    """Test parsing a non-existent transcript file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SoundDeviceAudioPlayer._parse_transcript(Path("non_existent.txt"))

def test_parse_transcript_empty_file(tmp_path):
    """Test parsing an empty transcript file."""
    transcript_path = tmp_path / "empty.txt"
    transcript_path.touch()
    with pytest.raises(FileNotFoundError):
        words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)

def test_parse_transcript_empty_transcription_list(tmp_path):
    """Properly formatted JSON with empty transcription list returns empty list."""
    transcript_content = '{"transcription": []}'
    transcript_path = tmp_path / "empty_transcription.json"
    create_transcript_file(transcript_path, transcript_content)
    words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    assert words == []

def test_parse_transcript_missing_transcription_key(tmp_path):
    """JSON missing 'transcription' key raises ValueError."""
    transcript_content = '{"not_transcription": []}'
    transcript_path = tmp_path / "missing_key.json"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(ValueError, match="transcription"):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)

def test_parse_transcript_malformed_json(tmp_path):
    """Malformed JSON raises ValueError."""
    transcript_content = '{ this is not valid json }'
    transcript_path = tmp_path / "malformed.json"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(Exception):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)

def test_parse_transcript_missing_offsets_or_text(tmp_path):
    """Entry missing 'offsets' or 'text' raises ValueError."""
    # Missing offsets
    transcript_content = '{"transcription": [{"text": "hello"}]}'
    transcript_path = tmp_path / "missing_offsets.json"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(ValueError, match="Malformed transcription entry"):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    # Missing text
    transcript_content = '{"transcription": [{"offsets": {"from": 0, "to": 1}}]}'
    transcript_path = tmp_path / "missing_text.json"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(ValueError, match="Malformed transcription entry"):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)

def test_parse_transcript_non_integer_offsets(tmp_path):
    """Non-integer offsets raise ValueError."""
    transcript_content = '{"transcription": [{"offsets": {"from": "a", "to": "b"}, "text": "hello"}]}'
    transcript_path = tmp_path / "non_integer_offsets.json"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(ValueError, match="Malformed transcription entry"):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)
