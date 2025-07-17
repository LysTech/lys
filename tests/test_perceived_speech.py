import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from lys.data_recording.perceived_speech import PerceivedSpeechTaskExecutor, PerceivedWordEvent, SoundDeviceAudioPlayer, AudioPlayerInterface
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


@pytest.fixture
def executor_paths(tmp_path):
    """Provides temporary paths for audio, transcript, and log files."""
    return {
        "audio_path": tmp_path / "test.wav",
        "transcript_path": tmp_path / "transcript.txt",
        "log_path": tmp_path / "log.jsonl",
    }


def test_on_word_puts_event_in_queue(executor_paths):
    """Verify that on_word() creates and queues a PerceivedWordEvent."""
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
    word_data = {"word": "hello", "start_ms": 100, "end_ms": 200, "confidence": 0.9}

    executor.on_word(word_data)
    event = executor.event_queue.get_nowait()

    assert isinstance(event, PerceivedWordEvent)
    assert event.word == "hello"
    assert event.confidence == 0.9


def test_on_pause_puts_event_in_queue(executor_paths):
    """Verify that on_pause() queues a PauseEvent."""
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
    executor.on_pause()
    event = executor.event_queue.get_nowait()
    assert isinstance(event, PauseEvent)


def test_on_resume_puts_event_in_queue(executor_paths):
    """Verify that on_resume() queues a ResumeEvent."""
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
    executor.on_resume()
    event = executor.event_queue.get_nowait()
    assert isinstance(event, ResumeEvent)


def test_event_stream_stops_on_signal(executor_paths):
    """Verify that the event_stream() generator stops when a stop signal is received."""
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
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


def test_log_event_writes_to_file(executor_paths):
    """Verify that log_event() writes a JSON line to the configured log file."""
    log_path = executor_paths["log_path"]
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
    event = PerceivedWordEvent("world", 200, 300, 0.95)

    executor.log_event(event)
    
    # The log file is flushed but not closed, so we can read its contents
    content = log_path.read_text()
    logged_event = json.loads(content)
    
    assert logged_event['event']["word"] == "world"
    assert logged_event['event']["event_type"] == "perceived_word"

def test_on_word_raises_error_if_confidence_is_missing(executor_paths):
    """Check that on_word raises KeyError if confidence is not in the data."""
    executor = PerceivedSpeechTaskExecutor(**executor_paths, gui=None)
    word_data = {"word": "test", "start_ms": 0, "end_ms": 100}  # No confidence key

    with pytest.raises(KeyError):
        executor.on_word(word_data)

# --- Tests for AudioPlayerInterface ---

class DummyPlayer(AudioPlayerInterface):
    pass

@pytest.fixture
def dummy_player():
    return DummyPlayer()

def test_audio_player_interface_raises_not_implemented(dummy_player):
    """Check that all methods of the interface raise NotImplementedError."""
    with pytest.raises(NotImplementedError):
        dummy_player.play()
    with pytest.raises(NotImplementedError):
        dummy_player.pause()
    with pytest.raises(NotImplementedError):
        dummy_player.resume()
    with pytest.raises(NotImplementedError):
        dummy_player.stop()
    with pytest.raises(NotImplementedError):
        dummy_player.is_playing()
    with pytest.raises(NotImplementedError):
        dummy_player.set_on_word_boundary(None)
    with pytest.raises(NotImplementedError):
        dummy_player.set_on_pause(None)
    with pytest.raises(NotImplementedError):
        dummy_player.set_on_resume(None)
    with pytest.raises(NotImplementedError):
        dummy_player.set_on_stop(None)


# --- Tests for SoundDeviceAudioPlayer ---

def create_transcript_file(path: Path, content: str):
    """Helper to create a transcript file."""
    path.write_text(content)

def test_parse_transcript_valid(tmp_path):
    """Test parsing a valid transcript file."""
    transcript_content = """
[00:00:01.000 --> 00:00:02.000]   Hello
[00:00:02.500 --> 00:00:03.500]   world
"""
    transcript_path = tmp_path / "transcript.txt"
    create_transcript_file(transcript_path, transcript_content)

    words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    assert len(words) == 2
    assert words[0]['word'] == 'Hello'
    assert words[0]['start_ms'] == 1000
    assert words[0]['end_ms'] == 2000
    assert words[1]['word'] == 'world'
    assert words[1]['start_ms'] == 2500
    assert words[1]['end_ms'] == 3500

def test_parse_transcript_file_not_found():
    """Test parsing a non-existent transcript file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        SoundDeviceAudioPlayer._parse_transcript(Path("non_existent.txt"))

def test_parse_transcript_empty_file(tmp_path):
    """Test parsing an empty transcript file."""
    transcript_path = tmp_path / "empty.txt"
    transcript_path.touch()
    words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    assert words == []

def test_parse_transcript_malformed_line(tmp_path):
    """Test that malformed lines in a transcript raise a ValueError."""
    transcript_content = "this is not a valid line"
    transcript_path = tmp_path / "malformed.txt"
    create_transcript_file(transcript_path, transcript_content)
    with pytest.raises(ValueError, match="Malformed transcript line 1: this is not a valid line"):
        SoundDeviceAudioPlayer._parse_transcript(transcript_path)

def test_parse_transcript_line_with_no_word(tmp_path):
    """Test that a transcript line with timestamps but no word is parsed as an empty string."""
    transcript_content = "[00:00:01.000 --> 00:00:02.000]"
    transcript_path = tmp_path / "transcript.txt"
    create_transcript_file(transcript_path, transcript_content)
    words = SoundDeviceAudioPlayer._parse_transcript(transcript_path)
    assert len(words) == 1
    assert words[0]['word'] == ""
