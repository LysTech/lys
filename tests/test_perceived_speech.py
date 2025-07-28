import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock, call
import numpy as np
import tempfile
import time
from queue import Queue

from lys.data_recording.perceived_speech import (
    PerceivedSpeechTaskExecutor, 
    PerceivedWordEvent, 
    SoundDeviceAudioPlayer,
    AudioEventLogger,
    AudioSessionManager,
    AudioPlaybackConfig,
    validate_audio_transcript_folder
)
from lys.abstract_interfaces.event import PauseEvent, ResumeEvent


def test_on_word_raises_error_if_confidence_is_missing():
    """Check that on_word raises KeyError if confidence is not in the data."""
    executor = PerceivedSpeechTaskExecutor()
    word_data = {"word": "test", "start_ms": 0, "end_ms": 100}  # No confidence key, we expect this from whisper.cpp

    with pytest.raises(KeyError):
        executor.on_word(word_data)


def create_transcript_file(path: Path, content: str):
    """Helper to create a transcript file."""
    path.write_text(content)


def test_parse_transcript_valid(tmp_path):
    """Test parsing a valid transcript file with a single JSON object containing a transcription list.
    
        Don't be confused by this:
        - the transcript_content is meant to look like the transcript file created by whisper.cpp, which you'll fine in the audio assets folder,
          it's not mean to look like the log file.
    """
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


# ====================================
# NEW TESTS - AudioEventLogger
# ====================================

def test_audio_event_logger_creates_log_file(tmp_path):
    """AudioEventLogger creates a log file in the session path."""
    logger = AudioEventLogger(tmp_path)
    
    # Check that a log file was created
    log_files = list(tmp_path.glob("perceived_speech_log_*.jsonl"))
    assert len(log_files) == 1
    assert log_files[0].exists()
    
    logger.close()


def test_audio_event_logger_logs_events(tmp_path):
    """AudioEventLogger writes events to the log file."""
    logger = AudioEventLogger(tmp_path)
    
    # Create and log some events
    word_event = PerceivedWordEvent("hello", 1000, 2000, 0.95)
    pause_event = PauseEvent()
    
    logger.log_event(word_event)
    logger.log_event(pause_event)
    logger.close()
    
    # Read the log file and verify content
    log_files = list(tmp_path.glob("perceived_speech_log_*.jsonl"))
    content = log_files[0].read_text().strip().split('\n')
    
    assert len(content) == 2
    
    # Parse and verify the logged events
    logged_word = json.loads(content[0])['event']
    assert logged_word["event_type"] == "perceived_word"
    assert logged_word["word"] == "hello"
    assert logged_word["confidence"] == 0.95
    
    logged_pause = json.loads(content[1])['event']
    assert logged_pause["event_type"] == "pause"


def test_audio_event_logger_queue_operations(tmp_path):
    """AudioEventLogger properly handles event queueing."""
    logger = AudioEventLogger(tmp_path)
    
    # Test queueing and retrieving events
    event1 = PerceivedWordEvent("test", 0, 100, 1.0)
    event2 = PauseEvent()
    
    logger.queue_event(event1)
    logger.queue_event(event2)
    
    # Retrieve events
    retrieved1 = logger.get_queued_event()
    retrieved2 = logger.get_queued_event()
    none_event = logger.get_queued_event()  # Should be None when queue empty
    
    assert isinstance(retrieved1, PerceivedWordEvent)
    assert retrieved1.word == "test"
    assert isinstance(retrieved2, PauseEvent)
    assert none_event is None
    
    logger.close()


# ====================================
# NEW TESTS - AudioSessionManager
# ====================================

def test_audio_session_manager_load_valid_folder(tmp_path):
    """AudioSessionManager successfully loads from a valid folder."""
    # Create test files
    audio_file = tmp_path / "test.wav"
    transcript_file = tmp_path / "test.json"
    audio_file.touch()
    transcript_file.touch()
    
    manager = AudioSessionManager()
    result = manager.load_from_folder(tmp_path)
    
    assert result is True
    assert manager.is_ready() is True
    assert manager.audio_path == audio_file
    assert manager.transcript_path == transcript_file


def test_audio_session_manager_load_invalid_folder(tmp_path):
    """AudioSessionManager fails with invalid folder structure."""
    # Create too many files
    (tmp_path / "test1.wav").touch()
    (tmp_path / "test2.wav").touch()
    (tmp_path / "test.json").touch()
    
    manager = AudioSessionManager()
    result = manager.load_from_folder(tmp_path)
    
    assert result is False
    assert manager.is_ready() is False
    assert manager.audio_path is None


def test_audio_session_manager_get_files_when_not_ready():
    """AudioSessionManager raises error when getting files before loading."""
    manager = AudioSessionManager()
    
    with pytest.raises(ValueError, match="Audio session not ready"):
        manager.get_files()


def test_validate_audio_transcript_folder():
    """Test the folder validation helper function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test valid folder
        audio_file = tmp_path / "test.wav"
        transcript_file = tmp_path / "test.json"
        audio_file.touch()
        transcript_file.touch()
        
        result = validate_audio_transcript_folder(tmp_path)
        assert result is not None
        assert result[0] == audio_file
        assert result[1] == transcript_file
        
        # Test invalid folder (too many wav files)
        (tmp_path / "test2.wav").touch()
        result = validate_audio_transcript_folder(tmp_path)
        assert result is None


# ====================================
# NEW TESTS - PerceivedSpeechTaskExecutor
# ====================================

def test_executor_event_handlers():
    """Test that the executor properly creates and queues events."""
    executor = PerceivedSpeechTaskExecutor()
    
    # Mock the event logger
    mock_logger = Mock()
    executor.event_logger = mock_logger
    
    # Test word event
    word_data = {"word": "test", "start_ms": 100, "end_ms": 200, "confidence": 0.9}
    executor.on_word(word_data)
    
    # Verify the correct event was queued
    mock_logger.queue_event.assert_called_once()
    queued_event = mock_logger.queue_event.call_args[0][0]
    assert isinstance(queued_event, PerceivedWordEvent)
    assert queued_event.word == "test"
    assert queued_event.confidence == 0.9
    
    # Test pause event
    mock_logger.reset_mock()
    executor.on_pause()
    mock_logger.queue_event.assert_called_once()
    pause_event = mock_logger.queue_event.call_args[0][0]
    assert isinstance(pause_event, PauseEvent)
    
    # Test resume event
    mock_logger.reset_mock()
    executor.on_resume()
    mock_logger.queue_event.assert_called_once()
    resume_event = mock_logger.queue_event.call_args[0][0]
    assert isinstance(resume_event, ResumeEvent)


def test_executor_stop_handling():
    """Test that the executor properly handles stop events."""
    executor = PerceivedSpeechTaskExecutor()
    mock_logger = Mock()
    executor.event_logger = mock_logger
    
    # Mock the _notify_stop method to avoid actual notification
    executor._notify_stop = Mock()
    
    executor.on_stop()
    
    assert executor._stopped is True
    mock_logger.signal_stop.assert_called_once()
    executor._notify_stop.assert_called_once()


# ====================================
# NEW TEST - End-to-end Integration
# ====================================

@patch('lys.data_recording.perceived_speech.sd.OutputStream')
@patch('lys.data_recording.perceived_speech.sf.read')
def test_end_to_end_perceived_speech_flow(mock_sf_read, mock_output_stream, tmp_path):
    """
    End-to-end test that simulates the complete perceived speech flow without GUI or actual audio.
    This is similar to the main block but fully automated and testable.
    """
    # Setup mock audio data
    mock_audio_data = np.random.random(44100).astype('float32')  # 1 second of fake audio
    mock_sf_read.return_value = (mock_audio_data, 44100)
    
    # Create test files
    audio_file = tmp_path / "test.wav"
    transcript_file = tmp_path / "test.json"
    session_path = tmp_path / "session"
    
    # Create a realistic transcript
    transcript_content = {
        "transcription": [
            {"offsets": {"from": 0, "to": 500}, "text": "hello"},
            {"offsets": {"from": 500, "to": 1000}, "text": "world"}
        ]
    }
    
    audio_file.write_bytes(b"fake wav data")
    transcript_file.write_text(json.dumps(transcript_content))
    
    # Mock the audio stream
    mock_stream = Mock()
    mock_output_stream.return_value = mock_stream
    
    # Create executor and start the session
    config = AudioPlaybackConfig(
        timing_offset_ms=0,  # No offset for predictable testing
        poll_interval_ms=10,  # Fast polling for quick test
        word_boundary_check_interval_ms=5
    )
    executor = PerceivedSpeechTaskExecutor(config)
    
    # Mock Qt components to avoid GUI
    with patch('lys.data_recording.perceived_speech.QTimer') as mock_timer:
        mock_timer_instance = Mock()
        mock_timer.return_value = mock_timer_instance
        
        # Start the session with files directly
        executor.start_with_files(audio_file, transcript_file, session_path)
        
        # Verify setup
        assert executor.event_logger is not None
        assert executor.player is not None
        assert executor.session_manager.is_ready()
        
        # Simulate word boundary events by calling the handler directly
        # This mimics what would happen during actual playback
        word1_data = {"word": "hello", "start_ms": 0, "end_ms": 500, "confidence": 1.0}
        word2_data = {"word": "world", "start_ms": 500, "end_ms": 1000, "confidence": 1.0}
        
        executor.on_word(word1_data)
        executor.on_pause()
        executor.on_resume()
        executor.on_word(word2_data)
        executor.on_stop()
        
        # Process the event queue manually (normally done by timer)
        events_logged = []
        while True:
            event = executor.event_logger.get_queued_event()
            if event is None:
                break
            events_logged.append(event)
            executor.event_logger.log_event(event)
        
        # Verify the expected events were generated
        assert len(events_logged) == 4  # 2 words + pause + resume
        
        # Check event types and content
        assert isinstance(events_logged[0], PerceivedWordEvent)
        assert events_logged[0].word == "hello"
        assert isinstance(events_logged[1], PauseEvent)
        assert isinstance(events_logged[2], ResumeEvent)
        assert isinstance(events_logged[3], PerceivedWordEvent)
        assert events_logged[3].word == "world"
        
        # Verify log file was created and contains events
        log_files = list(session_path.glob("perceived_speech_log_*.jsonl"))
        assert len(log_files) == 1
        
        # Clean up
        executor.event_logger.close()


def test_perceived_word_event_serialization():
    """Test that PerceivedWordEvent properly serializes to dict."""
    event = PerceivedWordEvent("test", 100, 200, 0.95)
    
    event_dict = event.to_dict()['event'] #structure is {timestamp: ..., event: ...}
    
    assert event_dict["event_type"] == "perceived_word"
    assert event_dict["word"] == "test"
    assert event_dict["start_ms"] == 100
    assert event_dict["end_ms"] == 200
    assert event_dict["confidence"] == 0.95


def test_audio_playback_config_defaults():
    """Test that AudioPlaybackConfig has sensible defaults."""
    config = AudioPlaybackConfig()
    
    assert config.timing_offset_ms == 100
    assert config.poll_interval_ms == 50
    assert config.audio_buffer_size == 512
    assert config.word_boundary_check_interval_ms == 10
    assert config.default_timing_offset == 75
