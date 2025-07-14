import whisper
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from utils.paths import check_file_exists
from interfaces.event import Event, PauseEvent, ResumeEvent
from interfaces.task_executor import TaskExecutor
import time
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QApplication
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
import sys
import queue
import threading
from PyQt5.QtWidgets import QLabel
import pygame

class PerceivedWordEvent(Event):
    """Event representing a perceived word during audio playback."""
    def __init__(self, word: str, start_ms: int, end_ms: int, confidence: float):
        self.word = word
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.confidence = confidence

    def _to_dict(self) -> dict:
        return {
            "event_type": "perceived_word",
            "word": self.word,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence
        }

def transcribe_mp3_to_json(
    audio_path: Path,
    output_file: Path,
    model_size: str = "base",
    language: str = "en"
) -> List[Dict[str, Any]]:
    """
    Transcribe an MP3 file to JSON with word-level timestamps using Whisper.

    Args:
        audio_path (Path): Path to the input MP3 file.
        output_file (Path): Path to the output JSON file.
        model_size (str): Whisper model size (e.g., 'base', 'large-v2').
        language (str): Language code for transcription.

    Returns:
        List[Dict[str, Any]]: List of word-level timing dictionaries.
    """
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=language
    )

    words_with_timing = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            words_with_timing.append({
                "word": word_info["word"].strip(),
                "start_ms": int(word_info["start"] * 1000),
                "end_ms": int(word_info["end"] * 1000),
                "confidence": word_info.get("probability", 1.0)
            })

    output_data = {
        "total_words": len(words_with_timing),
        "duration_seconds": result["segments"][-1]["end"] if result["segments"] else 0,
        "full_text": result.get("text", ""),
        "words": words_with_timing
    }

    # Ensure output directory exists using check_file_exists, create if not
    output_dir = output_file.parent
    try:
        check_file_exists(str(output_dir))
    except FileNotFoundError:
        output_dir.mkdir(parents=True, exist_ok=True)

    with output_file.open('w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved timestamps to: {output_file}")
    return words_with_timing

class AudioPlayerInterface:
    """
    Interface for an audio player. Concrete implementations (e.g., PyQt5 GUI) should implement these methods.
    """
    def play(self):
        raise NotImplementedError
    def pause(self):
        raise NotImplementedError
    def resume(self):
        raise NotImplementedError
    def is_playing(self) -> bool:
        raise NotImplementedError
    def set_on_word_boundary(self, callback):
        """Register a callback for word boundary events."""
        raise NotImplementedError

class PygameAudioPlayer(AudioPlayerInterface, QWidget):
    """
    Audio player using pygame for playback, with PyQt5 GUI controls and word display.
    """
    def __init__(self, audio_path: Path, transcript_path: Path):
        super().__init__()
        QWidget.__init__(self)
        print("Loading audio file:", str(audio_path.resolve()))
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        pygame.mixer.init()
        pygame.mixer.music.load(str(audio_path))
        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.resume_button = QPushButton('Resume')
        self.stop_button = QPushButton('Stop')
        self.word_label = QLabel('')
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.stop_button.clicked.connect(self.stop)
        layout = QHBoxLayout()
        layout.addWidget(self.play_button)
        layout.addWidget(self.pause_button)
        layout.addWidget(self.resume_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.word_label)
        self.setLayout(layout)
        # Load transcript
        with open(transcript_path, 'r') as f:
            self.transcript = json.load(f)
        self.words = self.transcript['words']
        self.word_index = 0
        self._on_word_boundary = None
        self._on_pause = None
        self._on_resume = None
        self._on_stop = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_word_boundary)

    def play(self):
        print("Play pressed")
        pygame.mixer.music.play()
        self.timer.start(10)
        print("Timer started")

    def pause(self):
        print("Pause pressed")
        pygame.mixer.music.pause()
        self.timer.stop()
        print("Timer stopped")
        if self._on_pause:
            self._on_pause()

    def resume(self):
        print("Resume pressed")
        pygame.mixer.music.unpause()
        self.timer.start(10)
        print("Timer started")
        if self._on_resume:
            self._on_resume()

    def stop(self):
        print("Stop pressed")
        pygame.mixer.music.stop()
        self.timer.stop()
        print("Timer stopped, word index reset")
        self.word_index = 0
        self.word_label.setText('')
        if self._on_stop:
            self._on_stop()

    def is_playing(self) -> bool:
        busy = pygame.mixer.music.get_busy()
        print("is_playing called, busy:", busy)
        return busy

    def set_on_word_boundary(self, callback):
        self._on_word_boundary = callback

    def set_on_pause(self, callback):
        self._on_pause = callback

    def set_on_resume(self, callback):
        self._on_resume = callback

    def set_on_stop(self, callback):
        self._on_stop = callback

    def _check_word_boundary(self):
        if self.word_index >= len(self.words):
            print("All words processed, timer stopped")
            self.timer.stop()
            self.word_label.setText('')
            return
        current_ms = pygame.mixer.music.get_pos()
        word = self.words[self.word_index]
        print(f"_check_word_boundary: current_ms={current_ms}, word_start={word['start_ms']}, word={word['word']}")
        if current_ms >= word['start_ms']:
            print(f"Displaying word: {word['word']}")
            self.word_label.setText(word['word'])
            if self._on_word_boundary:
                print(f"Calling on_word_boundary callback for word: {word['word']}")
                self._on_word_boundary(word)
            self.word_index += 1


class PerceivedSpeechTaskExecutor(TaskExecutor):
    """
    TaskExecutor for perceived speech tasks. Streams PerceivedWordEvent, PauseEvent, ResumeEvent, etc. in real time.
    Integrates with a PyQt5AudioPlayer GUI for real-time control and logging.
    Uses a thread-safe queue to yield events as they occur.
    """
    def __init__(self, transcript_path: Path, audio_path: Path, gui: Optional[AudioPlayerInterface] = None, log_path: Optional[Path] = None):
        super().__init__()
        self.transcript_path = transcript_path
        self.audio_path = audio_path
        self.gui = gui
        self.log_file = open(log_path, 'w') if log_path else None
        self.event_queue = queue.Queue()
        self._stopped = False

        if self.gui:
            self.gui.set_on_word_boundary(self.on_word)
            self.gui.set_on_pause(self.on_pause)
            self.gui.set_on_resume(self.on_resume)
            self.gui.set_on_stop(self.on_stop)

    def log_event(self, event):
        if self.log_file:
            self.log_file.write(json.dumps(event.to_dict()) + "\n")
            self.log_file.flush()

    def on_word(self, word):
        event = PerceivedWordEvent(
            word=word['word'],
            start_ms=word['start_ms'],
            end_ms=word['end_ms'],
            confidence=word['confidence']
        )
        self.event_queue.put(event)

    def on_pause(self):
        self.event_queue.put(PauseEvent())

    def on_resume(self):
        self.event_queue.put(ResumeEvent())

    def on_stop(self):
        self._stopped = True

    def event_stream(self, session_path: Path):
        """
        Yield events from the queue as they arrive. Ends when stopped.
        """
        while not self._stopped:
            try:
                event = self.event_queue.get(timeout=0.1)
                yield event
            except queue.Empty:
                continue

# --- Main block ---
if __name__ == "__main__":
    audio_path = Path("churchill_chapter1.wav")
    transcript_path = Path("../protocols/churchill_chapter1_timestamps.json")
    log_path = Path("test_log.jsonl")
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    app = QApplication(sys.argv)
    player = PygameAudioPlayer(audio_path, transcript_path)
    player.show()
    executor = PerceivedSpeechTaskExecutor(transcript_path, audio_path, gui=player, log_path=log_path)
    def poll_event_stream():
        try:
            event = executor.event_queue.get_nowait()
            executor.log_event(event)
        except queue.Empty:
            pass
    poll_timer = QTimer()
    poll_timer.timeout.connect(poll_event_stream)
    poll_timer.start(50)
    sys.exit(app.exec_())