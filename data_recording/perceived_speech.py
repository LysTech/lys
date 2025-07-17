import whisper
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from lys.utils.paths import check_file_exists
from lys.interfaces.event import Event, PauseEvent, ResumeEvent
from lys.interfaces.task_executor import TaskExecutor
import time
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QApplication
from PyQt5.QtCore import QTimer, pyqtSignal
import sys
import queue
import sounddevice as sd
import soundfile as sf
from PyQt5.QtWidgets import QLabel
import numpy as np

#TODO: it's bad that i don't currently do executor.run() in the main block


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


class AudioPlayerInterface:
    """
    Interface for an audio player. Concrete implementations should implement these methods.
    """
    def play(self):
        raise NotImplementedError
    def pause(self):
        raise NotImplementedError
    def resume(self):
        raise NotImplementedError
    def stop(self):
        raise NotImplementedError
    def is_playing(self) -> bool:
        raise NotImplementedError
    def set_on_word_boundary(self, callback):
        """Register a callback for word boundary events."""
        raise NotImplementedError
    def set_on_pause(self, callback):
        raise NotImplementedError
    def set_on_resume(self, callback):
        raise NotImplementedError
    def set_on_stop(self, callback):
        raise NotImplementedError


class SoundDeviceAudioPlayer(AudioPlayerInterface, QWidget):
    """
    Audio player using sounddevice for reliable, sample-accurate playback,
    with PyQt5 GUI controls and word display.
    """
    playbackFinished = pyqtSignal()

    def __init__(self, audio_path: Path, transcript_path: Path, timing_offset_ms: int = 75):
        super().__init__()
        QWidget.__init__(self)

        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.timing_offset_ms = timing_offset_ms

        # Load audio data using soundfile
        self.audio_data, self.samplerate = sf.read(str(audio_path), dtype='float32')

        self.stream: Optional[sd.OutputStream] = None
        self.current_frame = 0
        self.paused = False

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

        # Connect the finished signal to the main-thread handler
        self.playbackFinished.connect(self.on_playback_finished_main_thread)

        # Load and parse transcript
        self.words = self._parse_transcript(transcript_path)
        if self.words:
            offset = self.words[0]['start_ms']
            for word in self.words:
                word['start_ms'] -= offset
                word['end_ms'] -= offset
        
        self.word_index = 0
        self._on_word_boundary = None
        self._on_pause = None
        self._on_resume = None
        self._on_stop = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._check_word_boundary)

    @staticmethod
    def _parse_transcript(transcript_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a transcript file in the format:
        [hh:mm:ss.sss --> hh:mm:ss.sss]   word
        Returns a list of dicts with keys: word, start_ms, end_ms, confidence.
        Raises FileNotFoundError if transcript is not found.
        Raises ValueError if a line is malformed.
        """
        import re
        def timestamp_to_ms(ts: str) -> int:
            """Convert a timestamp string hh:mm:ss.sss to milliseconds."""
            m = re.match(r"(\d+):(\d+):(\d+\.\d+)", ts)
            if not m:
                raise ValueError(f"Invalid timestamp format: {ts}")
            h, m_, s = m.groups()
            return int(float(h) * 3600000 + float(m_) * 60000 + float(s) * 1000)
        
        words = []
        # Allow for zero or more spaces after the timestamp, and handle cases with no word.
        line_re = re.compile(r"\[(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.*)")
        with open(transcript_path, 'r') as f:
            for i, line in enumerate(f):
                line_stripped = line.strip()
                if not line_stripped: # Ignore empty lines
                    continue

                match = line_re.match(line_stripped)
                if match:
                    start, end, word = match.groups()
                    # Always add the event, even if the word is an empty string.
                    words.append({
                        'word': word.strip(),
                        'start_ms': timestamp_to_ms(start),
                        'end_ms': timestamp_to_ms(end),
                        'confidence': 1.0
                    })
                else:
                    raise ValueError(f"Malformed transcript line {i+1}: {line_stripped}")
        return words

    def audio_callback(self, outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
        """The heart of sounddevice playback, called by the audio thread."""
        if status:
            print(status, file=sys.stderr)
        
        if self.paused:
            outdata.fill(0)
            return

        remaining_frames = len(self.audio_data) - self.current_frame
        
        if remaining_frames <= 0:
            outdata.fill(0)
            raise sd.CallbackStop
        
        valid_frames = min(frames, remaining_frames)
        chunk = self.audio_data[self.current_frame : self.current_frame + valid_frames]
        # Ensure chunk is 2D (frames, channels)
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        outdata[:valid_frames] = chunk
        if valid_frames < frames:
            outdata[valid_frames:] = 0

        self.current_frame += valid_frames

    def on_playback_finished(self):
        """
        Called by the sounddevice thread when playback is naturally complete.
        This method MUST be thread-safe. Do not interact with GUI elements directly.
        """
        print("Playback finished callback from audio thread.")
        self.playbackFinished.emit()

    def on_playback_finished_main_thread(self):
        """This slot is executed in the main GUI thread and is safe for GUI updates."""
        print("Playback finished handler in main thread.")
        self.timer.stop()
        self.word_index = 0
        self.current_frame = 0
        self.word_label.setText('')
        if self.stream:
            self.stream.close()
            self.stream = None
        if self._on_stop:
            self._on_stop()
        from PyQt5.QtWidgets import QApplication
        QApplication.instance().quit()

    def play(self):
        print("Play pressed")
        if self.stream:
            self.stream.close()
        
        self.current_frame = 0
        self.word_index = 0

        try:
            device_info = sd.query_devices(sd.default.device, 'output')
            low_latency = device_info['default_low_output_latency']
        except Exception as e:
            print(f"Could not query device latency, falling back to 'low': {e}")
            low_latency = 'low'
        
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self.audio_callback,
            finished_callback=self.on_playback_finished,
            latency=low_latency,
            blocksize=512  # Using a smaller blocksize for lower latency
        )
        self.stream.start()
        self.timer.start(10)

    def pause(self):
        if self.stream and self.stream.active and not self.paused:
            print("Pause pressed")
            self.paused = True
            self.timer.stop()
            if self._on_pause:
                self._on_pause()

    def resume(self):
        if self.stream and self.paused:
            print("Resume pressed")
            self.paused = False
            self.timer.start(10)
            if self._on_resume:
                self._on_resume()

    def stop(self):
        print("Stop pressed")
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.timer.stop()
        self.current_frame = 0
        self.word_index = 0
        self.word_label.setText('')
        if self._on_stop:
            self._on_stop()

    def closeEvent(self, event):
        """Ensure the audio stream is stopped when the window is closed."""
        print("Closing window, stopping audio stream.")
        self.stop()
        event.accept()

    def is_playing(self) -> bool:
        return self.stream is not None and self.stream.active and not self.paused

    def set_on_word_boundary(self, callback):
        self._on_word_boundary = callback

    def set_on_pause(self, callback):
        self._on_pause = callback

    def set_on_resume(self, callback):
        self._on_resume = callback

    def set_on_stop(self, callback):
        self._on_stop = callback

    def _check_word_boundary(self):
        if not self.is_playing() or not self.words:
            return

        if self.word_index >= len(self.words):
            return

        # Reliable timing based on frames played
        current_ms = (self.current_frame / self.samplerate) * 1000
        
        # Apply a manual offset to compensate for audio buffer latency, aligning the
        # displayed word more closely with the audible sound.
        adjusted_ms = current_ms - self.timing_offset_ms

        word = self.words[self.word_index]
        
        if adjusted_ms >= word['start_ms']:
            self.word_label.setText(word['word'])
            if self._on_word_boundary:
                self._on_word_boundary(word)
            self.word_index += 1


class PerceivedSpeechTaskExecutor(TaskExecutor):
    """
    TaskExecutor for perceived speech tasks. Streams PerceivedWordEvent, PauseEvent, ResumeEvent, etc. in real time.
    Integrates with an AudioPlayerInterface GUI for real-time control and logging.
    Uses a thread-safe queue to yield events as they occur.
    Log file is now always saved in the session_path directory as 'perceived_speech_log.jsonl'.
    """
    def __init__(self, transcript_path: Path, audio_path: Path):
        super().__init__()
        self.transcript_path = transcript_path
        self.audio_path = audio_path
        self.log_file = None
        self.event_queue = queue.Queue()
        self._stopped = False
        self.player = None
        self.poll_timer = None
        self.app = None

    def log_event(self, event: Event):
        if self.log_file:
            self.log_file.write(json.dumps(event.to_dict()) + "\n")
            self.log_file.flush()

    def on_word(self, word: Dict[str, Any]):
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
        self.event_queue.put(None) # None is a stop signal

    def event_stream(self, session_path: Path) -> Iterator[Event]:
        """
        Yield events from the queue as they arrive. Ends when a stop event is received.
        """
        while True:
            try:
                event = self.event_queue.get_nowait()
                if event is None:  # Stop signal
                    break
                yield event
            except queue.Empty:
                if self._stopped:
                    break
                time.sleep(0.01)

    def _start(self, session_path: Path):
        """
        Start the perceived speech task executor. The log file is created in the session_path directory as 'perceived_speech_log_{timestamp}.jsonl', where timestamp is in YYYYMMDD_HHMMSS format.
        """
        import sys
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = session_path / f"perceived_speech_log_{timestamp}.jsonl"
        session_path.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, 'w')

        self.app = QApplication(sys.argv)
        # Adjust timing_offset_ms to fine-tune the synchronization between the
        # displayed word and the played audio. A value between 50-150ms is typical.
        self.player = SoundDeviceAudioPlayer(self.audio_path, self.transcript_path, timing_offset_ms=100)
        self.player.set_on_word_boundary(self.on_word)
        self.player.set_on_pause(self.on_pause)
        self.player.set_on_resume(self.on_resume)
        self.player.set_on_stop(self.on_stop)
        self.player.show()

        def poll_event_stream():
            try:
                event = self.event_queue.get_nowait()
                if event is None:
                    if self.poll_timer is not None:
                        self.poll_timer.stop()
                    return
                self.log_event(event)
            except queue.Empty:
                pass

        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(poll_event_stream)
        self.poll_timer.start(50)  # Poll every 50ms

        self.app.exec_()

# --- Main block ---
if __name__ == "__main__":
    audio_path = Path("churchill_chapter1_16k_mono.wav")
    transcript_path = Path("transcription.txt")
    # session_path is unused in this executor, but required by interface
    session_path = Path(".")
    executor = PerceivedSpeechTaskExecutor(transcript_path, audio_path)
    executor.start(session_path)