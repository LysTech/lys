import whisper
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple
from lys.utils.paths import check_file_exists, get_audio_assets_path
from lys.interfaces.event import Event, PauseEvent, ResumeEvent
from lys.interfaces.task_executor import TaskExecutor
import time
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QApplication, QFileDialog, QLabel, QVBoxLayout
from PyQt5.QtCore import QTimer, pyqtSignal, QObject
import sys
import queue
import sounddevice as sd
import soundfile as sf
import numpy as np


def validate_audio_transcript_folder(folder: Path) -> Optional[Tuple[Path, Path]]:
    """
    Given a folder, returns (audio_path, transcript_path) if exactly one .wav and one .json file exist, else None.
    """
    wavs = list(folder.glob('*.wav'))
    jsons = list(folder.glob('*.json'))
    if len(wavs) == 1 and len(jsons) == 1:
        return wavs[0], jsons[0]
    return None


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


class PerceivedSpeechTaskExecutor(TaskExecutor):
    """
    TaskExecutor for perceived speech tasks. Streams PerceivedWordEvent, PauseEvent, ResumeEvent, etc. in real time.
    Now expects audio and transcript paths to be provided at playback time, not at construction.
    """
    def __init__(self):
        super().__init__()
        self.log_file = None
        self.event_queue = queue.Queue()
        self._stopped = False
        self.player = None
        self.widget = None
        self.app = None
        self.audio_path = None
        self.transcript_path = None

    def _start(self, session_path: Path):
        """
        This method is required by TaskExecutor but is not used. Use start() instead.
        """
        raise NotImplementedError("Use start() instead of _start() for PerceivedSpeechTaskExecutor.")

    def log_event(self, event: Event):
        if self.log_file:
            self.log_file.write(json.dumps(event.to_dict()) + "\n")
            self.log_file.flush()

    def on_word(self, word: dict):
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

    def start(self, session_path: Path):
        """
        Launches the GUI for folder selection and playback. The user selects a folder containing audio and transcript.
        """
        import sys
        from PyQt5.QtWidgets import QApplication
        self.app = QApplication(sys.argv)
        self.widget = PerceivedSpeechWidget(self)
        self.widget.show()
        self.session_path = session_path
        self.app.exec_()

    def start_with_files(self, audio_path: Path, transcript_path: Path, session_path: Path):
        """
        Start playback and logging with the given audio and transcript files.
        """
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = session_path / f"perceived_speech_log_{timestamp}.jsonl"
        session_path.mkdir(parents=True, exist_ok=True)
        self.log_file = open(log_path, 'w')
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.player = SoundDeviceAudioPlayer(audio_path, transcript_path, timing_offset_ms=100)
        if self.widget is not None:
            self.widget.set_player(self.player)
        # Connect player signals to event queue
        self.player.word_boundary.connect(self.on_word)
        self.player.playback_finished.connect(self._on_playback_finished)
        # Use a QTimer to periodically check the event queue and log events
        def poll_event_queue():
            while True:
                try:
                    event = self.event_queue.get_nowait()
                    if event is None:
                        if self.app is not None:
                            self.app.quit()
                        return
                    self.log_event(event)
                except queue.Empty:
                    break
        self.poll_timer = QTimer()
        self.poll_timer.timeout.connect(poll_event_queue)
        self.poll_timer.start(50)

    def _on_playback_finished(self):
        # Called when playback is finished
        if self.widget is not None:
            self.widget.close()
        # The widget's closeEvent will stop the player and trigger app.quit via event queue


class SoundDeviceAudioPlayer(QObject):
    """
    Audio player using sounddevice for reliable, sample-accurate playback.
    Emits signals for playback finished and word boundary events.
    No GUI code.
    """
    playback_finished = pyqtSignal()
    word_boundary = pyqtSignal(dict)

    def __init__(self, audio_path: Path, transcript_path: Path, timing_offset_ms: int = 75):
        super().__init__()
        QObject.__init__(self)

        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.timing_offset_ms = timing_offset_ms

        # Load audio data using soundfile
        self.audio_data, self.samplerate = sf.read(str(audio_path), dtype='float32')

        self.stream: Optional[sd.OutputStream] = None
        self.current_frame = 0
        self.paused = False

        # Load and parse transcript
        self.words = self._parse_transcript(transcript_path)
        if self.words:
            offset = self.words[0]['start_ms']
            for word in self.words:
                word['start_ms'] -= offset
                word['end_ms'] -= offset
        self.word_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self._check_word_boundary)

        self._on_pause = None
        self._on_resume = None
        self._on_stop = None

    @staticmethod
    def _parse_transcript(transcript_path: Path) -> List[Dict[str, Any]]:
        """
        Parse a transcript file in the new JSON format:
        {
            ...,
            "transcription": [
                {
                    "timestamps": {"from": ..., "to": ...},
                    "offsets": {"from": int, "to": int},
                    "text": str
                },
                ...
            ]
        }
        Returns a list of dicts with keys: word, start_ms, end_ms, confidence.
        Raises FileNotFoundError if transcript is not found or is empty.
        Raises ValueError if the JSON is malformed or missing required fields.
        """
        if not transcript_path.exists() or transcript_path.stat().st_size == 0:
            raise FileNotFoundError(f"Transcript file not found or empty: {transcript_path}")
        with open(transcript_path, 'r') as f:
            data = json.load(f)
        if 'transcription' not in data:
            raise ValueError("Transcript JSON missing 'transcription' key")
        words = []
        for i, entry in enumerate(data['transcription']):
            try:
                word = entry['text'].strip()
                offsets = entry['offsets']
                start_ms = int(offsets['from'])
                end_ms = int(offsets['to'])
                words.append({
                    'word': word,
                    'start_ms': start_ms,
                    'end_ms': end_ms,
                    'confidence': 1.0
                })
            except Exception as e:
                raise ValueError(f"Malformed transcription entry at index {i}: {entry} ({e})")
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
        self.playback_finished.emit()

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
            self.word_boundary.emit(word)
            self.word_index += 1


class PerceivedSpeechWidget(QWidget):
    """
    GUI widget for perceived speech task. Provides folder picker, Play, Pause, Resume, Stop buttons and displays the current word.
    Interacts with a SoundDeviceAudioPlayer via signals/slots.
    """
    def __init__(self, executor: PerceivedSpeechTaskExecutor):
        super().__init__()
        self.executor = executor
        self.player = None
        self.selected_folder = None
        self.audio_path = None
        self.transcript_path = None

        self.folder_button = QPushButton('Select Folder')
        self.folder_label = QLabel('No folder selected')
        self.file_label = QLabel('')
        self.play_button = QPushButton('Play')
        self.pause_button = QPushButton('Pause')
        self.resume_button = QPushButton('Resume')
        self.stop_button = QPushButton('Stop')
        self.word_label = QLabel('')

        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.stop_button.setEnabled(False)

        top_layout = QVBoxLayout()
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_button)
        folder_layout.addWidget(self.folder_label)
        top_layout.addLayout(folder_layout)
        top_layout.addWidget(self.file_label)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.resume_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.word_label)
        top_layout.addLayout(controls_layout)
        self.setLayout(top_layout)

        self.folder_button.clicked.connect(self.select_folder)
        self.play_button.clicked.connect(self.play)
        self.pause_button.clicked.connect(self.pause)
        self.resume_button.clicked.connect(self.resume)
        self.stop_button.clicked.connect(self.stop)

    def set_player(self, player: SoundDeviceAudioPlayer):
        self.player = player
        self.pause_button.setEnabled(True)
        self.resume_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.player.word_boundary.connect(self.on_word_boundary)

    def select_folder(self):
        assets_path = get_audio_assets_path()
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folder = QFileDialog.getExistingDirectory(self, 'Select Audio/Transcript Folder', str(assets_path), options=options)
        if folder:
            self.selected_folder = Path(folder)
            result = validate_audio_transcript_folder(self.selected_folder)
            if result:
                self.audio_path, self.transcript_path = result
                self.folder_label.setText(str(self.selected_folder))
                self.file_label.setText(f"Audio: {self.audio_path.name}, Transcript: {self.transcript_path.name}")
                self.play_button.setEnabled(True)
            else:
                self.audio_path = None
                self.transcript_path = None
                self.folder_label.setText('Invalid folder')
                self.file_label.setText('Folder must contain exactly one .wav and one .json file')
                self.play_button.setEnabled(False)

    def play(self):
        if self.audio_path and self.transcript_path:
            self.executor.start_with_files(self.audio_path, self.transcript_path, self.executor.session_path)
            if self.player:
                self.player.play()

    def pause(self):
        if self.player:
            self.player.pause()

    def resume(self):
        if self.player:
            self.player.resume()

    def stop(self):
        if self.player:
            self.player.stop()

    def on_word_boundary(self, word: dict):
        self.word_label.setText(word['word'])

    def closeEvent(self, event):
        if self.player:
            self.player.stop()
        event.accept()


if __name__ == "__main__":
    from lys.interfaces.task import Task
    from lys.utils.flow2_device_manager import Flow2DeviceManager
    
    subject = "thomas"
    experiment_name = "perceived_speech"
    device = "flow2"

    executor = PerceivedSpeechTaskExecutor()
    device_manager = Flow2DeviceManager()
    task = Task(executor, device_manager)
    
    session_path = task.make_new_session(subject, experiment_name, device)
    task.start(session_path)