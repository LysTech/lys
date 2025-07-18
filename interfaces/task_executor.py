from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
from typing import Iterator
from utils.json_logging import setup_json_logging
from interfaces.event import Event, PauseEvent, ResumeEvent, StartEvent
from utils.paths import create_session_path, get_subjects_dir, get_next_session_number

class TaskExecutor(ABC):
    """
    Abstract base class for task executors.
    Subclasses must implement _start(), which is called by start().
    The base start() method logs a StartEvent and then delegates to _start().
    All events must be logged via log_event().
    For non-GUI executors, run_event_loop() can be used to log all events from event_stream().
    """
    def __init__(self):
        setup_json_logging()

    def log_event(self, event: Event):
        """Log an Event object as JSON."""
        logging.info(json.dumps(event.to_dict()))

    def start(self, session_path: Path):
        """
        Template method: logs a StartEvent and then calls the subclass's _start().
        """
        self.log_event(StartEvent())
        self._start(session_path)

    @abstractmethod
    def _start(self, session_path: Path):
        """
        Subclasses implement their task logic here. Called by start().
        """
        pass

    def run_event_loop(self, session_path: Path):
        """
        Helper for non-GUI executors: logs all events from event_stream().
        Subclasses can call this in _start() if they use a blocking event loop.
        """
        for event in self.event_stream(session_path):
            self.log_event(event)

    def pause(self):
        self.log_event(PauseEvent())

    def resume(self):
        self.log_event(ResumeEvent())

    def create_new_session_dir(self, subject: str, experiment_name: str, device: str = 'nirs') -> Path:
        """
        Create a new session directory for the given subject, experiment, and device.
        The session path is {LYS_DATA_DIR}/subjects/{subject}/{device}/{experiment_name}/session-{N},
        where N is the next available integer.
        """
        root = get_subjects_dir() / subject / device / experiment_name
        root_existed = root.exists()
        root.mkdir(parents=True, exist_ok=True)
        if not root_existed:
            print(f"[WARNING] Created new experiment directory: {root}")
        next_n = get_next_session_number(root)
        session_path = create_session_path(subject, experiment_name, device, next_n)
        return session_path

    def event_stream(self, session_path: Path) -> Iterator[Event]:
        """
        Yield Event objects as the task progresses (for non-GUI executors).
        Subclasses should override this if using run_event_loop().
        """
        raise NotImplementedError("event_stream() not implemented for this executor.")