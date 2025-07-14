from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
from typing import Iterator
from utils.json_logging import setup_json_logging
from interfaces.event import Event, PauseEvent, ResumeEvent

class TaskExecutor(ABC):
    """Base class for task executors. Logs a stream of Event objects yielded by the child class."""

    def __init__(self):
        setup_json_logging()

    def run(self, session_path: Path):
        """Main loop: logs all events yielded by the child class."""
        for event in self.event_stream(session_path):
            self.log_event(event)

    def log_event(self, event: Event):
        """Log an Event object as JSON."""
        logging.info(json.dumps(event.to_dict()))

    def pause(self):
        """Pause the session and log a PauseEvent."""
        self.log_event(PauseEvent())

    def resume(self):
        """Resume the session and log a ResumeEvent."""
        self.log_event(ResumeEvent())

    @abstractmethod
    def event_stream(self, session_path: Path) -> Iterator[Event]:
        """Yield Event objects as the task progresses."""
        pass