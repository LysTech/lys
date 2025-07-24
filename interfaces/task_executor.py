from abc import ABC, abstractmethod
from pathlib import Path
import logging
import json
from typing import Iterator, Callable, Optional
from utils.json_logging import setup_json_logging
from interfaces.event import Event, PauseEvent, ResumeEvent, StartEvent
from utils.paths import create_session_path, get_subjects_dir, get_next_session_number

class TaskExecutor(ABC):
    """
    Abstract base class for task executors.
    
    TaskExecutor coordinates with Task and DeviceManager through a callback-based pattern
    to ensure accurate timing of device protocol markers with actual protocol execution.
    
    Callback Coordination Pattern:
    - Task registers start/stop callbacks during initialization
    - TaskExecutor calls _notify_start() when protocol actually begins (not GUI launch)
    - This triggers Task to call DeviceManager.mark_protocol_start() at the right moment
    - TaskExecutor calls _notify_stop() when protocol ends
    - This triggers Task to call DeviceManager.mark_protocol_end() for protocol end marking
    
    Implementation Requirements:
    1. Implement _start(session_path) for task-specific logic (GUI launch, etc.)
    2. Call _notify_start() when actual protocol execution begins (event logging starts)
    3. Call _notify_stop() when protocol execution ends
    4. Use log_event() to record all protocol events
    
    Example timing for GUI-based executor:
    - start() called → GUI launches, but protocol not yet started
    - User interaction (e.g., clicks play) → call _notify_start() → device marks protocol start
    - Protocol runs with event logging
    - User stops or completion → call _notify_stop() → device marks protocol end
    
    For non-GUI executors, use run_event_loop() to automatically log events from event_stream().
    """
    def __init__(self):
        setup_json_logging()
        self._on_stop_callback: Optional[Callable[[], None]] = None
        self._on_start_callback: Optional[Callable[[], None]] = None

    def set_on_stop_callback(self, callback: Callable[[], None]):
        """
        Register a callback to be called when the executor stops.
        Used by Task to coordinate device manager finalisation.
        """
        self._on_stop_callback = callback

    def set_on_start_callback(self, callback: Callable[[], None]):
        """
        Register a callback to be called when the executor actually starts the protocol.
        Used by Task to coordinate device manager initialisation with actual protocol start.
        """
        self._on_start_callback = callback

    def _notify_stop(self):
        """
        Notify registered callback that the executor is stopping.
        Subclasses should call this when they are about to stop.
        """
        if self._on_stop_callback:
            self._on_stop_callback()

    def _notify_start(self):
        """
        Notify registered callback that the executor is actually starting the protocol.
        Subclasses should call this when they begin actual event logging/protocol execution.
        """
        if self._on_start_callback:
            self._on_start_callback()

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