from pathlib import Path
from .task_executor import TaskExecutor
from .device_manager import DeviceManager
from utils.paths import create_session_path, get_subjects_dir, get_next_session_number

class Task:
    """
    Represents a protocol task, coordinating a TaskExecutor and a DeviceManager.
    Handles session creation and coordinated start.
    """
    def __init__(self, executor: TaskExecutor, device_manager: DeviceManager):
        self.executor = executor
        self.device_manager = device_manager

    @staticmethod
    def make_new_session(subject: str, experiment_name: str, device: str) -> Path:
        """
        Create a new session directory for the given subject, experiment, and device.
        Returns the session path.
        """
        root = get_subjects_dir() / subject / device / experiment_name
        next_n = get_next_session_number(root)
        session_path = create_session_path(subject, experiment_name, device, next_n)
        return session_path

    def start(self, session_path: Path):
        """
        Initialise the device and start the task executor for the session.
        """
        self.device_manager.initialise(session_path)
        self.executor.start(session_path) 