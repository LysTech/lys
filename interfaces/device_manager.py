from abc import ABC, abstractmethod
from pathlib import Path

class DeviceManager(ABC):
    """
    Abstract base class for device managers. Responsible for device-specific actions at protocol start.
    """
    @abstractmethod
    def initialise(self, session_path: Path):
        """
        Initialise the device for a new session. Called at protocol start.
        """
        pass 