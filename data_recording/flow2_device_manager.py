from interfaces.device_manager import DeviceManager
from pathlib import Path

class Flow2DeviceManager(DeviceManager):
    """
    Device manager for the Kernel Flow2 device. Handles device-specific initialisation.
    """
    def initialise(self, session_path: Path):
        """
        Send a message to the device to log protocol start. (Not yet implemented.)
        """
        pass 