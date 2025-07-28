from abc import ABC, abstractmethod
from pathlib import Path

class DeviceManager(ABC):
    """
    Abstract base class for device managers. 
    
    DeviceManager handles device-specific protocol start/end marking during protocol execution.
    It works in coordination with Task and TaskExecutor through a callback-based pattern:
    
    Lifecycle:
    1. Task registers callbacks with TaskExecutor during Task.__init__()
    2. Task.start() launches TaskExecutor (e.g., GUI) but does NOT mark protocol start
    3. When protocol actually begins (e.g., user clicks play), TaskExecutor calls _notify_start()
    4. Task receives callback and calls DeviceManager.mark_protocol_start() - protocol start marked in device
    5. Protocol runs with device recording and protocol markers active
    6. When protocol ends, TaskExecutor calls _notify_stop()
    7. Task receives callback and calls DeviceManager.mark_protocol_end() - protocol end marked in device
    
    This ensures device protocol markers are accurately timed with actual protocol execution,
    not just GUI launch or other preliminary setup.
    
    Subclasses must implement mark_protocol_start() and mark_protocol_end() for specific hardware.
    """
    @abstractmethod
    def mark_protocol_start(self, session_path: Path):
        """
        Mark the start of the protocol in the device's data recording.
        
        Called by Task when the protocol actually starts (not when GUI launches).
        This happens via callback when TaskExecutor calls _notify_start().
        
        For NIRS devices, this typically sends a "protocol_started" event that creates
        a timestamp marker in the SNIRF file, allowing alignment of neural data with protocol timing.
        
        Args:
            session_path: Path where session data will be stored
            
        Raises:
            Should raise appropriate exceptions if protocol start cannot be marked.
            Task will catch and log warnings but continue execution.
        """
        pass 

    @abstractmethod
    def mark_protocol_end(self):
        """
        Mark the end of the protocol in the device's data recording.
        
        Called by Task when the protocol actually ends.
        This happens via callback when TaskExecutor calls _notify_stop().
        
        For NIRS devices, this typically sends a "protocol_ended" event that creates
        a timestamp marker in the SNIRF file, marking the end of the protocol session.
        
        Raises:
            Should raise appropriate exceptions if protocol end cannot be marked.
            Task will catch and log warnings.
        """
        pass 