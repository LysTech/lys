from pathlib import Path
from lys.abstract_interfaces.task_executor import TaskExecutor
from lys.abstract_interfaces.device_manager import DeviceManager
from lys.utils.paths import create_session_path, lys_subjects_dir, get_next_session_number

class Task:
    """
    Orchestrates protocol execution by coordinating a TaskExecutor and DeviceManager.
    Basically we have callbacks from the TaskExecutor to the Task, which then calls the DeviceManager
    to log "start" and "end" events in the device snirf file (these are called "protocol markers")    
    
    Coordination Architecture:
    ┌─────────────┐    callbacks    ┌──────────────┐    protocol markers ┌───────────────┐
    │    Task     │ ◄──────────────► │ TaskExecutor │                     │ DeviceManager │
    │             │                  │              │                     │               │
    │ _on_start() │                  │_notify_start()│                     │mark_protocol_ │
    │ _on_stop()  │                  │_notify_stop() │                     │start()/end() │
    └─────────────┘                  └──────────────┘                     └───────────────┘
    
    Execution Flow:
    1. Task.__init__(): Register start/stop callbacks with TaskExecutor
    2. Task.start(): Launch TaskExecutor (e.g., show GUI), store session_path
    3. User interaction triggers protocol start in TaskExecutor
    4. TaskExecutor._notify_start() → Task._on_executor_start() → DeviceManager.mark_protocol_start()
    5. Protocol runs with device recording and protocol markers active
    6. Protocol ends (user stop/completion) in TaskExecutor  
    7. TaskExecutor._notify_stop() → Task._on_executor_stop() → DeviceManager.mark_protocol_end()
    
    This design ensures:
    - Device "protocol_started" events align with actual protocol start, not GUI launch
    - Protocol markers happen at precisely the right moments in the device data
    - Clean separation of concerns between execution logic and device protocol marking
    - Robust error handling with graceful degradation
    
    Session Management:
    - Creates session directories with format: subjects/{subject}/{device}/{experiment}/session-{N}
    - Passes session_path to both executor and device manager for coordinated logging
    """
    def __init__(self, executor: TaskExecutor, device_manager: DeviceManager):
        self.executor = executor
        self.device_manager = device_manager
        # Register to be notified when executor stops
        self.executor.set_on_stop_callback(self._on_executor_stop)
        # Register to be notified when executor actually starts the protocol
        self.executor.set_on_start_callback(self._on_executor_start)

    def _on_executor_stop(self):
        """
        Called when the executor stops. Marks protocol end in the device manager.
        """
        try:
            self.device_manager.mark_protocol_end()
        except Exception as e:
            print(f"Warning: Failed to mark protocol end: {e}")

    def _on_executor_start(self):
        """
        Called when the executor actually starts the protocol. Marks protocol start in the device manager.
        """
        try:
            self.device_manager.mark_protocol_start(self.session_path)
        except Exception as e:
            print(f"Warning: Failed to mark protocol start: {e}")

    @staticmethod
    def make_new_session(subject: str, experiment_name: str, device: str) -> Path:
        """
        Create a new session directory for the given subject, experiment, and device.
        Returns the session path.
        """
        root = lys_subjects_dir() / subject / device / experiment_name
        next_n = get_next_session_number(root)
        session_path = create_session_path(subject, experiment_name, device, next_n)
        return session_path

    def start(self, session_path: Path):
        """
        Store the session path and start the task executor.
        Device manager initialisation happens automatically via callback when protocol actually starts.
        """
        self.session_path = session_path
        self.executor.start(session_path)

    def stop(self):
        """
        Stop the task executor. Device manager finalisation happens automatically
        via the registered callback when the executor stops.
        """
        # For now, TaskExecutors don't have a standard stop() method
        # This could be added in the future for programmatic stopping
        pass 