from lys.abstract_interfaces.device_manager import DeviceManager
from pathlib import Path
import json
import socket
import struct
import time

class Flow2DeviceConnectionError(Exception):
    """Raised when unable to connect to Flow2 acquisition software (kortex)."""
    pass

class Flow2DeviceManager(DeviceManager):
    """
    Device manager for the Kernel Flow2 device. Handles device-specific initialisation.
    Uses the Kernel Tasks SDK to send events to the Flow2 acquisition system (kortex).
    
    Note: This tests connection to the kortex acquisition software, not the physical Flow2 device.
    Kortex handles the physical device connection and will queue events appropriately.
    
    Always fails fast if kortex is not running.
    """
    def __init__(self, acquisition_host: str = 'localhost', port: int = 6767):
        self.acquisition_host = acquisition_host
        self.port = port
        self.event_id = 0

    def test_kortex_connection(self) -> bool:
        """
        Test if we can connect to the kortex acquisition software.
        Returns True if kortex is running and accepting connections, False otherwise.
        
        Note: This does NOT test if the physical Flow2 device is connected.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 second timeout
            sock.connect((self.acquisition_host, self.port))
            sock.close()
            return True
        except Exception:
            return False

    def mark_protocol_start(self, session_path: Path):
        """
        Send a start_experiment event to the kortex acquisition software via TCP.
        This follows Kernel Tasks SDK conventions where the first event must be "start_experiment".
        
        The absolute timestamp is stored in the "value" field since kortex converts the 
        "timestamp" field to relative time when storing in SNIRF, but preserves "value" as-is.
        
        Raises Flow2DeviceConnectionError if kortex is not running.
        Note: Does not require the physical Flow2 device to be connected.
        """
        if not self.test_kortex_connection():
            raise Flow2DeviceConnectionError(
                f"Cannot connect to kortex acquisition software at {self.acquisition_host}:{self.port}. "
                "Ensure kortex is installed and running. Check with: kortex --version"
            )
        
        # Use current timestamp as value so we can retrieve absolute time from SNIRF
        absolute_timestamp = time.time()
        self._send_event("start_experiment", str(absolute_timestamp))

    def mark_protocol_end(self):
        """
        Send an end_experiment event to the kortex acquisition software via TCP.
        This follows Kernel Tasks SDK conventions where the last event must be "end_experiment".
        
        The absolute timestamp is stored in the "value" field since kortex converts the 
        "timestamp" field to relative time when storing in SNIRF, but preserves "value" as-is.
        
        Raises Flow2DeviceConnectionError if kortex is not running.
        Note: Does not require the physical Flow2 device to be connected.
        """
        if not self.test_kortex_connection():
            raise Flow2DeviceConnectionError(
                f"Cannot connect to kortex acquisition software at {self.acquisition_host}:{self.port}. "
                "Ensure kortex is still running to properly end the protocol."
            )
        
        # Use current timestamp as value so we can retrieve absolute time from SNIRF
        absolute_timestamp = time.time()
        self._send_event("end_experiment", str(absolute_timestamp))

    def _send_event(self, event_name: str, event_value: str):
        """
        Send an event to kortex using the Kernel Tasks SDK protocol.
        Follows the SDK conventions documented at: https://docs.kernel.com/docs/kernel-tasks-sdk
        Kortex will handle coordination with the physical Flow2 device.
        
        Protocol:
        1. Send 4-byte big-endian integer with JSON payload size
        2. Send JSON payload with id, timestamp (microseconds), event, value
        
        Event names should follow SDK conventions:
        - start_experiment, end_experiment for context setting
        - event_* for instantaneous events
        - Other names are treated as metadata
        """
        try:
            # Create socket and connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)  # 5 second timeout
            sock.connect((self.acquisition_host, self.port))
            
            # Increment event ID
            self.event_id += 1
            
            # Create event payload
            timestamp_microseconds = int(time.time() * 1e6)
            data_to_send = {
                "id": self.event_id,
                "timestamp": timestamp_microseconds,
                "event": event_name,
                "value": event_value,
            }
            
            # Encode JSON to bytes
            event_json = json.dumps(data_to_send).encode("utf-8")
            
            # Create message: 4-byte size (big-endian) + JSON bytes
            size_header = struct.pack("!I", len(event_json))
            message = size_header + event_json
            
            # Send message
            sock.sendall(message)
            sock.close()
            
            print(f"✓ Sent Kernel SDK event to kortex: {event_name} = {event_value} at {timestamp_microseconds}")
            
        except Exception as e:
            raise Flow2DeviceConnectionError(
                f"Failed to send event to kortex at {self.acquisition_host}:{self.port} - {e}"
            ) 