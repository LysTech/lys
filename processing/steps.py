import numpy as np
from scipy import signal

from lys.objects import Session
from lys.interfaces.processing_step import ProcessingStep

class ZTransform(ProcessingStep):
    def _do_process(self, session: Session) -> None:
        """ this modifies session.processed_data inplace """
        session.processed_data["wl1"] = self.z_transform(session.processed_data["wl1"])
        session.processed_data["wl2"] = self.z_transform(session.processed_data["wl2"])

    def z_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a Z-transform to the data.
        """
        return data



class BandpassFilter(ProcessingStep):
    """
    A processing step that applies a bandpass filter to wavelength data.
    
    Uses a 3rd order Butterworth filter to filter data between the specified
    frequency bounds. Applies the filter to both wl1 and wl2 data channels.
    """
    
    def __init__(self, upper_bound: float, lower_bound: float):
        """
        Initialize the bandpass filter.
        
        Args:
            upper_bound: Upper frequency bound in Hz
            lower_bound: Lower frequency bound in Hz
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def _do_process(self, session: Session) -> None:
        """
        Apply bandpass filter to both wl1 and wl2 data channels.
        
        Args:
            session: The session to process (modified in-place)
        """
        session.processed_data["wl1"] = self._apply_filter(session.processed_data["wl1"])
        session.processed_data["wl2"] = self._apply_filter(session.processed_data["wl2"])
    
    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a 3rd order Butterworth bandpass filter to the data.
        
        Args:
            data: Input data array to filter
            
        Returns:
            Filtered data array
        """
        # Design the Butterworth filter
        nyquist = 0.5  # Assuming normalized frequency
        low = self.lower_bound / nyquist
        high = self.upper_bound / nyquist
        
        # Create 3rd order Butterworth bandpass filter
        b, a = signal.butter(3, [low, high], btype='band')
        
        # Apply the filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        
        return filtered_data