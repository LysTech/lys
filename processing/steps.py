from abc import ABC, abstractmethod
import numpy as np
from scipy import signal
from lys.objects import Session


class ProcessingStep(ABC):
    """
    Abstract base class for processing steps that operate on Session objects.
    
    Processing steps are responsible for modifying the session.processed_data attribute
    using information from the Session object. They should be stateless and focused
    on a single transformation.
    
    IMPORTANT: This class mutates Session objects in-place for memory efficiency.
    Only session.processed_data and session.metadata should be modified.
    """
    
    def process(self, session: Session) -> None:
        """
        Template method that enforces metadata recording.
        
        This method MUTATES the session object in-place. The contract is:
        1. Only modifies session.processed_data and session.metadata
        2. Never modifies session.raw_data, session.physio_data, or other immutable fields
        3. Is idempotent (can be called multiple times safely)
        4. Automatically records processing information in session.metadata
        
        Args:
            session: The session to process (modified in-place)
        """
        self._do_process(session)
        self._record_processing_step(session)
    
    @abstractmethod
    def _do_process(self, session: Session) -> None:
        """
        Actual processing logic - implement this method in subclasses.
        
        This method should contain the actual processing logic that modifies
        session.processed_data. The framework handles metadata recording.
        
        Args:
            session: The session to process (modified in-place)
        """
        pass
    
    def _record_processing_step(self, session: Session, **kwargs) -> None:
        """
        Record processing information in session metadata.
        
        Args:
            session: The session to update
            **kwargs: Key-value pairs to store in metadata
        """
        if 'processing_steps' not in session.metadata:
            session.metadata['processing_steps'] = []
        
        step_info = {
            'step_name': str(self),
            'timestamp': np.datetime64('now'),
            **kwargs
        }
        session.metadata['processing_steps'].append(step_info)
    
    def __str__(self) -> str:
        """Return a string representation of the processing step."""
        return f"{self.__class__.__name__}"


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