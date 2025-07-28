from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class ISessionAdapter(ABC):
    """ Preprocessing: turning raw files of fucked format into useful .npz files that can be loaded (for processing!)
        We distinguish this from processing, which does stuff like bandpass filtering, etc.

        In charge of turning .snirf files and such into useful / quick to load numpy arrays for example.

        The idea is that we can have a bunch of different adapters for different devices, and we can just
        add a new adapter for a new device."""
    @abstractmethod
    def can_handle(self, session_path: Path) -> bool:
        """ Check if this adapter can process the given session by looking
            at the file extensions of stuff in the session. """
        raise NotImplementedError
    
    @abstractmethod
    def extract_data(self, session_path: Path) -> dict:
        """Extract data from device-specific formats and return as a dictionary.
        The dictionary keys will become the npz file keys."""
        raise NotImplementedError
    
    def process(self, session_path: Path):
        """
        Template method that enforces npz file saving.
        Subclasses must implement extract_data() but cannot override this method.
        """
        data = self.extract_data(session_path)
        output_path = session_path / 'raw_channel_data.npz'
        np.savez(output_path, **data)

