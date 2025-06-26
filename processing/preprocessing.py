from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from lys.interfaces import ISessionAdapter

""" 
Preprocessing: turning raw files of fucked format into useful .npz files that can be loaded (for processing!)
We distinguish this from processing, which does stuff like bandpass filtering, etc.

In charge of turning .snirf files and such into useful / quick to load numpy arrays for example.

The idea is that we can have a bunch of different adapters for different devices, and we can just
add a new adapter for a new device.

How to use this code:

# Process a single session:
RawSessionPreProcessor.preprocess(session_path)

# Process a whole experiment:
from lys.objects.session import get_session_paths
paths = get_session_paths("experiment_name", "scanner_name")
for path in paths:
    RawSessionPreProcessor.preprocess(path)
"""

#TODO: BettinaSessionAdapter is a terrible name?
#TODO: currently we don't do optodes stuff for BettinaSessionAdapter
#TODO: implement the Flow2SessionAdapter
#TODO: generally figure out the optodes thing, see comment in Notion Architecture page
#TODO: RawSessionPreProcessor is a PREprocessor so it should be renamed

class RawSessionPreProcessor:
    def __init__(self, session_path: Path):
        self.session_path = Path(session_path)
        self.session_adapter = self._select_strategy()
    
    @classmethod
    def preprocess(cls, session_path: Path):
        """Preprocess a session using the appropriate strategy based on session content."""
        processor = cls(session_path)
        return processor.session_adapter.process(processor.session_path)
    
    def _select_strategy(self) -> 'ISessionAdapter':
        """
        Select the appropriate strategy (adapter) based on session content.
        Uses the Strategy pattern to choose the best adapter for the job.
        """
        available_strategies = [
            BettinaSessionAdapter(),
            # Add more adapters here as they're implemented
        ]
        
        for strategy in available_strategies:
            if strategy.can_handle(self.session_path):
                return strategy
        
        raise ValueError(f"No suitable adapter found for session at {self.session_path}")

class BettinaSessionAdapter(ISessionAdapter):
    def can_handle(self, session_path: Path) -> bool:
        """Check if this session contains the required .w;1 and .wl2 files for Bettina processing."""
        assert session_path.is_dir()
        files = [f.name for f in session_path.iterdir() if f.is_file()]
        has_wl1 = any(file.endswith(".wl1") for file in files)
        has_wl2 = any(file.endswith(".wl2") for file in files)
        return has_wl1 and has_wl2

    def extract_data(self, session_path: Path) -> dict:
        """Extract data from device-specific formats and return as a dictionary.
        The dictionary keys will become the npz file keys."""
        wl1_path = next(session_path.glob('*.wl1'))
        wl2_path = next(session_path.glob('*.wl2'))
        wl1 = np.loadtxt(wl1_path)
        wl2 = np.loadtxt(wl2_path)
        return {'wl1': wl1, 'wl2': wl2}