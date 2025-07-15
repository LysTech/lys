from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import snirf

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
            Flow2MomentsSessionAdapter(),
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


class Flow2MomentsSessionAdapter(ISessionAdapter):
    """
    Adapter for Kernel Flow2 .snirf files from the 'Moments' pipeline.
    
    This adapter extracts intensity, mean time of flight, and variance from the
    SNIRF file.
    """
    def can_handle(self, session_path: Path) -> bool:
        """Check if this session contains a .snirf file."""
        assert session_path.is_dir()
        return any(f.name.endswith("_MOMENTS.snirf") for f in session_path.iterdir() if f.is_file())

    def extract_data(self, session_path: Path) -> dict:
        """
        Extracts data from a .snirf file and returns a dictionary with:
        - 'data': np.ndarray of shape (n_timepoints, n_channels, n_wavelengths, 3)
        - 'channels': list of (source, detector) tuples
        - 'wavelengths': list of wavelength indices
        - 'moment_names': ['amplitude', 'mean_time_of_flight', 'variance']
        - 'time': time vector (n_timepoints,)
        """
        snirf_path = next(session_path.glob('*_MOMENTS.snirf'))
        snirf_data = snirf.Snirf(str(snirf_path), 'r')
        nirs_data_block = snirf_data.nirs[0].data[0]
        mlist = nirs_data_block.measurementList
        n_timepoints = nirs_data_block.dataTimeSeries.shape[0]

        # 1. Build unique channel and wavelength lists
        channel_tuples = sorted(set((m.sourceIndex, m.detectorIndex) for m in mlist))
        wavelength_indices = sorted(set(getattr(m, 'wavelengthIndex', 1) for m in mlist))
        channel_index = {ch: i for i, ch in enumerate(channel_tuples)}
        wavelength_index = {w: i for i, w in enumerate(wavelength_indices)}

        # 2. Map dataTypeIndex/dataUnit to moment index
        moment_map = {
            (2, ''): 0,        # amplitude
            (1, 'ps'): 1,      # mean time of flight
            (3, 'ps^2'): 2     # variance
        }
        moment_names = ['amplitude', 'mean_time_of_flight', 'variance']

        n_channels = len(channel_tuples)
        n_wavelengths = len(wavelength_indices)
        n_moments = 3

        # 3. Allocate output array
        data = np.full((n_timepoints, n_channels, n_wavelengths, n_moments), np.nan)

        # 4. Fill array
        for col, m in enumerate(mlist):
            ch = (m.sourceIndex, m.detectorIndex)
            w = getattr(m, 'wavelengthIndex', 1)
            dtype_idx = getattr(m, 'dataTypeIndex', None)
            dunit = getattr(m, 'dataUnit', '')
            if isinstance(dtype_idx, int):
                moment_idx = moment_map.get((dtype_idx, dunit))
                if moment_idx is not None:
                    cidx = channel_index[ch]
                    widx = wavelength_index[w]
                    data[:, cidx, widx, moment_idx] = nirs_data_block.dataTimeSeries[:, col]

        return {
            'data': data,
            'channels': channel_tuples,
            'wavelengths': wavelength_indices,
            'moment_names': moment_names,
            'time': nirs_data_block.time
        }
