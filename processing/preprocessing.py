from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import snirf
from datetime import datetime, timezone

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

    nb that idk if this works for the "Hb Moments" pipeline

    This extracts a (num_timepoints, num_channels, num_wavelengths, num_moments) array.
    where num_moments is 3, and the moments are amplitude, mean_time_of_flight, and variance.
    
    """
    #TODO: This doesn't get the EEG data out, but it should!
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
        - 'time': time vector (n_timepoints,) of absolute unix timestamps
        """
        snirf_path = next(session_path.glob('*_MOMENTS.snirf'))
        snirf_data = snirf.Snirf(str(snirf_path), 'r')
        nirs_data_block = snirf_data.nirs[0].data[0]
        mlist = nirs_data_block.measurementList
        n_timepoints = nirs_data_block.dataTimeSeries.shape[0]

        # Extract absolute start time from metaDataTags
        meta = snirf_data.nirs[0].metaDataTags
        assert hasattr(meta, 'MeasurementDate'), "metaDataTags must have 'MeasurementDate'"
        assert hasattr(meta, 'MeasurementTime'), "metaDataTags must have 'MeasurementTime'"
        date = getattr(meta, 'MeasurementDate')
        time = getattr(meta, 'MeasurementTime')
        absolute_start_time_str = f"{date}T{time}"

        #TODO: double check this UTC stuff manually (its not mentioned in the docs)
        # Convert start time to UTC timestamp
        start_datetime = datetime.fromisoformat(absolute_start_time_str)
        # check if timezone-aware, if not, assume UTC
        if start_datetime.tzinfo is None:
            start_datetime = start_datetime.replace(tzinfo=timezone.utc)
        start_timestamp = start_datetime.timestamp()

        # Create absolute time vector
        relative_time_vector = nirs_data_block.time
        absolute_time_vector = relative_time_vector + start_timestamp


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

        # Align data with the protocol start time
        data, absolute_time_vector = self._align_with_protocol(
            data, absolute_time_vector, session_path
        )

        return {
            'data': data,
            'channels': channel_tuples,
            'wavelengths': wavelength_indices,
            'moment_names': moment_names,
            'time': absolute_time_vector,
        }

    def _align_with_protocol(self, data: np.ndarray, time_vector: np.ndarray, session_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Aligns NIRS data with the protocol start time by trimming the beginning."""
        protocol_start_time = self._get_protocol_start_time(session_path)
        
        alignment_index = self._find_alignment_index(time_vector, protocol_start_time)
        
        trimmed_data = data[alignment_index:, ...]
        trimmed_time_vector = time_vector[alignment_index:]
        
        # Assert that the alignment was successful and the new start time is correct.
        if len(trimmed_time_vector) > 0 and len(time_vector) > 1:
            # Calculate sampling frequency from original, untrimmed time vector
            fs = 1 / np.mean(np.diff(time_vector))
            
            actual_start_time = trimmed_time_vector[0]
            time_difference = abs(actual_start_time - protocol_start_time)
            
            # The difference should be less than half a sample period.
            assert time_difference < (1 / fs) / 2, (
                f"Time difference after alignment ({time_difference:.4f}s) is larger than "
                f"half a sample period ({((1 / fs) / 2):.4f}s). Alignment failed."
            )

        return trimmed_data, trimmed_time_vector

    def _get_protocol_start_time(self, session_path: Path) -> float:
        """Finds and returns the protocol's start time from the .jsonl log file."""
        import json
        try:
            protocol_log_path = next(session_path.glob('*.jsonl'))
        except StopIteration:
            raise FileNotFoundError(f"No .jsonl protocol log file found in {session_path}")
            
        with open(protocol_log_path, 'r') as f:
            first_line = f.readline()
        
        try:
            first_event = json.loads(first_line)
            # Assuming the timestamp is a unix timestamp in a 'timestamp' field.
            return float(first_event['timestamp'])
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(
                f"Could not extract timestamp from the first line of {protocol_log_path}. "
                f"Ensure it's a valid JSON object with a 'timestamp' key. Error: {e}"
            )

    def _find_alignment_index(self, nirs_time_vector: np.ndarray, protocol_start_time: float) -> int:
        """Finds the index in the NIRS time vector corresponding to the protocol start."""
        nirs_start_time = nirs_time_vector[0]
        
        if nirs_start_time > protocol_start_time:
            raise ValueError(
                f"NIRS data recording started at {nirs_start_time} "
                f"which is AFTER the protocol started at {protocol_start_time}. "
                "Cannot align data."
            )
            
        # Find the index of the NIRS time point closest to the protocol start time
        alignment_index = np.argmin(np.abs(nirs_time_vector - protocol_start_time))
        
        return int(alignment_index)


if __name__=="__main__":
    import os
    from lys.utils import lys_data_dir
    session_path = Path(os.path.join(lys_data_dir(), "thomas/flow2/perceived_speech/session-01"))
    processor = Flow2MomentsSessionAdapter()
    data = processor.extract_data(session_path)