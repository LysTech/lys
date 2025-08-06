from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence
from pathlib import Path

import numpy as np

from lys.objects.jacobian import load_jacobians_from_session_dir
from lys.objects.patient import Patient
from lys.objects.protocol import Protocol
from lys.objects.jacobian import Jacobian
from lys.utils.paths import extract_patient_from_path


"""
The create_session and create_session_with_common_channels functions are called by 
experiment.py's create_experiment and create_experiment_with_common_channels. Common channels
means we use the `raw_channel_data_common_channels.npz` file created by preprocessing.py's
preprocess_experiment_with_common_channels.
"""


@dataclass
class Session:
    """
    Represents a single experimental session, containing all related data.

    Attributes:
        patient (Patient): The patient associated with the session.
        protocol (Protocol): The experimental protocol.
        raw_data (Dict[str, np.ndarray]): The raw scanner data as a dictionary with wavelength keys (e.g., 'wl1', 'wl2').
        jacobians (Optional[Sequence[Jacobian]]): The Jacobians (plural for multi-wavelength). Defaults to None.
        physio_data (Optional[Dict[str, np.ndarray]]): Physiological data as a dictionary. Defaults to None.
        processed_data (Optional[Dict[str, np.ndarray]]): Processed data as a dictionary with wavelength keys (e.g., 'wl1', 'wl2'). 
            If None, defaults to raw_data.
        metadata (Dict[str, Any]): A dictionary for metadata, including processing steps. 
            Defaults to {"processing_steps": []}.
    """
    patient: Patient
    protocol: Protocol
    raw_data: Dict[str, np.ndarray]
    jacobians: Optional[Sequence[Jacobian]] = None
    physio_data: Optional[Dict[str, np.ndarray]] = None
    processed_data: Dict[str, Any] = None
    metadata: Dict[str, Any] = field(default_factory=lambda: {"processing_steps": []})

    def __post_init__(self):
        """
        Post-initialization method to set default values that depend on other fields.
        """
        if self.processed_data is None:
            self.processed_data = {
                key: value.copy() for key, value in self.raw_data.items()
            }


def create_session(path: Path) -> Session:
    """Load a Session object from a session directory using standard channels.
    
    Loads all required and optional data for a session, including:
      - Patient and protocol information
      - Jacobians
      - Raw data from 'raw_channel_data.npz'
      - Processed data from 'processed_channel_data.npz' (optional)
      - Physiological data from 'physio_data.npz' (optional)
    
    Args:
        path: Path to the session directory
        
    Returns:
        A fully constructed Session object
        
    Raises:
        FileNotFoundError: If required data files are missing
    """
    patient = Patient.from_name(extract_patient_from_path(path))
    protocol = Protocol.from_session_path(path)
    jacobians = load_jacobians_from_session_dir(path)
    raw_npz, processed_npz, physio_npz = _load_numpy_formatted_data(path, "raw_channel_data.npz")
    return _make_session(patient, protocol, jacobians, raw_npz, processed_npz, physio_npz)


def create_session_with_common_channels(path: Path) -> Session:
    """Load a Session object from a session directory using common channels.
    
    Loads all required and optional data for a session, including:
      - Patient and protocol information
      - Jacobians
      - Raw data from 'raw_channel_data_common_channels.npz'
      - Processed data from 'processed_channel_data.npz' (optional)
      - Physiological data from 'physio_data.npz' (optional)
    
    Args:
        path: Path to the session directory
        
    Returns:
        A fully constructed Session object
        
    Raises:
        FileNotFoundError: If required data files are missing
    """
    patient = Patient.from_name(extract_patient_from_path(path))
    protocol = Protocol.from_session_path(path)
    jacobians = load_jacobians_from_session_dir(path)
    raw_npz, processed_npz, physio_npz = _load_numpy_formatted_data(path, "raw_channel_data_common_channels.npz")
    return _make_session(patient, protocol, jacobians, raw_npz, processed_npz, physio_npz)
    

def _load_numpy_formatted_data(path: Path, raw_data_filename: str) -> tuple[dict, Optional[dict], Optional[dict]]:
    """Load all numpy-formatted data files from session directory.
    
    Args:
        path: Path to session directory
        raw_data_filename: Name of the raw data file to load
        
    Returns:
        Tuple of (raw_npz, processed_npz, physio_npz)
    """
    raw_npz = _load_npz_or_error(path, raw_data_filename, required=True)
    processed_npz = _load_npz_or_error(path, "processed_channel_data.npz", required=False)
    physio_npz = _load_npz_or_error(path, "physio_data.npz", required=False)
    return raw_npz, processed_npz, physio_npz


def _make_session(patient: Patient, 
                  protocol: Protocol,
                  jacobians: Optional[Sequence[Jacobian]],
                  raw_npz: dict,
                  processed_npz: Optional[dict],
                  physio_npz: Optional[dict]) -> Session:
    """Create a Session object from loaded components.
    
    Args:
        patient: Patient object
        protocol: Protocol object
        jacobians: Jacobians for the session
        raw_npz: Raw data dictionary
        processed_npz: Processed data dictionary (optional)
        physio_npz: Physiological data dictionary (optional)
        
    Returns:
        Constructed Session object
    """
    session_kwargs = {
        'patient': patient,
        'protocol': protocol,
        'raw_data': raw_npz,
        'jacobians': jacobians,
        'physio_data': physio_npz if physio_npz is not None else None,
    }
    
    # Add processed_data only if it exists
    if processed_npz is not None:
        session_kwargs['processed_data'] = processed_npz
    
    # Merge processed metadata with default metadata
    if processed_npz is not None and "metadata" in processed_npz:
        session_kwargs['metadata'] = processed_npz["metadata"]

    return Session(**session_kwargs)


def _load_npz_or_error(path: Path, filename: str, required: bool = True) -> Optional[dict]:
    file_path = path / filename
    if not file_path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file '{filename}' not found in {path}. "
                f"Preprocessing steps have not been performed for this session. "
                f"Please create the necessary .npz files."
            )
        else:
            return None
    return dict(np.load(file_path, allow_pickle=True))

