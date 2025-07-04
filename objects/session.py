from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Sequence
from pathlib import Path

import numpy as np

from lys.objects.jacobian import load_jacobians_from_session_dir
from lys.objects.patient import Patient
from lys.objects.protocol import Protocol
from lys.objects.jacobian import Jacobian
from lys.utils.paths import lys_data_dir, extract_patient_from_path

#TODO: add a post-init method that checks time-alignment of everything
#TODO: refactor using pull-up or something like that for generalisation
#TODO: think: some experiments will have 2 wavelengths, some might have 6 -> is it ok that raw_data is dict (npz) as solution to this?

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
    processed_data: Any = None #TODO: perhaps clarify the type, not super sure...
    metadata: Dict[str, Any] = field(default_factory=lambda: {"processing_steps": []})

    def __post_init__(self):
        """
        Post-initialization method to set default values that depend on other fields.
        """
        if self.processed_data is None:
            self.processed_data = {
                key: value.copy() for key, value in self.raw_data.items()
            }


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


def create_session(path: Path):
    """
    #TODO: revisit?
    notes: i'm not sure this is good architecture/code principles:
        - is it bad that this coupled to data types like npz?
        worth revisiting.

    Loads a Session object from a given session directory.

    This function loads all required and optional data for a session, including:
      - Patient and protocol information
      - Jacobians
      - Raw data (from 'raw_data.npz', must contain a 'data' array)
      - Processed data (from 'processed_data.npz', optional, must contain a 'data' array and may contain 'metadata')
      - Physiological data (from 'physio_data.npz', optional, must contain a 'data' array if present)

    The processed data's metadata (if present) is attached to the Session's metadata field.

    Parameters:
        path (Path): Path to the session directory containing the .npz files and protocol.

    Returns:
        Session: A fully constructed Session object with all loaded data.

    Raises:
        FileNotFoundError: If 'raw_data.npz' is missing, with a message
            indicating that preprocessing steps are required.
    """
    patient = Patient.from_name(extract_patient_from_path(path))
    protocol = Protocol.from_session_path(path)
    jacobians = load_jacobians_from_session_dir(path)
    raw_npz = _load_npz_or_error(path, "raw_channel_data.npz", required=True)
    processed_npz = _load_npz_or_error(path, "processed_channel_data.npz", required=False)
    physio_npz = _load_npz_or_error(path, "physio_data.npz", required=False)

    # raw_data is required, so raw_npz should never be None due to required=True
    assert raw_npz is not None, "raw_npz should never be None when required=True"
    raw_data = raw_npz
    
    # Only pass processed_data if it exists, otherwise let it default to raw_data
    session_kwargs = {
        'patient': patient,
        'protocol': protocol,
        'raw_data': raw_data,
        'jacobians': jacobians,
        'physio_data': physio_npz if physio_npz is not None else None,
    }
    
    # Add processed_data only if it exists
    if processed_npz is not None:
        session_kwargs['processed_data'] = processed_npz
    
    # Merge processed metadata with default metadata
    if processed_npz is not None and "metadata" in processed_npz:
        session_kwargs['metadata'] = processed_npz["metadata"]

    session = Session(**session_kwargs)
    return session


def get_session_paths(experiment_name, scanner_name):
    """
    Returns a list of Path objects for all session folders for the given experiment and scanner.
    Each subject is a subfolder in lys_data_dir() starting with 'P' followed by a number.
    Within each subject, looks for scanner_name/experiment_name/session* folders.
    Only folders matching 'session' in their name are included.
    """
    data_root = lys_data_dir()
    session_paths = []
    for subject_dir in data_root.iterdir():
        if subject_dir.is_dir() and subject_dir.name.startswith('P') and subject_dir.name[1:].isdigit():
            experiment_dir = subject_dir / scanner_name / experiment_name
            if experiment_dir.exists() and experiment_dir.is_dir():
                for session_dir in experiment_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.lower().startswith('session'):
                        session_paths.append(session_dir)
    return session_paths
    