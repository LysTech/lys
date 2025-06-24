from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np

from lys.objects.jacobian import load_jacobians_from_session_dir
from lys.objects import Patient, Protocol, Jacobian
from lys.utils.paths import lys_data_dir, extract_patient_from_path

#TODO: add a post-init method that checks time-alignment of everything
#TODO: refactor using pull-up or something like that for generalisation

@dataclass
class Session:
    """
    Represents a single experimental session, containing all related data.

    Attributes:
        patient (Patient): The patient associated with the session.
        protocol (Protocol): The experimental protocol.
        raw_data (np.ndarray): The raw scanner data, shape (n_channels, n_timepoints).
        jacobians (Optional[List[Jacobian]]): The Jacobians (plural for multi-wavelength). Defaults to None.
        physio_data (Optional[np.ndarray]): Physiological data, shape (m_channels, n_timepoints). Defaults to None.
        processed_data (np.ndarray): Processed data, initialized as a copy of raw_data.
        metadata (Dict[str, Any]): A dictionary for metadata, including processing steps.
    """
    patient: Patient
    protocol: Protocol
    raw_data: np.ndarray
    jacobians: Optional[List[Jacobian]] = None
    physio_data: Optional[np.ndarray] = None
    processed_data: np.ndarray = field(init=False, repr=False) #Maybe this isn't required and defaults to None?
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        - should we have this field(init=False) thing?
        - is this weirdly coupled to data types like npz?
        worth revisiting.

    Loads a Session object from a given session directory.

    This function loads all required and optional data for a session, including:
      - Patient and protocol information
      - Jacobians
      - Raw data (from 'raw_data.npz', must contain a 'data' array)
      - Processed data (from 'processed_data.npz', must contain a 'data' array and may contain 'metadata')
      - Physiological data (from 'physio_data.npz', optional, must contain a 'data' array if present)

    The processed data's metadata (if present) is attached to the Session's metadata field.

    Parameters:
        path (Path): Path to the session directory containing the .npz files and protocol.

    Returns:
        Session: A fully constructed Session object with all loaded data.

    Raises:
        FileNotFoundError: If 'raw_data.npz' or 'processed_data.npz' is missing, with a message
            indicating that preprocessing steps are required.
    """
    patient = Patient.from_name(extract_patient_from_path(path))
    protocol = Protocol.from_session_path(path)
    jacobians = load_jacobians_from_session_dir(path)
    raw_npz = _load_npz_or_error(path, "raw_channel_data.npz", required=True)
    processed_npz = _load_npz_or_error(path, "processed_channel_data.npz", required=True)
    physio_npz = _load_npz_or_error(path, "physio_data.npz", required=False)

    raw_data = raw_npz["data"] if raw_npz is not None else None
    processed_data = processed_npz["data"] if processed_npz is not None else None
    processed_metadata = processed_npz.get("metadata", {}) if processed_npz is not None else {}
    physio_data = physio_npz["data"] if physio_npz is not None else None

    session = Session(
        patient=patient,
        protocol=protocol,
        raw_data=raw_data,
        jacobians=jacobians,
        physio_data=physio_data,
        metadata=processed_metadata
    )
    # processed_data is a field(init=False), so set it after construction
    session.processed_data = processed_data
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
    