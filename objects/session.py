from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np

from lys.objects.jacobian import Jacobian
from lys.objects.patient import Patient
from lys.objects.protocol import Protocol
from lys.utils.paths import lys_data_dir

#TODO: add a post-init method that checks time-alignment of everything

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
    processed_data: np.ndarray = field(init=False, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_session(path):
    pass

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