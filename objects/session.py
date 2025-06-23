from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np

from lys.objects.jacobian import Jacobian
from lys.objects.patient import Patient
from lys.objects.protocol import Protocol

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
