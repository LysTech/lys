from dataclasses import dataclass
from typing import List
from lys.objects.session import create_session, create_session_with_common_channels, Session
from lys.utils.paths import get_session_paths

@dataclass
class Experiment:
    name : str
    scanner: str
    sessions : List[Session]

    def filter_by_subjects(self, patient_ids: List[str]):
        """Create new Experiment with only specified subjects."""
        filtered_sessions = [session for session in self.sessions
                           if session.patient.name in patient_ids]
        # add some checks that, e.g. all subjects exist
        return Experiment(f"{self.name}_filtered", self.scanner, filtered_sessions)


def create_experiment(experiment_name, scanner_name):
    """Create an experiment with standard channel configuration: each session has the data from each channel that was
        recorded in this session (with the Flow2 this can change session to session if you don't use the raw TD-fNIRS
        data, which we are currently not. see: https://docs.kernel.com/docs/data-export-pipelines)
    
    Args:
        experiment_name: Name of the experiment
        scanner_name: Name of the scanner device
        
    Returns:
        Experiment instance with sessions using standard channels
    """
    paths = get_session_paths(experiment_name, scanner_name)
    sessions = [create_session(p) for p in paths]
    return Experiment(experiment_name, scanner_name, sessions)


def create_experiment_with_common_channels(experiment_name, scanner_name):
    """Create an experiment with common channel configuration across sessions.
    
    Args:
        experiment_name: Name of the experiment
        scanner_name: Name of the scanner device
        
    Returns:
        Experiment instance with sessions using common channels
    """
    paths = get_session_paths(experiment_name, scanner_name)
    sessions = [create_session_with_common_channels(p) for p in paths]
    return Experiment(experiment_name, scanner_name, sessions)

