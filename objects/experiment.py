from dataclasses import dataclass
from typing import List
from lys.objects.session import Session

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
    paths = get_session_paths(experiment_name, scanner_name)
    sessions = [create_session(p) for p in paths]
    return Experiment(experiment_name, scanner_name, sessions)

