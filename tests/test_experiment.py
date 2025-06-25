from lys.objects.experiment import Experiment
from lys.objects.session import Session
import numpy as np

class DummyPatient:
    def __init__(self, name):
        self.name = name

class DummyProtocol:
    pass

def make_dummy_session(subject_id):
    patient = DummyPatient(subject_id)
    protocol = DummyProtocol()
    # raw_data is required, so pass a dummy dict with wavelength keys and numpy arrays
    raw_data = {'wl1': np.zeros((1, 1))}
    return Session(patient=patient, protocol=protocol, raw_data=raw_data)

def test_filter_by_subjects():
    sessions = [make_dummy_session(sid) for sid in ["P01", "P02", "P03"]]
    exp = Experiment("experiment_name", "scanner_name", sessions)
    filtered = exp.filter_by_subjects(["P02"])
    assert len(filtered.sessions) == 1
    assert filtered.sessions[0].patient.name == "P02" 