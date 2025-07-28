import pytest
import os
from pathlib import Path
from lys.utils.paths import check_file_exists, lys_data_dir, extract_patient_from_path, get_experiment_name_from_path, get_session_name_from_path
from lys.utils.paths import get_session_paths

def test_check_file_exists():
    # Test with existing file
    check_file_exists(os.getcwd())
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError, match="File not found: /this/is/dummy/path"):
        check_file_exists("/this/is/dummy/path")


def test_lys_data_dir():
    lys_data_dir()

def test_get_session_paths(tmp_path, monkeypatch):
    # Patch lys_subjects_dir to return tmp_path
    monkeypatch.setattr("lys.utils.paths.lys_subjects_dir", lambda: tmp_path)
    # Create subjects and sessions
    subjects = ['P01', 'P02']
    scanner = 'nirs'
    experiment = 'exp1'
    expected_paths = []
    for subj in subjects:
        session_names = ['session1', 'session2']
        for sess in session_names:
            session_path = tmp_path / subj / scanner / experiment / sess
            session_path.mkdir(parents=True)
            expected_paths.append(session_path)
    # Add a non-matching subject
    (tmp_path / 'not_a_subject').mkdir()
    # Add a non-matching session folder
    (tmp_path / 'P01' / scanner / experiment / 'not_a_session').mkdir(parents=True)
    # Call get_session_paths
    found_paths = get_session_paths(experiment, scanner)
    # Should match expected session paths
    assert set(found_paths) == set(expected_paths)
    for p in found_paths:
        assert isinstance(p, Path)

def test_extract_patient_from_path():
    # Valid cases
    assert extract_patient_from_path(Path("/Users/thomasrialan/Documents/code/Geometric-Eigenmodes/data/P03/nirs/fnirs_8classes/session-01")) == "P03"
    assert extract_patient_from_path(Path("P12/nirs/experiment/session-02")) == "P12"
    assert extract_patient_from_path(Path("/data/P123/other")) == "P123"
    assert extract_patient_from_path(Path("/P1/")) == "P1"
    # Invalid cases
    with pytest.raises(ValueError):
        extract_patient_from_path(Path("/Users/foo/Documents/data/03/nirs/session-01"))
    with pytest.raises(ValueError):
        extract_patient_from_path(Path("/Users/foo/Documents/data/Patient03/nirs/session-01"))
    with pytest.raises(ValueError):
        extract_patient_from_path(Path("/data/PP03/"))

def test_get_experiment_name_from_path():
    # Valid cases
    assert get_experiment_name_from_path(Path("/P03/nirs/exp1/session-01")) == "exp1"
    assert get_experiment_name_from_path(Path("/foo/P12/nirs/experiment/session-02")) == "experiment"
    assert get_experiment_name_from_path(Path("P123/nirs/expA/session-03")) == "expA"
    # Invalid cases
    with pytest.raises(ValueError):
        get_experiment_name_from_path(Path("/foo/bar/P03/nirs"))
    with pytest.raises(ValueError):
        get_experiment_name_from_path(Path("/foo/bar/03/nirs/exp1"))

def test_get_session_name_from_path():
    # Valid cases
    assert get_session_name_from_path(Path("/P03/nirs/exp1/session-01")) == "session-01"
    assert get_session_name_from_path(Path("/foo/P12/nirs/experiment/session-02")) == "session-02"
    assert get_session_name_from_path(Path("P123/nirs/expA/session-03")) == "session-03"
    # Invalid cases
    with pytest.raises(ValueError):
        get_session_name_from_path(Path("/foo/bar/P03/nirs/exp1"))
    with pytest.raises(ValueError):
        get_session_name_from_path(Path("/foo/bar/03/nirs/exp1/session-01"))

def test_create_session_path(monkeypatch, tmp_path):
    monkeypatch.setenv('LYS_DATA_DIR', str(tmp_path))
    from lys.utils.paths import create_session_path
    subject = 'P03'
    experiment_name = 'fnirs_8classes'
    device = 'nirs'
    session_int = 5
    expected = tmp_path / "subjects" / subject / device / experiment_name / f"session-{session_int}"
    result = create_session_path(subject, experiment_name, device, session_int)
    assert result == expected
