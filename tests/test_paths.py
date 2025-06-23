import pytest
import os
from pathlib import Path
from lys.utils.paths import check_file_exists, lys_data_dir
from lys.objects.session import get_session_paths

def test_check_file_exists():
    # Test with existing file
    check_file_exists(os.getcwd())
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError, match="File not found: /this/is/dummy/path"):
        check_file_exists("/this/is/dummy/path")


def test_lys_data_dir():
    lys_data_dir()

def test_get_session_paths(tmp_path, monkeypatch):
    # Setup mock data dir
    data_dir = tmp_path
    monkeypatch.setenv('LYS_DATA_DIR', str(data_dir))
    # Create subjects and sessions
    subjects = ['P01', 'P02']
    scanner = 'nirs'
    experiment = 'exp1'
    expected_paths = []
    for subj in subjects:
        session_names = ['session1', 'session2']
        for sess in session_names:
            session_path = data_dir / subj / scanner / experiment / sess
            session_path.mkdir(parents=True)
            expected_paths.append(session_path)
    # Add a non-matching subject
    (data_dir / 'not_a_subject').mkdir()
    # Add a non-matching session folder
    (data_dir / 'P01' / scanner / experiment / 'not_a_session').mkdir(parents=True)
    # Call get_session_paths
    found_paths = get_session_paths(experiment, scanner)
    # Should match expected session paths
    assert set(found_paths) == set(expected_paths)
    for p in found_paths:
        assert isinstance(p, Path)