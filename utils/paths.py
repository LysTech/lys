import os
from pathlib import Path
import re

def lys_data_dir():
    try:
        dirs = os.environ['LYS_DATA_DIR']
    except KeyError:
        msg = """ Please make sure LYS_DATA_DIR is in your system environment:
            add: 'export LYS_DATA_DIR=/path/ to your bashrc and source it.'"""
        raise RuntimeError(msg)
    return Path(dirs)


def check_file_exists(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")


def extract_patient_from_path(path: Path) -> str:
    """
    Extracts the patient string (e.g., 'P03') from a Path object by searching for a part matching the pattern 'P' followed by digits.
    Raises ValueError if no such part is found.
    """
    for part in path.parts:
        match = re.fullmatch(r"P\d+", part)
        if match:
            return match.group(0)
    raise ValueError(f"No patient string found in path: {path}")


def get_experiment_name_from_path(path: Path) -> str:
    """
    Given a path, finds the patient string and returns the experiment name, which is two levels beneath the patient directory.
    Raises ValueError if the structure is not found.
    """
    parts = str(path).split("/")
    for i, part in enumerate(parts):
        if re.fullmatch(r"P\d+", part):
            if i + 2 < len(parts):
                return parts[i + 2]
            else:
                break
    raise ValueError(f"Experiment name not found two levels beneath patient in path: {path}")


def get_session_name_from_path(path: Path) -> str:
    """
    Given a path, returns the session name, which is one level beneath the experiment name (three levels beneath patient).
    Raises ValueError if the structure is not found.
    """
    parts = str(path).split("/")
    for i, part in enumerate(parts):
        if re.fullmatch(r"P\d+", part):
            if i + 3 < len(parts):
                return parts[i + 3]
            else:
                break
    raise ValueError(f"Session name not found one level beneath experiment in path: {path}")


def get_subjects_dir() -> Path:
    """
    Returns the Path to the subjects directory within the data directory.
    """
    return lys_data_dir() / "subjects"


def create_session_path(subject: str, experiment_name: str, device: str, session_int: int) -> Path:
    """
    Create a session path rooted at get_subjects_dir(), e.g. {LYS_DATA_DIR}/subjects/{subject}/{experiment_name}/{date}/{session_id}
    """
    root = get_subjects_dir()
    return root / subject / device / experiment_name / f"session-{session_int}"


def get_session_paths(experiment_name, scanner_name):
    """
    Returns a list of Path objects for all session folders for the given experiment and scanner.
    Each subject is a subfolder in get_subjects_dir() starting with 'P' followed by a number.
    Within each subject, looks for scanner_name/experiment_name/session* folders.
    Only folders matching 'session' in their name are included.
    """
    data_root = get_subjects_dir()
    session_paths = []
    for subject_dir in data_root.iterdir():
        if subject_dir.is_dir():
            experiment_dir = subject_dir / scanner_name / experiment_name
            if experiment_dir.exists() and experiment_dir.is_dir():
                for session_dir in experiment_dir.iterdir():
                    if session_dir.is_dir() and session_dir.name.lower().startswith('session'):
                        session_paths.append(session_dir)
    return session_paths


def get_next_session_number(root: Path) -> int:
    """
    Given a root path, find the next available session integer (N) for session-N folders.
    """
    nums = []
    for p in root.iterdir():
        if p.is_dir():
            m = re.match(r'session-(\d+)$', p.name)
            if m:
                nums.append(int(m.group(1)))
    if nums:
        return max(nums) + 1
    else:
        return 1


def get_audio_assets_path() -> Path:
    """
    Returns the Path to the audio assets directory within the data directory. Raises FileNotFoundError if it does not exist.
    """
    path = lys_data_dir() / "assets" / "audio"
    if not path.exists():
        raise FileNotFoundError(f"Audio assets directory does not exist: {path}")
    return path

