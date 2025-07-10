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

