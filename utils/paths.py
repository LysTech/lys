import os
from pathlib import Path

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

