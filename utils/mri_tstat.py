import numpy as np
import bvbabel as bv
from pathlib import Path
from typing import Dict, Tuple, Any

from .paths import lys_data_dir


def _parse_task_name(task_name: str) -> str:
    """
    Parse task name by removing brackets, spaces, and '+1' suffix.
    
    Args:
        task_name: Raw task name from SMP header
        
    Returns:
        Cleaned task name
    """
    return task_name.replace('[', '').replace(']', '').replace(' ', '').replace('+1', '')


def _get_smp_file_path(patient_name: str) -> Path:
    """
    Generate the SMP file path for a given patient.
    
    Args:
        patient_name: Patient identifier (e.g., 'P03')
        
    Returns:
        Path to the SMP file containing t-stats
    """
    data_dir = lys_data_dir()
    smp_filename = f"{patient_name}_contrasts_MNI_min_prt_preprocessed_mtc_all_nothreshold.smp"
    return data_dir / patient_name / "func" / "t-stats" / smp_filename


def _read_smp_file(file_path: Path) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Read SMP file and return header and data.
    
    Args:
        file_path: Path to the SMP file
        
    Returns:
        Tuple of (smp_header, tstat_data)
        
    Raises:
        FileNotFoundError: If the SMP file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"SMP file not found: {file_path}")
    
    return bv.smp.read_smp(str(file_path))


def _create_task_map(smp_header: Dict[str, Any], tstat_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Create a mapping from task names to t-stat arrays.
    
    Args:
        smp_header: Header from SMP file
        tstat_data: T-stat data array
        
    Returns:
        Dictionary mapping task names to t-stat arrays
    """
    return {
        _parse_task_name(smp_header['Map'][ix]['Name']): tstat_data[:, ix]
        for ix in range(smp_header['Nr maps'])
    }


def get_mri_tstats(patient_name: str, task: str) -> np.ndarray:
    """
    Get MRI t-stats for a specific patient and task.
    
    Args:
        patient_name: Patient identifier (e.g., 'P03')
        task: Task name (e.g., 'MT', 'SN')
        
    Returns:
        Array of t-stats for the specified task
        
    Raises:
        FileNotFoundError: If the SMP file doesn't exist
        KeyError: If the task is not found in the SMP file
    """
    smp_file_path = _get_smp_file_path(patient_name)
    smp_header, tstat_data = _read_smp_file(smp_file_path)
    task_map = _create_task_map(smp_header, tstat_data)
    
    if task not in task_map:
        available_tasks = list(task_map.keys())
        raise KeyError(f"Task '{task}' not found. Available tasks: {available_tasks}")
    
    return task_map[task]


def get_all_mri_tstats(patient_name: str) -> Dict[str, np.ndarray]:
    """
    Get all MRI t-stats for a specific patient.
    
    Args:
        patient_name: Patient identifier (e.g., 'P03')
        
    Returns:
        Dictionary mapping task names to t-stat arrays
        
    Raises:
        FileNotFoundError: If the SMP file doesn't exist
    """
    smp_file_path = _get_smp_file_path(patient_name)
    smp_header, tstat_data = _read_smp_file(smp_file_path)
    return _create_task_map(smp_header, tstat_data)
