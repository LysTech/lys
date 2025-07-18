""" A segmentation is an instance of the Atlas class from objects.atlas. This file
    contains loader functions.
"""
import numpy as np
import warnings

from lys.objects.volume import from_jnii
from lys.objects.atlas import Atlas
from lys.utils.paths import check_file_exists, get_subjects_dir
from lys.utils.strings import validate_patient_string
from typing import Optional
from pathlib import Path


def load_charm_segmentation(patient: str) -> Optional[Atlas]:
    validate_patient_string(patient)
    path_obj = _segmentation_path(patient)
    if not path_obj.exists():
        warnings.warn(f"Segmentation file not found for patient {patient} at {path_obj}. Returning None.")
        return None
    check_file_exists(str(path_obj))
    volume = from_jnii(str(path_obj))
    return Atlas(volume.array.astype(np.int32))


def _segmentation_path(patient: str) -> Path:
    root = get_subjects_dir()
    return Path(root) / patient / "anat" / "volumes" / f"{patient}_7tissues.jnii"

