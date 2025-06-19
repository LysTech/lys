""" A segmentation is an instance of the Atlas class from objects.atlas. This file
    contains loader functions.
"""
import numpy as np
import os

from lys.objects.volume import from_jnii
from lys.objects.atlas import Atlas
from lys.utils.paths import lys_data_dir, check_file_exists
from lys.utils.strings import validate_patient_string


def load_charm_segmentation(patient: str) -> Atlas:
    validate_patient_string(patient)
    path = _segmentation_path(patient)
    check_file_exists(path)
    volume = from_jnii(path, show=False)
    return Atlas(volume.array.astype(np.int32))


def _segmentation_path(patient: str) -> str:
    root = lys_data_dir()
    return os.path.join(root, patient, "anat", "volumes", f"{patient}_7tissues.jnii")
