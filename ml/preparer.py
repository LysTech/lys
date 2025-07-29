import numpy as np
from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from lys.objects.session import Session


class MLDataPreparer:
    """
    Prepares session data for machine learning applications.
    """

    def __init__(self, allowed_keys: Set[str] = None):
        """
        Initializes the MLDataPreparer.

        Args:
            allowed_keys: A set of keys to look for in processed_data for ML.
                          If None, defaults to {"wl1", "wl2", "HbO", "HbR"}.
        """
        if allowed_keys is None:
            self.allowed_keys = {"wl1", "wl2", "HbO", "HbR"}
        else:
            self.allowed_keys = allowed_keys

    def prepare(self, session: "Session"):
        """
        Gathers data from processed_data into a single array for ML purposes.

        This method looks for keys defined in `self.allowed_keys`
        in the session's processed_data dictionary. If found, their corresponding
        numpy arrays are stacked into a single array and stored under the "data_for_ml"
        key.

        A warning is issued for any other keys present in processed_data that were
        not part of this process.
        """
        ml_data_arrays = []
        
        original_keys = set(session.processed_data.keys())
        
        for key in sorted(list(self.allowed_keys)):
            if key in session.processed_data and isinstance(session.processed_data[key], np.ndarray):
                ml_data_arrays.append(session.processed_data[key])

        if ml_data_arrays:
            try:
                session.processed_data["data_for_ml"] = np.stack(ml_data_arrays, axis=-1)
            except ValueError as e:
                print(f"🔴 Warning: Could not stack data for ML in session {session.patient.name}. Error: {e}")
                session.processed_data["data_for_ml"] = np.array([])
        else:
            session.processed_data["data_for_ml"] = np.array([])
            
        processed_ml_keys = {key for key in self.allowed_keys if key in original_keys}
        warnable_keys = original_keys - processed_ml_keys
        
        if warnable_keys:
            print(f"🔴 Warning: The following keys from session.processed_data were not included in 'data_for_ml': {sorted(list(warnable_keys))}") 