import re

def validate_patient_string(patient: str) -> None:
    if not isinstance(patient, str) or not re.match(r"^P\d+$", patient):
        raise ValueError(f"Invalid patient string: {patient}. Must be 'P' followed by a number, e.g., 'P03'.")

