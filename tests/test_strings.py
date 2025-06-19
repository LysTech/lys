import pytest
from lys.utils.strings import validate_patient_string

def test_validate_patient_string():
    # Test a few correct patient strings
    validate_patient_string("P03")
    validate_patient_string("P04")
    validate_patient_string("P123")

    # Test a few incorrect patient strings 
    with pytest.raises(ValueError):
        validate_patient_string("P03.4")
    with pytest.raises(ValueError):
        validate_patient_string("P03-4")
    with pytest.raises(ValueError):
        validate_patient_string("X03")