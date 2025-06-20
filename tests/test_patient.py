from lys.objects.patient import Patient
from lys.objects.atlas import Atlas
from lys.objects.mesh import Mesh


def test_create_patient_from_name():
    """
    Tests the creation of a Patient object using the from_name classmethod.
    This is an integration test and requires the "P03" patient data to be available.
    """
    patient = Patient.from_name("P03")

    assert isinstance(patient, Patient)
    assert patient.name == "P03"
    assert isinstance(patient.segmentation, Atlas)
    assert isinstance(patient.mesh, Mesh) 