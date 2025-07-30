import numpy as np

from lys.utils.mri_tstat import get_mri_tstats


def test_get_mri_tstats_dimensionality():
    """
    Test that get_mri_tstats returns an array with correct dimensionality.
    This is an integration test that requires actual P03 patient data to be available.
    """
    result = get_mri_tstats('P03', 'SN')
    
    assert isinstance(result, np.ndarray)
    assert len(result.shape) == 1  # Should be a 1D array
    assert len(result) == 32492  # Should contain t-stat values 