import pytest
import os

from lys.utils.paths import check_file_exists, lys_data_dir

def test_check_file_exists():
    # Test with existing file
    check_file_exists(os.getcwd())
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError, match="File not found: /this/is/dummy/path"):
        check_file_exists("/this/is/dummy/path")

def test_lys_data_dir():
    lys_data_dir()