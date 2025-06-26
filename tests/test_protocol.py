import pytest
from pathlib import Path
from objects.protocol import Protocol
import tempfile
import os


def test_protocol_from_dict_orders_intervals():
    protocol_dict = {
        'A': [(2.0, 3.0), (0.0, 1.0)],
        'B': [(1.5, 2.5)]
    }
    protocol = Protocol.from_dict(protocol_dict)
    starts = [t[0] for t in protocol.intervals]
    assert starts == sorted(starts)
    labels = [t[2] for t in protocol.intervals]
    assert protocol.tasks == {'A', 'B'}


def test_protocol_from_dict_content():
    protocol_dict = {
        'A': [(0.0, 1.0)],
        'B': [(2.0, 3.0)]
    }
    protocol = Protocol.from_dict(protocol_dict)
    assert protocol.intervals == [
        (0.0, 1.0, 'A'),
        (2.0, 3.0, 'B')
    ]


def test_protocol_from_prt_parses_and_orders():
    prt_content = """
FileVersion:        2

ResolutionOfTime:   Seconds
    # iterate over subjects in folder structure
        # call create_session(subject_name, experiment_name)

Experiment:         NIRS-2011-05-25_002

BackgroundColor:    0 0 0
TextColor:          255 255 255
TimeCourseColor:    255 255 30
TimeCourseThick:    2
ReferenceFuncColor: 30 200 30
ReferenceFuncThick: 2

NrOfConditions:  2

MT
2
   59.7    64.7
   10.0    20.0
Color: 0 255 0

SN
1
   15.0    25.0
Color: 0 255 255
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        prt_path = Path(tmpdir) / "test.prt"
        prt_path.write_text(prt_content)
        protocol = Protocol.from_prt(prt_path)
        # Should be ordered by t_start
        assert protocol.intervals == [
            (10.0, 20.0, 'MT'),
            (15.0, 25.0, 'SN'),
            (59.7, 64.7, 'MT'),
        ]


def test_protocol_from_prt_raises_on_malformed():
    """ Deliberatly pass is a malformed prt file to make sure it raises an error """
    prt_content = """
FileVersion:        2
NrOfConditions:  1

MT
2
   59.7    64.7
   10.0
Color: 0 255 0
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        prt_path = Path(tmpdir) / "bad.prt"
        prt_path.write_text(prt_content)
        with pytest.raises(ValueError):
            Protocol.from_prt(prt_path)


def test_create_protocol_simple(tmp_path, monkeypatch):
    """A minimal test for create_protocol: loads a protocol with one interval and label."""
    import importlib
    # Patch lys_data_dir to return tmp_path as a Path object for this test only
    monkeypatch.setattr("lys.utils.paths.lys_data_dir", lambda: Path(tmp_path))
    import lys.objects.protocol as prot
    importlib.reload(prot)  # Force reload so it picks up the patched lys_data_dir

    # Setup directory structure
    patient = "P01"
    experiment = "exp"
    session = "sess"
    session_dir = tmp_path / patient / "nirs" / experiment / session
    session_dir.mkdir(parents=True)
    prt_path = session_dir / "protocol.prt"
    prt_content = """
FileVersion: 2
NrOfConditions: 1
REST
1
  0.0  10.0
Color: 0 0 0
"""
    prt_path.write_text(prt_content)
    protocol = prot.create_protocol(patient, experiment, session)
    assert protocol.intervals == [(0.0, 10.0, "REST")]