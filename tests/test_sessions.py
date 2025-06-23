from lys.objects.session import get_session_paths

def test_get_session_paths_fnirs_8classes():
    session_paths = get_session_paths('fnirs_8classes', 'nirs')
    assert len(session_paths) == 8, f"Expected 8 sessions, got {len(session_paths)}" 