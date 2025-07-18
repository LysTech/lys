from pathlib import Path
from lys.data_recording.perceived_speech import PerceivedSpeechTaskExecutor


task_executor = PerceivedSpeechTaskExecutor()
session_dir = task_executor.create_new_session_dir("thomas", "perceived_speech", "flow2")
task_executor.start(session_dir)