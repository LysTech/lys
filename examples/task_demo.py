from pathlib import Path
from lys.data_recording.perceived_speech import PerceivedSpeechTaskExecutor

transcript_path = Path("/Users/thomasrialan/Documents/code/Geometric-eigenmodes/data/assets/audio/churchill/churchill_chunk_1_16k_mono.wav.json")

audio_path = Path("/Users/thomasrialan/Documents/code/Geometric-eigenmodes/data/assets/audio/churchill/churchill_chunk_1_16k_mono.wav")
task_executor = PerceivedSpeechTaskExecutor(transcript_path, audio_path)
session_dir = task_executor.create_new_session_dir("thomas", "perceived_speech", "flow2")
task_executor.start(session_dir)