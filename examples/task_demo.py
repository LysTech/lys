
recorder = KernelFlowRecorder()
executor = PerceivedSpeechExecutor()
task = Task(recorder, executor)
session = task.create_new_session("thomas", "perceived_speech", "flow2")
task.start(session)