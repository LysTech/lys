from lys.objects.experiment import create_experiment

experiment_name = "perceived_speech"
experiment = create_experiment(experiment_name, "flow2")

print(experiment.sessions[0].raw_data.keys())
print(experiment.sessions[0].raw_data['data'].shape)
print(experiment.sessions[0].raw_data['time'][:10])
print(experiment.sessions[0].protocol.intervals[:10])

time_data = experiment.sessions[0].raw_data['time']
print('First 10 time values:')
print(time_data[:10])
print('\\nLast 10 time values:')
print(time_data[-10:])
print('\\nTime differences:')
print(time_data[1:11] - time_data[0:10])
print('\\nUnique time values:', len(set(time_data)))
print('Total time samples:', len(time_data))



