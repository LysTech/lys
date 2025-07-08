from lys.objects.experiment import create_experiment

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment = experiment.filter_by_subjects(["P03"])

from lys.visualization import ChannelsPlot
data = experiment.sessions[0].raw_data["wl1"]
plot = ChannelsPlot()
plot.plot(data.T)