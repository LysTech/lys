from lys.objects.experiment import create_experiment
from lys.processing.preprocessing import RawSessionPreProcessor
from lys.objects.session import get_session_paths

paths = get_session_paths("8classes", "nirs")
for path in paths:
    RawSessionPreProcessor.preprocess(path)
experiment_name = "8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment = experiment.filter_by_subjects(["P03"])

from lys.visualization import ChannelsPlot
data = experiment.sessions[0].raw_data["wl1"]
plot = ChannelsPlot()
plot.plot(data.T) #takes about 15 seconds to render on my mac