from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")

config = {
    "BandpassFilter": {
        "upper_bound": 0.1,
        "lower_bound": 0.01,
    },
}

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)

