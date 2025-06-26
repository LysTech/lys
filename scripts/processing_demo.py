from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")

#TODO: would be more explicit that order is kept if config was a list

config = {
    "BandpassFilter": {
        "upper_bound": 0.1,
        "lower_bound": 0.01,
    },
    "ZTransform": {},
}

#TODO: metadata should store the params! why?

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)

#TODO: can we put the whole reconstruction pipeline here EASILY?

