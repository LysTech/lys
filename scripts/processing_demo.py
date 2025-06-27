from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")

#TODO: would be more explicit that order is kept if config was a list
config = [
    {"ConvertWavelengthsToOD": {}},
    {"ConvertODtoHbOandHbR": {}},
    {"RemoveScalpEffect": {}},
    {"ConvertToTStats": {}}, #TODO: do we wanna have a /statistics folder to keep some of these functions?
    {"ReconstructEigenmodes": {"num_eigenmodes": 200,
                              "regularisation_param": 0.01}} #TODO: I guessed this param for my test, total BS probably!
]

#TODO: metadata should store the params! why?

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)

