from lys.objects.experiment import create_experiment

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")

config = {
    "BandPassFilter": {
        "low_cutoff": 0.01,
        "high_cutoff": 0.1,
    },
}


processing_pipeline = ProcessingPipeline(config)
dataset = processing_pipeline.run(experiment)

