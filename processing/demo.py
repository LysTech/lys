from lys.objects.experiment import create_experiment


experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")
processing_pipeline = ProcessingPipeline(config)
dataset = processing_pipeline.run(experiment)

