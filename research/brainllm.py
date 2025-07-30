from lys.ml.experiment_to_dataset_converter import ExperimentToDatasetConverter
from lys.ml.splitting_strategies import TemporalSplitter
from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline


if __name__ == "__main__":
    experiment = create_experiment("perceived_speech", "flow2")
    
    pipeline = ProcessingPipeline([])
    experiment = pipeline.apply(experiment)
    
    splitter = TemporalSplitter()
    converter = ExperimentToDatasetConverter(splitter)
    datasets = converter.convert(experiment) 

    print(f"Training set X shape: {datasets.train.X.shape}, y shape: {datasets.train.y.shape}")
    print(f"Validation set X shape: {datasets.val.X.shape}, y shape: {datasets.val.y.shape}")
    print(f"Test set X shape: {datasets.test.X.shape}, y shape: {datasets.test.y.shape}")