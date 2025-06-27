from lys.objects import Experiment
from lys.processing.steps import ProcessingStep, BandpassFilter
import importlib
import inspect
from tqdm import tqdm


class ProcessingPipeline:
    """
    A pipeline that applies a sequence of processing steps to experiments.
    
    The pipeline is configured from a dictionary that specifies which processing
    steps to apply and their parameters.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the processing pipeline with a configuration.
        
        Args:
            config: Dictionary mapping step class names to parameter dictionaries.
                   If None, creates an empty pipeline.
                   
        Example:
            config = {
                "BandpassFilter": {
                    "lower_bound": 0.01,
                    "upper_bound": 0.1,
                },
            }
        """
        self.steps = []
        
        if config is not None:
            self._configure_from_dict(config)
    
    def apply(self, experiment: Experiment):
        """
        Apply all processing steps to the experiment.
        
        Args:
            experiment: The experiment to process
        """
        for session in tqdm(experiment.sessions, desc="Processing sessions"):
            for step in self.steps:
                step.process(session)
        return experiment

    def _configure_from_dict(self, config: dict) -> None:
        """
        Configure the pipeline from a dictionary specification.
        
        Args:
            config: Dictionary mapping step class names to parameter dictionaries
        """
        self.steps = []
        
        for step_name, step_params in config.items():
            step_class = self._get_processing_step_class(step_name)
            step_instance = step_class(**step_params)
            self.steps.append(step_instance)
    
    def _get_processing_step_class(self, step_name: str):
        """
        Get a ProcessingStep subclass by name.
        
        Args:
            step_name: Name of the processing step class
            
        Returns:
            The ProcessingStep subclass
            
        Raises:
            ValueError: If the processing step is unknown or if there are duplicate names
        """
        available_steps = self._get_available_processing_steps()
        matching_classes = [cls for cls in available_steps if cls.__name__ == step_name]
        
        if len(matching_classes) == 0:
            available_names = [cls.__name__ for cls in available_steps]
            raise ValueError(f"Unknown processing step: '{step_name}'. Available steps: {available_names}")
        elif len(matching_classes) > 1:
            raise ValueError(f"Multiple processing steps found with name '{step_name}': {matching_classes}")
        else:
            return matching_classes[0]
    
    def _get_available_processing_steps(self):
        """
        Get all available ProcessingStep subclasses.
        
        Returns:
            List of ProcessingStep subclasses
        """
        # Import the steps module to access all ProcessingStep subclasses
        from lys.processing import steps
        
        available_steps = []
        
        # Get all classes from the steps module
        for name, obj in inspect.getmembers(steps):
            if (inspect.isclass(obj) and 
                issubclass(obj, ProcessingStep) and 
                obj != ProcessingStep):
                available_steps.append(obj)
        
        return available_steps
