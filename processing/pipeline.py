from lys.objects import Experiment


class ProcessingPipeline:
    def from_config(self, config: dict):
        pass

    def apply(self, experiment: Experiment):
        pass