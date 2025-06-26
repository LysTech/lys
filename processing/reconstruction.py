from lys.interfaces import ProcessingStep
from lys.objects import Session

class ReconstructionStep(ProcessingStep):
    def _do_process(self, session: Session) -> None:
        pass