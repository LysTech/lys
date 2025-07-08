import numpy as np

from lys.objects.experiment import create_experiment
from lys.processing.pipeline import ProcessingPipeline

""" Demo of ProcessingPipeline. """

""" Load experiment """
experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment = experiment.filter_by_subjects(["P03"])
#experiment.sessions = experiment.sessions[:1]

""" Check mesh and volume alignmnent """
from lys.visualization import VTKScene
mesh = experiment.sessions[0].patient.mesh
segmentation = experiment.sessions[0].patient.segmentation
scene = VTKScene(title="Mesh and segmentation alignment")
# See how our alignment is not perfect :( !! 
scene.add(mesh).add(segmentation).format(segmentation, opacity=0.02).show()


""" Process the experiment with a few ProcessingSteps """ 
config = [
    {"ConvertWavelengthsToOD": {}},
    {"ConvertODtoHbOandHbR": {}},
    {"RemoveScalpEffect": {}},
]

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)


""" Check correlations """
from lys.utils.mri_tstat import get_mri_tstats 
for session in experiment.sessions:
    for task in session.protocol.tasks:
        print(f"Task: {task}")
        reconstructed_tstats = session.processed_data["t_HbO_reconstructed"][task]
        mri_tstats = get_mri_tstats(session.patient.name, task)
        corr = np.corrcoef(reconstructed_tstats, mri_tstats)[0, 1]
        print(f"Correlation: {corr}")

    