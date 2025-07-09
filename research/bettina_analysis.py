import numpy as np

from lys.visualization import VTKScene
from lys.objects.experiment import create_experiment
from lys.objects.mesh import StaticMeshData
from lys.objects.jacobian import jacobian_to_vertex_val
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "fnirs_8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment.sessions = experiment.sessions[0:1]


if False:
    """ Check mesh and volume alignmnent """
    mesh = experiment.sessions[0].patient.mesh
    segmentation = experiment.sessions[0].patient.segmentation
    scene = VTKScene(title="Mesh and segmentation alignment")
    # See how our alignment is not perfect :( !! 
    scene.add(mesh).add(segmentation).format(segmentation, opacity=0.02).show()

    """ Check projected Jacobian """
    scene = VTKScene(title="Projected Jacobian")
    sd_vertex_jacobian_wl1 = experiment.sessions[0].jacobians[0].sample_at_vertices(mesh.vertices)
    vertex_jacobian_wl1 = jacobian_to_vertex_val(sd_vertex_jacobian_wl1)
    sd_mesh = StaticMeshData(mesh, vertex_jacobian_wl1)
    scene.add(sd_mesh).add(segmentation).format(segmentation, opacity=0.02).show()


config = [
    {"ConvertWavelengthsToOD": {}},
    {"ConvertODtoHbOandHbR": {}},
    {"RemoveScalpEffect": {}},
    {"ConvertToTStats": {}},
    {"ReconstructDual": {"num_eigenmodes": 390}}
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

    