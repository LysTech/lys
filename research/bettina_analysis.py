import numpy as np

from lys.visualization import VTKScene
from lys.objects.experiment import create_experiment
from lys.objects.mesh import StaticMeshData
from lys.objects.jacobian import jacobian_to_vertex_val
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "8classes"
experiment = create_experiment(experiment_name, "nirs")
#experiment.sessions = experiment.sessions[0:1]

##
#if False:
""" Check mesh and volume alignmnent """
mesh = experiment.sessions[0].patient.mesh
segmentation = experiment.sessions[0].patient.segmentation
scene = VTKScene(title="Mesh and segmentation alignment")
# See how our alignment is not perfect :( !!
scene.add(mesh).add(segmentation).format(segmentation, opacity=0.02).show()

""" Check projected Jacobian """
scene = VTKScene(title="Projected Jacobian")
sd_vertex_jacobian_wl1 = experiment.sessions[0].jacobians[0].sample_at_vertices(mesh.vertices)
vertex_jacobian_wl1 = jacobian_to_vertex_val(sd_vertex_jacobian_wl1, mode = "sum")
sd_mesh = StaticMeshData(mesh, np.sqrt(vertex_jacobian_wl1))
scene.add(sd_mesh).add(segmentation).format(segmentation, opacity=0.02).show()


config = [
    {"DetectBadChannels": {}},
    {"ConvertWavelengthsToOD": {}},
    {"ConvertODtoHbOandHbR": {}},
    {"RemoveScalpEffect": {}},
    {"BandpassFilter": {"lower_bound": 0.01, "upper_bound": 0.1}},
    {"ConvertToTStats": {}},
    {"ReconstructDualWithoutBadChannels":
        {"num_eigenmodes": 390,
         "lambda_selection": "corr",
         "mu_fixed": 0}},
]

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)



""" Check correlations """
from lys.utils.mri_tstat import get_mri_tstats
tstat_firsts = []
for session in experiment.sessions:
    for task in session.protocol.tasks:
        print(f"Task: {task}")
        reconstructed_tstats_HbO = session.processed_data["t_HbO_reconstructed"][task]
        reconstructed_tstats_HbR = session.processed_data["t_HbR_reconstructed"][task]
        print(f"Bad channels: {session.processed_data["bad_channels"]}")
        print(f"Number of bad channels: {len(session.processed_data["bad_channels"])}")
        mri_tstats = get_mri_tstats(session.patient.name, task)
        corr = np.corrcoef(reconstructed_tstats_HbO, mri_tstats)[0, 1]
        tstat_firsts.append(corr)
        print(f"Correlation: {corr}")


avg_score = np.mean(tstat_firsts)
min_score = np.min(tstat_firsts)
max_score = np.max(tstat_firsts)

print(f"Average fMRI–fNIRS (tstat-first) correlation across all tasks: {avg_score * 100:.2f}%")
print(f"Minimum fMRI–fNIRS correlation: {min_score * 100:.2f}%")
print(f"Maximum fMRI–fNIRS correlation: {max_score * 100:.2f}%")

