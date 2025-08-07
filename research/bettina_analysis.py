#external imports
import numpy as np
import matplotlib

from lys.objects import TimeSeriesMeshData

matplotlib.use("Agg")

#internal imports
from lys.visualization import VTKScene
from lys.objects.experiment import create_experiment
from lys.objects.mesh import StaticMeshData, TimeSeriesMeshData
from lys.objects.jacobian import jacobian_to_vertex_val
from lys.processing.pipeline import ProcessingPipeline
from lys.utils.mri_tstat import get_mri_tstats
from lys.processing.steps import plot_hrf

# create experiment
experiment_name = "8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment.sessions = experiment.sessions[3:4]


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
    #{"BandpassFilter": {"lower_bound": 0.01, "upper_bound": 0.1}},
    {"ExtractHRFviaCanonicalFit": {
        "tmin":  -5,
        "tmax":  30,
        "tau_grid":   np.arange(0.6, 1.45, 0.05),
        "delay_grid": np.arange(-2.0, 2.25, 0.25),
        "ratio_grid": np.arange(0.10, 0.35, 0.05),
        "ridge_lambda": None,          # keep OLS
        "loss": "mad"                  # ← NEW (robust metric)
    }},
    {"ConvertToTStatsWithExtractedHRF": {}},
    {"ReconstructDualWithoutBadChannels":
        {"num_eigenmodes": 390,
         "lambda_selection": "corr",
         "mu_fixed": 0.1}},
    {"ReconstructSpatioTemporalEvolutionOLD": {
        "num_eigenmodes": 390,
        "lambda_reg": 1e3,
        "tmin": -5,
        "tmax": 30,
        "window": "hann"
    }},
]

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)


""" Plot hrf extracted from first session """
plot_hrf(experiment.sessions[0])

""" Check correlations and save reconstructed maps in a dictionary"""
corr_vals   = []                     # keeps the correlations only
recon_maps  = {                      # session‑ & task‑indexed store
    "HbO": {},                      #  recon_maps["HbO"][(sess_idx, task)] = map
    "HbR": {},
}

for sess_idx, session in enumerate(experiment.sessions):
    bad = session.processed_data["bad_channels"]
    print(f"\nSession {sess_idx}: {len(bad)} bad channel(s) → {bad}")

    for task in session.protocol.tasks:
        HbO_map = session.processed_data["t_HbO_reconstructed"][task]
        HbR_map = session.processed_data["t_HbR_reconstructed"][task]
        fmri_ts = get_mri_tstats(session.patient.name, task)

        r = np.corrcoef(HbO_map, fmri_ts)[0, 1]
        corr_vals.append(r)
        print(f"  {task:<20}  corr = {r:+.3f}")

        # ---------- stash the maps for later use ---------------------
        recon_maps["HbO"][(sess_idx, task)] = HbO_map
        recon_maps["HbR"][(sess_idx, task)] = HbR_map

# quick summary
avg, mn, mx = np.mean(corr_vals), np.min(corr_vals), np.max(corr_vals)
print(f"\nAverage fMRI–fNIRS correlation: {avg*100:5.2f}% "
      f"(min {mn*100:5.2f}% / max {mx*100:5.2f}%)")


ms_td = TimeSeriesMeshData(mesh, experiment.sessions[0].processed_data["neural_recon"]["HbO"]["MS"])
md_td = TimeSeriesMeshData(mesh, experiment.sessions[0].processed_data["neural_recon"]["HbO"]["MD"])
dscene = VTKScene(title = "Spatio-temporal reconstruction of mental drawing (MD)")
dscene.add(md_td)
dscene.show()

md_mean = np.mean(experiment.sessions[0].processed_data["neural_recon"]["HbO"]["MD"], axis=1)
scene = VTKScene(title = "Temporally averaged spatio-temporal reconstruction of mental drawing (MD)")
sdata = StaticMeshData(mesh, md_mean)
scene.add(sdata)
scene.show()

scene2 = VTKScene("Time-independent reconstruction of mental drawing (MD)")
sdata2 = StaticMeshData(mesh, recon_maps["HbO"][(0,"MD")])
scene2.add(sdata2)
scene2.show()
