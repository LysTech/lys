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
        "tmax":  20,
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
    {"ReconstructSpatioTemporalEvolution": {
        "num_eigenmodes": 390,
        "lambda_reg": 5e3,
        "tmin": -5,
        "tmax": 20,
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
md_td_neural = TimeSeriesMeshData(mesh, experiment.sessions[0].processed_data["neural_recon"]["neural"]["MD"])
dscene = VTKScene(title = "Spatio-temporal reconstruction of mental drawing (MD)")
dscene.add(md_td_neural)
dscene.show()

md_mean = np.mean(experiment.sessions[0].processed_data["neural_recon"]["HbO"]["MD"], axis=1)
scene = VTKScene(title = "Temporally averaged spatio-temporal reconstruction of mental drawing (MD)")
sdata = StaticMeshData(mesh, md_mean)
scene.add(sdata)
scene.show()

scene2 = VTKScene(title = "Time-independent reconstruction of mental drawing (MD)")
sdata2 = StaticMeshData(mesh, recon_maps["HbO"][(0,"MD")])
scene2.add(sdata2)
scene2.show()

scene_fmri = VTKScene(title = "fMRI t-stat map")
fmri_data = StaticMeshData(mesh, get_mri_tstats(experiment.sessions[0].patient.name, "MD"))
scene_fmri.add(fmri_data)
scene_fmri.show()

# ------------------------------------------------------------------
# ---------- 5)  Z-normalise maps for fair comparison --------------
# ------------------------------------------------------------------
import numpy as np
from lys.objects.mesh import TimeSeriesMeshData, StaticMeshData
from lys.visualization import VTKScene

# -------- helper – robust z-score (median / MAD) ------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    """Return (x − median) / MAD with ε-guard against divide-by-zero."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return (x - med) / (mad + 1e-12)


# ------------------------------------------------------------------
# 6)  Spatio-temporal z-maps   (per-frame robust z-scoring)
# ------------------------------------------------------------------
sess0 = experiment.sessions[0]          # shorthand

# ----- Mental-subtraction (MS) ------------------------------------
ms_raw = sess0.processed_data["neural_recon"]["HbO"]["MS"]   # shape (V, T)
ms_z   = np.apply_along_axis(robust_z, 0, ms_raw)            # z per frame
ms_td  = TimeSeriesMeshData(mesh, ms_z)

scene_ms = VTKScene(title="Spatio-temporal z-map – mental subtraction (MS)")
scene_ms.add(ms_td).show()


# ----- Mental-drawing (MD) ---------------------------------------
md_raw = sess0.processed_data["neural_recon"]["neural"]["MD"]
md_z   = np.apply_along_axis(robust_z, 0, md_raw)
md_td  = TimeSeriesMeshData(mesh, md_z)

scene_md = VTKScene(title="Spatio-temporal z-map – mental drawing (MD)")
scene_md.add(md_td).show()


# ------------------------------------------------------------------
# 7)  Static maps (temporal mean + time-independent recon)
# ------------------------------------------------------------------

# ---- (a)  Mean over time of MD spatio-temporal recon ------------
md_mean_raw = np.mean(md_raw, axis=1)        # (V,)
md_mean_z   = robust_z(md_mean_raw)
sdata_mean  = StaticMeshData(mesh, md_mean_z)

scene_mean = VTKScene(title="Mean z-map – mental drawing (MD)")
scene_mean.add(sdata_mean).show()


# ---- (b)  Time-independent dual reconstruction of MD ------------
dual_raw = recon_maps["HbO"][(0, "MD")]      # (V,)
dual_z   = robust_z(dual_raw)
sdata_dual = StaticMeshData(mesh, dual_z)

scene_dual = VTKScene(title="Time-independent z-map – mental drawing (MD)")
scene_dual.add(sdata_dual).show()

# ------------------------------------------------------------------
# 6)  Vertex-wise z-score across time  -----------------------------
#     (keeps true amplitude relationships between frames)
# ------------------------------------------------------------------

def z_along_time(mat_VT: np.ndarray) -> np.ndarray:
    """
    Robust z-score *per vertex* across its time-course.
    Input shape (V, T) – returns same shape.
    """
    med  = np.median(mat_VT, axis=1, keepdims=True)
    mad  = np.median(np.abs(mat_VT - med), axis=1, keepdims=True)
    return (mat_VT - med) / (mad + 1e-12)

ms_z = z_along_time(ms_raw)          # shape (V, T)
md_z = z_along_time(md_raw)

ms_td = TimeSeriesMeshData(mesh, ms_z)
md_td = TimeSeriesMeshData(mesh, md_z)

VTKScene("Vertex-wise z-map – MS").add(ms_td).show()
VTKScene("Vertex-wise z-map – MD").add(md_td).show()
