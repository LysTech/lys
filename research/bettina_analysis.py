import numpy as np

from lys.visualization import VTKScene
from lys.objects.experiment import create_experiment
from lys.objects.mesh import StaticMeshData
from lys.objects.jacobian import jacobian_to_vertex_val
from lys.processing.pipeline import ProcessingPipeline

experiment_name = "8classes"
experiment = create_experiment(experiment_name, "nirs")
experiment.sessions = experiment.sessions[3:5]

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
    {"BandpassFilter": {"lower_bound": 0.01*3.4722, "upper_bound": 0.2*3.4722}},
    {"ExtractHRF": {"tmin": -5.0, "tmax": 20.0}},  # ← new
    {"ConvertToTStats": {}},
    {"ReconstructDualWithoutBadChannels":
        {"num_eigenmodes": 390,
         "lambda_selection": "corr",
         "mu_fixed": 0.1}},
]

processing_pipeline = ProcessingPipeline(config)
experiment = processing_pipeline.apply(experiment)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Sequence, Union



import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union

def plot_hrf(session,
             tasks:      Union[None, str, Sequence[str]] = None,
             channels:   Union[str, int, Sequence[int]] = "mean",
             colors:     tuple[str, str] = ("C0", "C3"),  # HbO, HbR
             outfile:    str = "hrf.png"):
    """
    Grand‑average HRF (HbO & HbR) for a lys *Session*.

    Parameters
    ----------
    tasks   : • None  – all tasks are pooled and averaged
              • str   – *one* task name (plot this task only)
              • list/tuple[str] – explicit set of tasks to average
    channels: • "mean" – average across all *good* channels
              • int    – *one* channel index (plots that channel only)
              • list/tuple[int] – average across this set of indices
    colors  : pair of matplotlib colours (HbO, HbR)
    outfile : target path for the PNG

    Notes
    -----
    *Bad* channels (as detected by `DetectBadChannels`) are always ignored.
    A ±1 SD ribbon is drawn when more than one channel is averaged.
    """
    hrf   = session.processed_data["hrf"]
    t     = hrf["time"]                                # (L,)

    # -------- task selection -------------------------------------------
    if tasks is None:                    # all tasks
        sel_tasks = list(hrf["HbO"].keys())
    elif isinstance(tasks, str):         # single task
        if tasks not in hrf["HbO"]:
            raise ValueError(f"Task '{tasks}' not found in HRF data.")
        sel_tasks = [tasks]
    else:                                # explicit list/tuple
        sel_tasks = list(tasks)
        missing   = set(sel_tasks) - set(hrf["HbO"])
        if missing:
            raise ValueError(f"Unknown task(s): {', '.join(missing)}")

    # -------- good / bad channel bookkeeping ---------------------------
    bad   = np.asarray(session.processed_data.get("bad_channels", []), int)
    C_tot = next(iter(hrf["HbO"].values())).shape[1]
    good  = np.setdiff1d(np.arange(C_tot), bad, assume_unique=True)

    # -------- channel selection ----------------------------------------
    if channels == "mean":
        sel_ch = good
    elif isinstance(channels, int):
        if channels in bad:
            raise ValueError(f"Channel {channels} was flagged bad.")
        sel_ch = np.asarray([channels])
    else:
        sel_ch = np.setdiff1d(np.asarray(channels, int), bad, assume_unique=True)
        if sel_ch.size == 0:
            raise ValueError("All requested channels are bad.")

    # -------------------------------------------------------------------
    plt.figure(figsize=(6, 3))

    for key, col in zip(("HbO", "HbR"), colors):
        mats = np.stack([hrf[key][task] for task in sel_tasks], axis=0)  # (T, L, C)
        grand = mats.mean(axis=0)             # (L, C) average over tasks
        sel   = grand[:, sel_ch]              # (L, K)
        mu    = sel.mean(axis=1)
        sd    = sel.std(axis=1, ddof=0)

        plt.plot(t, mu, color=col, label=key)
        if sel_ch.size > 1:
            plt.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.4, linewidth=0)

    # -------- cosmetics -------------------------------------------------
    plt.axvline(0, lw=.6, color="k")
    plt.xlabel("Time [s]")
    plt.ylabel("ΔµM (baseline‑zeroed)")
    plt.title("Session‑averaged HRF (μ ± σ)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

# after processing_pipeline.apply(experiment) …
# first_session = experiment.sessions[0]
# plot_hrf(first_session,                       # default: all tasks & channels
#          channels="mean",                    # average across S·D
#          kinds=("HbO", "HbR"))


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

