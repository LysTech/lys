from __future__ import annotations
import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import convolve
from scipy.stats import gamma
from lys.objects import Session
from lys.abstract_interfaces.processing_step import ProcessingStep

fs = 3.4722
DPF = 6.0
source_detector_distance = 3.0
extHbO_wl1 = 586
extHbR_wl1 = 1548.52
extHbO_wl2 = 1058
extHbR_wl2 = 691.32
ell1 = DPF * source_detector_distance
ell2 = DPF * source_detector_distance
A = np.array(
    [[extHbO_wl1 * ell1, extHbR_wl1 * ell1], [extHbO_wl2 * ell2, extHbR_wl2 * ell2]]
)
A_inv = np.linalg.inv(A)
num_sources = 16 #TODO: correct thing would be to have this be a property of session (perhaps session has all hardware info?)
num_detectors = 24

def canonical_double_gamma_hrf(tr=1.0, duration=30.0):
    times = np.arange(0, duration, tr)
    peak1, peak2 = 4, 10
    ratio = 1/6

    hrf1 = gamma.pdf(times, peak1)
    hrf2 = gamma.pdf(times, peak2)
    hrf = hrf1 - ratio * hrf2
    return hrf

# ─── extract_hrf.py ──────────────────────────────────────────────────────────
import numpy as np
from typing import Dict, List, Tuple
from lys.abstract_interfaces.processing_step import ProcessingStep
import matplotlib.pyplot as plt
from typing import Sequence, Union

# def plot_hrf(session,
#              tasks:      Union[None, str, Sequence[str]] = None,
#              channels:   Union[str, int, Sequence[int]] = "mean",
#              colors:     tuple[str, str] = ("C0", "C3"),
#              outfile:    str = "hrf.png",
#              *,
#              block_averaging: str = "all"):
#     """
#     Grand‑average HRF (HbO & HbR).
#
#     The *block_averaging* argument must match the setting that was used
#     when **ExtractHRF** ran on this session.  It is stored only for
#     user clarity – the plotting routine itself uses whatever HRF was
#     extracted earlier.
#     """
#     if block_averaging not in ("all", "longest"):
#         raise ValueError("block_averaging must be 'all' or 'longest'")
#
#     hrf   = session.processed_data["hrf"]
#     t     = hrf["time"]                                # (L,)
#
#     # -------- task selection -------------------------------------------
#     if tasks is None:                    # all tasks
#         sel_tasks = list(hrf["HbO"].keys())
#     elif isinstance(tasks, str):         # single task
#         if tasks not in hrf["HbO"]:
#             raise ValueError(f"Task '{tasks}' not found in HRF data.")
#         sel_tasks = [tasks]
#     else:                                # explicit list/tuple
#         sel_tasks = list(tasks)
#         missing   = set(sel_tasks) - set(hrf["HbO"])
#         if missing:
#             raise ValueError(f"Unknown task(s): {', '.join(missing)}")
#
#     # -------- good / bad channel bookkeeping ---------------------------
#     bad   = np.asarray(session.processed_data.get("bad_channels", []), int)
#     C_tot = next(iter(hrf["HbO"].values())).shape[1]
#     good  = np.setdiff1d(np.arange(C_tot), bad, assume_unique=True)
#
#     # -------- channel selection ----------------------------------------
#     if channels == "mean":
#         sel_ch = good
#     elif isinstance(channels, int):
#         if channels in bad:
#             raise ValueError(f"Channel {channels} was flagged bad.")
#         sel_ch = np.asarray([channels])
#     else:
#         sel_ch = np.setdiff1d(np.asarray(channels, int), bad, assume_unique=True)
#         if sel_ch.size == 0:
#             raise ValueError("All requested channels are bad.")
#
#     # -------------------------------------------------------------------
#     plt.figure(figsize=(6, 3))
#
#     for key, col in zip(("HbO", "HbR"), colors):
#         mats = np.stack([hrf[key][task] for task in sel_tasks], axis=0)  # (T, L, C)
#         grand = mats.mean(axis=0)             # (L, C) average over tasks
#         sel   = grand[:, sel_ch]
#         mu = np.nanmean(sel, axis=1)
#         sd = np.nanstd(sel, axis=1)
#         plt.plot(t, mu, color=col, label=key)
#         if sel_ch.size > 1:
#             plt.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.4, linewidth=0)
#
#     # -------- cosmetics -------------------------------------------------
#     plt.axvline(0, lw=.6, color="k")
#     plt.xlabel("Time [s]")
#     plt.ylabel("ΔµM (baseline‑zeroed)")
#     plt.title("Session‑averaged HRF (μ ± σ)")
#     plt.legend(frameon=False, fontsize=8)
#     plt.tight_layout()
#     plt.savefig(outfile)
#     plt.close()

def plot_hrf(session,
             tasks:      Union[None, str, Sequence[str]] = None,
             channels:   Union[str, int, Sequence[int]] = "mean",
             colors:     tuple[str, str] = ("C0", "C3"),
             outfile:    str = "hrf.png",
             outfile_fft: str = "hrf_fft.png",
             *,
             block_averaging: str = "all"):
    """
    Grand-average HRF (HbO & HbR) plus its Fourier-amplitude spectrum.

    The *block_averaging* argument must match the setting that was used
    when **ExtractHRF** ran on this session.  It is stored only for
    user clarity – the plotting routine itself uses whatever HRF was
    extracted earlier.

    Two files are written:

      • *outfile*       – time-domain HRF plot (default ``hrf.png``)
      • *outfile_fft*   – magnitude spectrum of the HbO HRF
                          (default ``hrf_fft.png``)
    """
    if block_averaging not in ("all", "longest"):
        raise ValueError("block_averaging must be 'all' or 'longest'")

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

    # -------- good / bad channel bookkeeping ---------------------------
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
    mu_HbO = None                                    # will store mean HbO HRF

    for key, col in zip(("HbO", "HbR"), colors):
        mats = np.stack([hrf[key][task] for task in sel_tasks], axis=0)  # (T, L, C)
        grand = mats.mean(axis=0)             # (L, C) average over tasks
        sel   = grand[:, sel_ch]
        mu = np.nanmean(sel, axis=1)
        sd = np.nanstd(sel, axis=1)
        plt.plot(t, mu, color=col, label=key)
        if sel_ch.size > 1:
            plt.fill_between(t, mu - sd, mu + sd, color=col, alpha=0.4, linewidth=0)

        if key == "HbO":                       # capture for Fourier spectrum
            mu_HbO = mu.copy()

    # -------- cosmetics -------------------------------------------------
    plt.axvline(0, lw=.6, color="k")
    plt.xlabel("Time [s]")
    plt.ylabel("ΔµM (baseline-zeroed)")
    plt.title("Session-averaged HRF (μ ± σ)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

    # ---------- Fourier spectrum ---------------------------------------
    if mu_HbO is not None:
        dt    = np.mean(np.diff(t))                 # sampling period
        freqs = np.fft.rfftfreq(mu_HbO.size, dt)    # one-sided frequencies
        amp   = np.abs(np.fft.rfft(mu_HbO))         # magnitude spectrum

        plt.figure(figsize=(6, 3))
        plt.plot(freqs, amp, color=colors[0])
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("|H(f)|")
        plt.title("Fourier spectrum of HRF (HbO mean)")
        plt.tight_layout()
        plt.savefig(outfile_fft)
        plt.close()



# ------------------------------------------------------------------
# ↓↓↓ 1) ExtractHRF with block_averaging option  ↓↓↓
# ------------------------------------------------------------------
# ─── extract_hrf.py ──────────────────────────────────────────────────────
import numpy as np
from typing import Dict, List, Tuple
from lys.abstract_interfaces.processing_step import ProcessingStep

# ─── extract_hrf_glm.py ────────────────────────────────────────────────────────
import numpy as np
from typing import Dict, Tuple, Sequence, Union
from lys.abstract_interfaces.processing_step import ProcessingStep

# Re‑use the global sampling rate defined elsewhere in your pipeline
fs = globals().get("fs", 3.4722)          # Hz

from scipy.linalg import pinv
from scipy.signal import savgol_filter

# ─── ExtractHRFviaGLM (spline-basis + ridge) ────────────────────────────────
import numpy as np
from scipy.interpolate import BSpline, splev
from scipy.linalg import toeplitz, cho_factor, cho_solve
from lys.abstract_interfaces.processing_step import ProcessingStep
from typing import Tuple, Dict

# ─── ExtractHRFviaGLM (spline‑basis + ridge) ───────────────────────────────
import numpy as np
from scipy.interpolate import BSpline, splev
from scipy.linalg import toeplitz, block_diag, cho_factor, cho_solve
from lys.abstract_interfaces.processing_step import ProcessingStep
from typing import Tuple, Dict

# ──────────────────────────────────────────────────────────────────────────
#  ExtractHRFviaGLM – low‑dimensional GLM HRF extractor
# ──────────────────────────────────────────────────────────────────────────


from typing import Dict, Tuple, Optional

import numpy as np
from numpy.linalg import pinv
from scipy.interpolate import BSpline, splev

from lys.abstract_interfaces.processing_step import ProcessingStep

# --------------------------------------------------------------------------
class ExtractHRFviaGLM(ProcessingStep):
    """
    Channel‑wise HRF estimation with a *basis‑function GLM*.

    Workflow
    --------
    1. Choose a temporal basis B(τ)  (cubic B‑splines by default).
    2. For every condition, convolve its box‑car on/off vector with **each**
       column of B to build one long design matrix *X*.
    3. Solve      β = (XᵀX + λI)⁻¹ Xᵀ y     (ridge optional).
    4. Recreate a dense HRF     h(τ) = B(τ) β     for every channel.

    Results are stored just like the “classic” ExtractHRF::

        session.processed_data["hrf"] = {
            "time" : 1‑D array (L,),
            "HbO"  : {cond: (L, C) ndarray},
            "HbR"  : {cond: (L, C) ndarray},
        }

    Parameters
    ----------
    tmin, tmax : float
        Window [s] around each onset that the model spans (‑5…30 s by default).
    basis : {"splines", "fir"}
        • *splines* – cubic B‑splines (recommended, smooth & compact).
        • *fir*     – “stick” FIR, one regressor per sample in the window.
    n_knots : int
        Number of interior knots for the spline basis (ignored for FIR).
    ridge_lambda : float or None
        If given, adds λI Tikhonov regularisation (ridge).  Set to *None* for
        ordinary least squares.
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 basis: str = "splines",
                 n_knots: int = 8,
                 ridge_lambda: Optional[float] = None):
        if basis not in ("splines", "fir"):
            raise ValueError("basis must be 'splines' or 'fir'")
        self.tmin          = float(tmin)
        self.tmax          = float(tmax)
        self.basis         = basis
        self.n_knots       = int(n_knots)
        self.ridge_lambda  = ridge_lambda

    # ======================================================================
    # helpers – temporal bases
    # ======================================================================
    def _spline_basis(self, t: np.ndarray) -> np.ndarray:
        """Cubic B‑spline basis evaluated at times *t* [s]."""
        deg   = 3                                               # cubic
        knots = np.linspace(self.tmin, self.tmax, self.n_knots) # interior
        # repeat first & last knot 'deg' times → open spline
        t_aug = np.r_[
            np.full(deg,  knots[0]),
            knots,
            np.full(deg,  knots[-1])
        ]
        # one minimal‑support coefficient → one basis function
        coeff = np.eye(len(t_aug) - deg - 1)
        spl   = [BSpline(t_aug, coeff[i], deg, extrapolate=False)
                 for i in range(coeff.shape[0])]
        B     = np.column_stack([splev(t, s) for s in spl])     # (N, P+deg)
        return B[:, deg:]                       # first 'deg' cols are zeros

    def _fir_basis(self, t: np.ndarray, fs: float) -> np.ndarray:
        """Simple FIR “stick” basis (one per sample)."""
        L   = int(round((self.tmax - self.tmin) * fs))
        idx = ((t - self.tmin) * fs).round().astype(int)
        B   = np.zeros((t.size, L))
        good = (idx >= 0) & (idx < L)
        B[good, idx[good]] = 1.0
        return B

    def _build_basis(self, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return (basis matrix Bτ, time_axis)."""
        L        = int(round((self.tmax - self.tmin) * fs))
        time_ax  = np.arange(L) / fs + self.tmin
        if self.basis == "splines":
            B_tau = self._spline_basis(time_ax)
        else:
            B_tau = self._fir_basis(time_ax, fs)
        return B_tau, time_ax

    # ======================================================================
    # main entry
    # ======================================================================
    def _do_process(self, session: "Session") -> None:          # noqa: N802
        HbO = session.processed_data["HbO"]      # (T, C)
        HbR = session.processed_data["HbR"]
        fs  = globals().get("fs", 3.4722)        # Hz

        B_tau, t_vec = self._build_basis(fs)     # (L, P)
        P    = B_tau.shape[1]
        T, C = HbO.shape

        hrf_HbO: Dict[str, np.ndarray] = {}
        hrf_HbR: Dict[str, np.ndarray] = {}

        for cond in session.protocol.tasks:
            # ---------------- design matrix ----------------------------
            X = np.zeros((T, P))
            for t0, _, lbl in session.protocol.intervals:
                if lbl != cond:
                    continue
                start = int(round((t0 + self.tmin) * fs))
                end   = start + B_tau.shape[0]
                if start < 0 or end > T:        # skip truncated epochs
                    continue
                X[start:end, :] += B_tau        # allow overlaps

            if not X.any():                     # no complete epochs
                continue

            # ---------------- ridge / OLS solve ------------------------
            if self.ridge_lambda is None:
                XtX_inv = pinv(X.T @ X)
            else:
                lam     = float(self.ridge_lambda)
                XtX_inv = pinv(X.T @ X + lam * np.eye(P))

            β_O = XtX_inv @ X.T @ HbO           # (P, C)
            β_R = XtX_inv @ X.T @ HbR

            hrf_HbO[cond] = B_tau @ β_O         # (L, C)
            hrf_HbR[cond] = B_tau @ β_R

        # ---------------- stash result --------------------------------
        session.processed_data["hrf"] = {
            "time": t_vec,
            "HbO":  hrf_HbO,
            "HbR":  hrf_HbR,
        }

# ─── canonical_hrf_tools.py ────────────────────────────────────────────────
import numpy as np
from numpy.linalg import lstsq
from scipy.signal import convolve
from scipy.stats import gamma
from lys.abstract_interfaces.processing_step import ProcessingStep

# ─── canonical_hrf.py ────────────────────────────────────────────────
import numpy as np
from scipy.stats import gamma


def canonical_hrf(t: np.ndarray,
                  tau: float   = 1.0,
                  ratio: float = 1/6,
                  delay: float = 0.0,
                  p1:   int    = 6,
                  p2:   int    = 16) -> np.ndarray:
    """
    Double-gamma haemodynamic response function.

    Parameters
    ----------
    t      : time vector [s]
    tau    : common time-dilation (>1 = slower / broader)
    ratio  : undershoot / peak amplitude (≈ 1/6 for SPM default)
    delay  : pure time shift of the whole kernel (+ = later)
    p1, p2 : shape parameters of the two gamma functions
             (kept fixed at 6 & 16 – can also be added to the grid)

    Returns
    -------
    hrf(t) normalised to peak +1.
    """
    t_scal = (t - delay) / tau
    hrf = gamma.pdf(t_scal, p1) - ratio * gamma.pdf(t_scal, p2)
    hrf /= np.max(np.abs(hrf)) + 1e-12
    return hrf


# ───────────────────────────────────────────────────────────────────────────
# ─── extract_hrf_via_canonical_fit.py ───────────────────────────────

import numpy as np
from itertools import product
from numpy.linalg import lstsq, pinv
from scipy.signal import convolve

from lys.abstract_interfaces.processing_step import ProcessingStep

'''
class ExtractHRFviaCanonicalFit(ProcessingStep):
    """
    Canonical-HRF GLM with **three global shape parameters**
    (τ, δ, ρ) chosen by grid search.

    Adds to `session.processed_data` the dictionary

        "hrf" = {
            "time" : (L,),                 # seconds  (tmin…tmax)
            "HbO"  : {task: (L,C) ndarray},
            "HbR"  : {task: (L,C) ndarray},
            "tau"  : τ*,   "delay": δ*,   "ratio": ρ*
        }

    Parameters
    ----------
    tmin, tmax : epoch window relative to onset [s]
    tau_grid   : iterable of τ values   (default 0.6 … 1.4)
    delay_grid : iterable of δ values   (default –2 … +2 s)
    ratio_grid : iterable of ρ values   (default 0.10 … 0.30)
    ridge_lambda : None = pure OLS, otherwise λ for ridge
    baseline : (float, float)
        baseline window [s] relative to onset subtracted per channel
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 *,
                 tau_grid:   np.ndarray | list = np.arange(0.6, 1.45, 0.05),
                 delay_grid: np.ndarray | list = np.arange(-2.0, 2.25, 0.25),
                 ratio_grid: np.ndarray | list = np.arange(0.10, 0.35, 0.05),
                 ridge_lambda: float | None = None,
                 baseline: tuple[float, float] = (-5.0, 0.0)):
        self.tmin, self.tmax = float(tmin), float(tmax)
        self.tau_grid   = np.asarray(tau_grid,   float)
        self.delay_grid = np.asarray(delay_grid, float)
        self.ratio_grid = np.asarray(ratio_grid, float)
        self.ridge_lambda = ridge_lambda
        self.baseline_window = baseline

    # ======================= helpers =================================
    def _design_matrix(self,
                       protocol,
                       fs: float,
                       n_time: int,
                       hrf_kernel: np.ndarray,
                       tasks: list[str]) -> np.ndarray:
        """Box-car for every task, convolved with *hrf_kernel*."""
        n_tasks = len(tasks)
        X = np.zeros((n_time, n_tasks))
        for j, task in enumerate(tasks):
            for onset, offset, lbl in protocol.intervals:
                if lbl != task:
                    continue
                on = int(round(onset * fs))
                off = int(round(offset * fs))
                if on >= n_time:
                    continue
                X[on:off, j] = 1.0
        # HRF convolution per column
        for j in range(n_tasks):
            X[:, j] = convolve(X[:, j], hrf_kernel, mode="full")[:n_time]
        return X

    def _fit_beta_rss(self, Y: np.ndarray, X: np.ndarray, lam: float | None):
        """Return β and residual-sum-of-squares."""
        if lam is None:                          # ordinary least squares
            β, *_ = lstsq(X, Y, rcond=None)
        else:                                    # ridge
            XtX_inv = pinv(X.T @ X + lam * np.eye(X.shape[1]))
            β = XtX_inv @ X.T @ Y
        res = Y - X @ β
        return β, float(np.sum(res * res))

    # ======================= main ====================================
    def _do_process(self, session: "Session") -> None:          # noqa: N802
        HbO = session.processed_data["HbO"]        # (T, C)
        HbR = session.processed_data["HbR"]
        fs  = globals().get("fs", 3.4722)          # Hz
        T, C = HbO.shape

        # ------------- baseline correction (optional) -----------------
        b0 = int(round(self.baseline_window[0] * fs))
        b1 = int(round(self.baseline_window[1] * fs))
        if 0 <= b0 < b1 <= T:
            HbO = HbO - HbO[b0:b1].mean(axis=0)
            HbR = HbR - HbR[b0:b1].mean(axis=0)

        tasks = list(session.protocol.tasks)

        # ------------- time axis for the HRF to be returned ----------
        L = int(round((self.tmax - self.tmin) * fs))
        t_hrf = np.arange(L) / fs + self.tmin

        # ------------- brute-force search over (τ, δ, ρ) --------------
        best = {"rss": np.inf}
        n_time = T
        for τ, δ, ρ in product(self.tau_grid,
                               self.delay_grid,
                               self.ratio_grid):
            hrf_kern = canonical_hrf(t_hrf, tau=τ, ratio=ρ, delay=δ)
            X = self._design_matrix(session.protocol, fs, n_time,
                                    hrf_kern, tasks)
            β_O, rss_O = self._fit_beta_rss(HbO, X, self.ridge_lambda)
            β_R, rss_R = self._fit_beta_rss(HbR, X, self.ridge_lambda)
            rss_tot = rss_O + rss_R
            if rss_tot < best["rss"]:
                best.update(tau=τ, delay=δ, ratio=ρ,
                            rss=rss_tot, βO=β_O, βR=β_R, X=X)
        # ------- NEW: automatic global-sign correction -------------------
        # median_beta = np.median(np.concatenate((best["βO"].ravel(),
        #                                         best["βR"].ravel())))
        # if median_beta < 0:
        #     print("[Canonical-Fit]   β-sign flipped (auto-correct)")
        #     best["βO"] *= -1.0
        #     best["βR"] *= -1.0

        # ------------- reconstruct channel-wise HRFs -----------------
        hrf_kernel = canonical_hrf(t_hrf,
                                   tau   = best["tau"],
                                   ratio = best["ratio"],
                                   delay = best["delay"])
        hrf_HbO = {}
        hrf_HbR = {}
        for j, task in enumerate(tasks):
            βO = best["βO"][j]          # (C,)
            βR = best["βR"][j]
            hrf_HbO[task] = np.outer(hrf_kernel, βO)   # (L, C)
            hrf_HbR[task] = np.outer(hrf_kernel, βR)

        # ------------- stash in session ------------------------------
        session.processed_data["hrf"] = {
            "time":  t_hrf,
            "HbO":   hrf_HbO,
            "HbR":   hrf_HbR,
            "tau":   best["tau"],
            "delay": best["delay"],
            "ratio": best["ratio"],
        }
        print(f"[Canonical-Fit]  τ*={best['tau']:.2f}  "
              f"δ*={best['delay']:+.2f}s  ρ*={best['ratio']:.2f}  "
              f"RSS={best['rss']:.3g}")
        # ------------- Diagnostics: β histogram + design matrix ----------
        import matplotlib.pyplot as plt

        # β-histogram (HbO only)
        β_all = best["βO"].ravel()
        plt.figure(figsize=(4, 3))
        plt.hist(β_all, bins=40, color="C0", edgecolor="k")
        plt.axvline(0, color="k", lw=0.6)
        plt.title("Distribution of β_O (HbO amplitudes)")
        plt.xlabel("β_O"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("canonical_beta_hist.png")
        plt.close()

        # Design matrix
        plt.figure(figsize=(6, 4))
        plt.imshow(best["X"], aspect="auto", cmap="gray_r", interpolation="nearest")
        plt.title("Design matrix X (HRF-convolved)")
        plt.xlabel("Regressor"); plt.ylabel("Time")
        plt.tight_layout()
        plt.savefig("canonical_design_matrix.png")
        plt.close()
'''
# ─── extract_hrf_via_canonical_fit.py ───────────────────────────────
import numpy as np
from itertools import product
from numpy.linalg import lstsq, pinv
from scipy.signal import convolve

from lys.abstract_interfaces.processing_step import ProcessingStep


class ExtractHRFviaCanonicalFit(ProcessingStep):
    """
    Canonical-HRF GLM with brute-force search over three global shape
    parameters (τ, δ, ρ).

    Adds to ``session.processed_data``::

        "hrf" = {
            "time"  : (L,),                 # seconds  (tmin…tmax)
            "HbO"   : {task: (L,C) ndarray},
            "HbR"   : {task: (L,C) ndarray},
            "tau"   : τ*,   "delay": δ*,   "ratio": ρ*
        }

    Parameters
    ----------
    tmin, tmax : float
        Window around each onset [s]  (default –5 … +30).
    tau_grid, delay_grid, ratio_grid : array-like
        Values tested for τ (time dilation), δ (delay) and ρ (undershoot/peak).
    ridge_lambda : float | None
        If *None* → ordinary least squares, else λ for ridge.
    baseline : (float, float)
        Baseline window [s] relative to onset, removed from each channel.
    loss : {"rss", "mad"}
        Error metric used during the grid search

        * ``"rss"`` – residual-sum-of-squares  Σ‖y−Xβ‖²  (original behaviour)
        * ``"mad"`` – **robust** sum of median-absolute-deviation (per channel)
                      Σ     MAD_t | (res_i − median(res_i)) |
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 *,
                 tau_grid:   np.ndarray | list = np.arange(0.6, 1.45, 0.05),
                 delay_grid: np.ndarray | list = np.arange(-2.0, 2.25, 0.25),
                 ratio_grid: np.ndarray | list = np.arange(0.10, 0.35, 0.05),
                 ridge_lambda: float | None = None,
                 baseline: tuple[float, float] = (-5.0, 0.0),
                 loss: str = "rss"):
        if loss not in ("rss", "mad"):
            raise ValueError("loss must be 'rss' or 'mad'")
        self.tmin, self.tmax = float(tmin), float(tmax)
        self.tau_grid   = np.asarray(tau_grid,   float)
        self.delay_grid = np.asarray(delay_grid, float)
        self.ratio_grid = np.asarray(ratio_grid, float)
        self.ridge_lambda    = ridge_lambda
        self.baseline_window = baseline
        self.loss            = loss    # NEW

    # ======================= helpers =================================
    def _design_matrix(self,
                       protocol,
                       fs: float,
                       n_time: int,
                       hrf_kernel: np.ndarray,
                       tasks: list[str]) -> np.ndarray:
        """Box-car for every task, convolved with *hrf_kernel*."""
        n_tasks = len(tasks)
        X = np.zeros((n_time, n_tasks))
        for j, task in enumerate(tasks):
            for onset, offset, lbl in protocol.intervals:
                if lbl != task:
                    continue
                on  = int(round(onset * fs))
                off = int(round(offset * fs))
                if on >= n_time:
                    continue
                X[on:off, j] = 1.0
        # Convolve each regressor with the HRF
        for j in range(n_tasks):
            X[:, j] = convolve(X[:, j], hrf_kernel, mode="full")[:n_time]
        return X

    # ------------------------------------------------------------------
    def _solve_beta(self, Y: np.ndarray, X: np.ndarray, lam: float | None):
        """Return β  (no error metric here)."""
        if lam is None:                              # OLS
            β, *_ = lstsq(X, Y, rcond=None)
        else:                                        # ridge
            XtX_inv = pinv(X.T @ X + lam * np.eye(X.shape[1]))
            β = XtX_inv @ X.T @ Y
        return β                                     # (P, C)

    # ======================= main ====================================
    def _do_process(self, session: "Session") -> None:          # noqa: N802
        HbO = session.processed_data["HbO"]        # (T, C)
        HbR = session.processed_data["HbR"]
        fs  = globals().get("fs", 3.4722)          # Hz
        T, C = HbO.shape

        # -------- baseline correction (optional) -----------------------
        b0 = int(round(self.baseline_window[0] * fs))
        b1 = int(round(self.baseline_window[1] * fs))
        if 0 <= b0 < b1 <= T:
            HbO = HbO - HbO[b0:b1].mean(axis=0)
            HbR = HbR - HbR[b0:b1].mean(axis=0)

        tasks = list(session.protocol.tasks)

        # -------- time axis for HRF -----------------------------------
        L = int(round((self.tmax - self.tmin) * fs))
        t_hrf = np.arange(L) / fs + self.tmin

        # -------- grid search over (τ, δ, ρ) --------------------------
        best = {"score": np.inf}
        n_time = T
        for τ, δ, ρ in product(self.tau_grid,
                               self.delay_grid,
                               self.ratio_grid):
            hrf_kern = canonical_hrf(t_hrf, tau=τ, ratio=ρ, delay=δ)
            X = self._design_matrix(session.protocol, fs, n_time,
                                    hrf_kern, tasks)

            β_O = self._solve_beta(HbO, X, self.ridge_lambda)   # (P,C)
            β_R = self._solve_beta(HbR, X, self.ridge_lambda)

            # ---------- compute error according to chosen metric -----
            res_O = HbO - X @ β_O
            res_R = HbR - X @ β_R

            if self.loss == "rss":
                score = float(np.sum(res_O * res_O) + np.sum(res_R * res_R))
            else:  # "mad"
                mad_O = np.median(np.abs(res_O -
                                         np.median(res_O, axis=0, keepdims=True)),
                                  axis=0)   # (C,)
                mad_R = np.median(np.abs(res_R -
                                         np.median(res_R, axis=0, keepdims=True)),
                                  axis=0)
                score = float(np.sum(mad_O + mad_R))

            if score < best["score"]:
                best.update(score=score, tau=τ, delay=δ, ratio=ρ,
                            βO=β_O, βR=β_R, X=X)

        # -------- reconstruct channel-wise HRFs -----------------------
        hrf_kernel = canonical_hrf(t_hrf,
                                   tau   = best["tau"],
                                   ratio = best["ratio"],
                                   delay = best["delay"])
        hrf_HbO, hrf_HbR = {}, {}
        for j, task in enumerate(tasks):
            hrf_HbO[task] = np.outer(hrf_kernel, best["βO"][j])  # (L,C)
            hrf_HbR[task] = np.outer(hrf_kernel, best["βR"][j])

        # -------- stash in session ------------------------------------
        session.processed_data["hrf"] = {
            "time":  t_hrf,
            "HbO":   hrf_HbO,
            "HbR":   hrf_HbR,
            "tau":   best["tau"],
            "delay": best["delay"],
            "ratio": best["ratio"],
        }

        print(f"[Canonical-Fit/{self.loss.upper()}]  "
              f"τ*={best['tau']:.2f}  δ*={best['delay']:+.2f}s  "
              f"ρ*={best['ratio']:.2f}  score={best['score']:.3g}")

        # ---------------- diagnostics (unchanged) ----------------------
        import matplotlib.pyplot as plt

        β_all = best["βO"].ravel()
        plt.figure(figsize=(4, 3))
        plt.hist(β_all, bins=40, color="C0", edgecolor="k")
        plt.axvline(0, color="k", lw=0.6)
        plt.title("Distribution of β_O (HbO amplitudes)")
        plt.xlabel("β_O"); plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig("canonical_beta_hist.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(best["X"], aspect="auto", cmap="gray_r",
                   interpolation="nearest")
        plt.title("Design matrix X (HRF-convolved)")
        plt.xlabel("Regressor"); plt.ylabel("Time")
        plt.tight_layout()
        plt.savefig("canonical_design_matrix.png")
        plt.close()

class ExtractHRF(ProcessingStep):
    """
    Event‑locked, baseline‑corrected HRF extraction.

    Parameters
    ----------
    tmin, tmax : float
        Window [s] relative to onset (default –5…+30).
    baseline : (float, float)
        Baseline window [s] relative to onset (default –5…0).
    block_averaging : {"all", "longest"}
        "all"      – use every block
        "longest"  – average **up to 5 longest blocks** per task
                     (fewer if the task occurs < 5 times)
    """

    def __init__(self,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 baseline: Tuple[float, float] = (-5.0, 0.0),
                 block_averaging: str = "all"):
        if block_averaging not in ("all", "longest"):
            raise ValueError("block_averaging must be 'all' or 'longest'")
        self.tmin, self.tmax = tmin, tmax
        self.baseline        = baseline
        self.block_mode      = block_averaging

    # ------------------------------------------------------------------ #
    # main entry
    # ------------------------------------------------------------------ #
    def _do_process(self, session: "Session") -> None:

        HbO = session.processed_data["HbO"]      # (T, C)
        HbR = session.processed_data["HbR"]
        fs  = globals().get("fs", 3.4722)        # Hz
        C   = HbO.shape[1]

        # unified time axis (full window from tmin … tmax)
        L_full   = int(round((self.tmax - self.tmin) * fs))
        t_axis   = np.arange(L_full) / fs + self.tmin

        # cache block lengths
        blocks = [(s, e, lbl, e - s) for s, e, lbl in session.protocol.intervals]

        hrf_HbO: Dict[str, np.ndarray] = {}
        hrf_HbR: Dict[str, np.ndarray] = {}

        # ================================================================
        # iterate over tasks / conditions
        # ================================================================
        for task in session.protocol.tasks:

            # ---------- choose which onsets to use ----------------------
            if self.block_mode == "longest":
                cand = sorted(
                    [(length, s) for s, _, lbl, length in blocks if lbl == task],
                    key=lambda t: t[0], reverse=True)[:5]         # TOP‑5
                onsets = [s for _, s in cand]
            else:       # "all"
                onsets = [s for s, _, lbl, _ in blocks if lbl == task]

            if not onsets:        # nothing to average
                continue

            seg_O, seg_R = [], []

            # ============================================================
            # build per‑epoch matrices
            # ============================================================
            for t0 in onsets:
                # retrieve true block length (seconds)
                true_len = next(length for s,e,lbl,length in blocks
                                if s == t0 and lbl == task)
                valid_L  = int(round((true_len - self.tmin) * fs))
                valid_L  = np.clip(valid_L, 0, L_full)            # guard rails

                # --- extract raw epoch --------------------------------
                start = int(round((t0 + self.tmin) * fs))
                if start < 0:                                      # pre‑data
                    continue
                raw_O = HbO[start:start + valid_L, :].copy()
                raw_R = HbR[start:start + valid_L, :].copy()

                # --- baseline correction -----------------------------
                b0 = int(round((t0 + self.baseline[0]) * fs))
                b1 = int(round((t0 + self.baseline[1]) * fs))
                if b0 >= 0 and b1 > b0:
                    raw_O -= HbO[b0:b1].mean(axis=0)
                    raw_R -= HbR[b0:b1].mean(axis=0)

                # --- pad with NaNs up to the full window length -------
                pad_O = np.full((L_full, C), np.nan)
                pad_R = np.full_like(pad_O, np.nan)
                pad_O[:valid_L, :] = raw_O
                pad_R[:valid_L, :] = raw_R

                seg_O.append(pad_O)
                seg_R.append(pad_R)

            # ============================================================
            # average across epochs  (ignoring NaNs)
            # ============================================================
            if seg_O:        # at least one good epoch
                cube_O = np.stack(seg_O)            # (E, L_full, C)
                cube_R = np.stack(seg_R)
                hrf_HbO[task] = np.nanmean(cube_O, axis=0)
                hrf_HbR[task] = np.nanmean(cube_R, axis=0)

        # stash result
        session.processed_data["hrf"] = {
            "time": t_axis,        # length = L_full
            "HbO":  hrf_HbO,
            "HbR":  hrf_HbR,
        }


# class ExtractHRF(ProcessingStep):
#     """
#     Event‑locked, baseline‑corrected HRF extraction.
#
#     Adds:
#         session.processed_data["hrf"] = {
#             "time" : 1‑D array,                 # seconds relative to onset
#             "HbO"  : {cond: (L, C) ndarray},    # L = n_lags, C = n_channels
#             "HbR"  : {cond: (L, C) ndarray},
#         }
#     """
#
#     def __init__(self,
#                  tmin: float = -5.0,          # s before onset  (baseline window starts here)
#                  tmax: float = 30.0,          # s after onset   (end of HRF window)
#                  baseline: Tuple[float,float] = (-5.0, 0.0)  # s w.r.t onset
#                  ):
#         self.tmin, self.tmax = tmin, tmax
#         self.baseline        = baseline
#
#     # ------------------------------------------------------------------
#     # main entry
#     # ------------------------------------------------------------------
#     def _do_process(self, session: "Session") -> None:
#         HbO = session.processed_data["HbO"]      # (T, C)
#         HbR = session.processed_data["HbR"]
#         fs  = globals().get("fs", 3.4722)        # Hz; already defined above
#         C   = HbO.shape[1]                       # S·D channels
#
#         # time axis for the extracted HRF
#         L         = int(round((self.tmax - self.tmin) * fs))          # samples per epoch
#         time_axis = np.arange(L) / fs + self.tmin                     # seconds
#
#         hrf_HbO: Dict[str,np.ndarray] = {}
#         hrf_HbR: Dict[str,np.ndarray] = {}
#
#         for cond in session.protocol.tasks:
#             # ---- collect onset indices ---------------------------------
#             onsets = [start for start, _, label in session.protocol.intervals
#                       if label == cond]
#             segments_O: List[np.ndarray] = []
#             segments_R: List[np.ndarray] = []
#
#             for t0 in onsets:
#                 start = int(round((t0 + self.tmin) * fs))
#                 end   = start + L
#                 if start < 0 or end > HbO.shape[0]:        # skip truncated epochs
#                     continue
#
#                 seg_O = HbO[start:end, :].copy()           # (L, C)
#                 seg_R = HbR[start:end, :].copy()
#
#                 # ---- baseline‑correct each trial -----------------------
#                 b0 = int(round((t0 + self.baseline[0]) * fs))
#                 b1 = int(round((t0 + self.baseline[1]) * fs))
#                 if b0 >= 0 and b1 > b0:
#                     base_O = HbO[b0:b1, :].mean(axis=0)
#                     base_R = HbR[b0:b1, :].mean(axis=0)
#                     seg_O -= base_O
#                     seg_R -= base_R
#
#                 segments_O.append(seg_O)
#                 segments_R.append(seg_R)
#
#             # ---- grand‑average over trials -----------------------------
#             if segments_O:                       # at least one good epoch
#                 hrf_HbO[cond] = np.mean(np.stack(segments_O, axis=0), axis=0)
#                 hrf_HbR[cond] = np.mean(np.stack(segments_R, axis=0), axis=0)
#
#         # stash results
#         session.processed_data["hrf"] = {
#             "time": time_axis,          # (L,)
#             "HbO":  hrf_HbO,
#             "HbR":  hrf_HbR,
#         }


def detect_bad_channels(raw_wl1: np.ndarray,
                        raw_wl2: np.ndarray,
                        cv_high_thresh: float = 0.15,
                        cv_low_thresh: float  = 0.001
                        ) -> list[int]:
    """
    Identify bad source–detector channels based on coefficient of variation
    in the *raw* wavelength intensities.

    Parameters
    ----------
    raw_wl1, raw_wl2 : ndarray
        Either shape (T, S, D) or shape (T, S*D).
    cv_high_thresh : float
        Channels with CV above this are marked bad (too noisy).
    cv_low_thresh : float
        Channels with CV below this are marked bad (flat).

    Returns
    -------
    bad_channels : list of int
        Flat channel indices 0..(S*D-1) to drop.
    """
    # flatten time × channel
    if raw_wl1.ndim == 3:
        T, S, D = raw_wl1.shape
        flat1 = raw_wl1.reshape(T, S * D)
        flat2 = raw_wl2.reshape(T, S * D)
    elif raw_wl1.ndim == 2:
        flat1, flat2 = raw_wl1, raw_wl2
    else:
        raise ValueError(f"Unsupported raw shape {raw_wl1.shape}")

    m1 = flat1.mean(axis=0)
    s1 = flat1.std(axis=0)
    m2 = flat2.mean(axis=0)
    s2 = flat2.std(axis=0)

    cv1 = s1 / np.where(m1 == 0, np.finfo(float).eps, m1)
    cv2 = s2 / np.where(m2 == 0, np.finfo(float).eps, m2)

    bad_mask = (cv1 > cv_high_thresh) | (cv2 > cv_high_thresh) \
            | (cv1 < cv_low_thresh)  | (cv2 < cv_low_thresh)

    return np.where(bad_mask)[0].tolist()


class DetectBadChannels(ProcessingStep):
    """
    Compute bad_channels from the *raw* wl1/wl2 and stash them
    so downstream steps can drop them.
    Must run *before* ConvertWavelengthsToOD!
    """
    def _do_process(self, session: Session) -> None:
        raw1 = session.processed_data["wl1"]
        raw2 = session.processed_data["wl2"]
        session.processed_data["bad_channels"] = detect_bad_channels(raw1, raw2)


class ReconstructDual(ProcessingStep):
    """
    A processing step that reconstructs dual-wavelength fNIRS data using eigenmodes.
    
    This step iterates over each task, finds the optimal regularization parameter
    by maximizing correlation with fMRI data, and reconstructs vertex-wise t-stat maps
    for both HbO and HbR using the dual-wavelength reconstruction approach.
    """
    
    def __init__(self, num_eigenmodes: int):
        """
        Initialize the dual reconstruction step.
        
        Args:
            num_eigenmodes: Number of eigenmodes to use for reconstruction
            use_fmri_prior: Whether to use fMRI data to find optimal regularization parameter
        """
        self.num_eigenmodes = num_eigenmodes

    def _do_process(self, session: Session) -> None:
        """
        Reconstruct the data for each task using dual-wavelength approach.
        
        Args:
            session: The session to process (modified in-place)
        """
        # Get mesh and eigenmode data
        vertices = session.patient.mesh.vertices
        if session.patient.mesh.eigenmodes is None:
            raise ValueError("Mesh has no eigenmodes. Please ensure eigenmodes are loaded.")
        
        # Truncate eigenmodes to the specified number (skip first 2 like in reconstruction script)
        start_idx = 2  # Skip first 2 eigenmodes like in reconstruction script
        end_idx = start_idx + self.num_eigenmodes
        phi = np.array([e for e in session.patient.mesh.eigenmodes[start_idx:end_idx]]).T  # Shape: (n_vertices, n_eigenmodes)
        eigenvals = np.array([e.eigenvalue for e in session.patient.mesh.eigenmodes[start_idx:end_idx]])
        # Set first eigenvalue to 0.0 like in reconstruction script
        eigenvals[0] = 0.0
        
        # Sample Jacobians at vertices
        vertex_jacobian_wl1 = session.jacobians[0].sample_at_vertices(vertices)
        vertex_jacobian_wl2 = session.jacobians[1].sample_at_vertices(vertices)

        # Compute Bmn matrices
        Bmn_wl1 = self.compute_Bmn(vertex_jacobian_wl1, phi)
        Bmn_wl2 = self.compute_Bmn(vertex_jacobian_wl2, phi)

        # Get t-stat data
        t_HbO_data = session.processed_data["t_HbO"]
        t_HbR_data = session.processed_data["t_HbR"]
        
        # Initialize reconstructed data dictionaries
        session.processed_data["t_HbO_reconstructed"] = {}
        session.processed_data["t_HbR_reconstructed"] = {}
        
        # Iterate over each task
        for task in session.protocol.tasks:
            print(f"Processing task: {task}")
            
            # Get t-stats for this task
            y_wl1 = t_HbO_data[task]  # Shape: (S, D)
            y_wl2 = t_HbR_data[task]  # Shape: (S, D)
            
            # Find optimal regularization parameter
            from lys.utils.mri_tstat import get_mri_tstats
            fmri_tstats = get_mri_tstats(session.patient.name, task)
            best_param = self.get_best_reg_param_dual(
                Bmn_wl1, Bmn_wl2, y_wl1, y_wl2, phi, eigenvals, 
                vertices, fmri_tstats
            )
            print(f"  Optimal regularization parameter: {best_param:.6f}")
            
            # Reconstruct using dual-wavelength approach
            reconstructed_HbO = self.reconstruct_dual(
                Bmn_wl1, Bmn_wl2, y_wl1, y_wl2, phi, best_param, eigenvals
            )
            
            # Store reconstructed data
            session.processed_data["t_HbO_reconstructed"][task] = reconstructed_HbO
            session.processed_data["t_HbR_reconstructed"][task] = reconstructed_HbO  # Note: reconstruct_dual only returns HbO
        
        # Clean up intermediate data
        del session.processed_data["t_HbO"]
        del session.processed_data["t_HbR"]



    def get_best_reg_param_dual(self, Bmn_wl1, Bmn_wl2, y_sd, y_sd_HbR, phi, eigenvals, vertices, fmri_tstats):
        """
        Find the regularization parameter that maximizes correlation with fMRI.
        
        Args:
            Bmn_wl1: Bmn matrix for wavelength 1
            Bmn_wl2: Bmn matrix for wavelength 2
            y_sd: HbO t-stats for the task (S, D)
            y_sd_HbR: HbR t-stats for the task (S, D)
            phi: Eigenmode matrix (n_vertices, n_eigenmodes)
            eigenvals: Eigenvalues array
            vertices: Mesh vertices
            fmri_tstats: fMRI t-stats for comparison
            
        Returns:
            Optimal regularization parameter
        """
        scores = []
        params = np.logspace(-5, 5, 65)
        
        for regularisation_param in params:
            X = self.reconstruct_dual(Bmn_wl1, Bmn_wl2, y_sd, y_sd_HbR, phi, regularisation_param, eigenvals)
            fnirs_tstats = np.array([X[ix] for ix in range(len(vertices))])

            score = np.corrcoef(fnirs_tstats, fmri_tstats)[0, 1]
            scores.append(score)

        best_param_ix = np.where(np.array(scores) == max(scores))[0]
        best_param = params[best_param_ix]
        assert len(best_param) > 0, "NaN scores, probably all t-stats are zero"
        return best_param[0]  # Return scalar value

    def reconstruct_dual(self, Bmn_wl1, Bmn_wl2,
                        y_wl1, y_wl2,
                        phi, lam, eigvals,
                        *,                        # keyword-only
                        ext=(586, 1548.52, 1058, 691.32),
                        w=(1., 1.),
                        rho=-0.6,                 # expected HbR/HbO ratio
                        mu=0.1):                  # coupling strength
        """
        Soft-tied HbO / HbR inversion: min ‖Bα−y‖² + λ‖Λ½α‖² + μ‖α_R−ρ α_O‖²

        Parameters
        ----------
        Bmn_wl1: Bmn matrix for wavelength 1
        Bmn_wl2: Bmn matrix for wavelength 2
        y_wl1: HbO t-stats for the task (S, D)
        y_wl2: HbR t-stats for the task (S, D)
        phi: Eigenmode matrix (n_vertices, n_eigenmodes)
        lam: spatial Tikhonov weight λ
        eigvals: Eigenvalues array
        ext: Extinction coefficients (εO1, εR1, εO2, εR2)
        w: Per-wavelength noise weights (1/σ)
        rho: expected α_R / α_O ratio (≈ –0.6)
        mu: coupling weight μ

        Returns
        -------
        x_HbO: vertex-wise HbO map
        """
        εO1, εR1, εO2, εR2 = ext
        n = eigvals.size
        if mu is None:
            mu = lam

        # wavelength blocks
        B1 = np.hstack((εO1*Bmn_wl1, εR1*Bmn_wl1))
        B2 = np.hstack((εO2*Bmn_wl2, εR2*Bmn_wl2))
        B = np.vstack((w[0]*B1, w[1]*B2))

        y = np.concatenate((w[0]*y_wl1.flatten(order='F'),
                           w[1]*y_wl2.flatten(order='F')))

        # regularisers
        Λ = np.diag(np.tile(eigvals, 2))            # spatial
        I = np.eye(n)
        C = mu * np.block([[I, -rho*I],
                          [-rho*I, rho**2 * I]])    # coupling

        A = B.T @ B + lam*Λ + C
        α = np.linalg.solve(A, B.T @ y)

        α_O, α_R = α[:n], α[n:]
        return phi @ α_O

    def compute_Bmn(self, vertex_jacobian: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Compute Bmn matrix from vertex Jacobian and eigenmodes.
        
        Parameters
        ----------
        vertex_jacobian : (N, S, D) array
            Jacobian evaluated at the selected vertices
            (N = n_vertices, S = n_sources, D = n_detectors).

        phi : (N, M) array
            Eigen-mode basis at those vertices (M = self.num_eigenmodes).

        Returns
        -------
        Bmn : (S·D, M) array
            Flattened in *Fortran* (column-major) order so all sources for the first
            detector come first, matching the original orientation.
        """
        # Contract the vertex dimension (N) → result is (S, D, M)
        # einsum keeps everything in-core and is ~ the same speed as tensordot.
        B_sdm = np.einsum('nsd,nm->sdm', vertex_jacobian, phi)

        # Reshape to (S·D, M) with Fortran ordering so sources vary fastest.
        S, D, M = B_sdm.shape
        Bmn = B_sdm.reshape(S * D, M, order='F')
        return Bmn


#OLD
# class ReconstructDualWithoutBadChannels(ProcessingStep):
#     """
#     Dual‐wavelength eigenmode reconstruction that first drops any flat/noisy
#     channels flagged in session.processed_data["bad_channels"].
#     """
#     def __init__(self, num_eigenmodes: int,
#                  lambda_selection: str = "lcurve"):   # "lcurve"  or  "corr"
#         self.num_eigenmodes   = num_eigenmodes
#         self.lambda_selection = lambda_selection      # ← new
#
#     # ------------------------------------------------------------------
#     # Pick parameter that maximizes correlation with fMRI
#     # ------------------------------------------------------------------
#     def _find_best_lambda_corr(self, B1, B2, y1, y2,
#                                phi, eigvals, verts, fmri, bad):
#         params  = np.logspace(-5, 5, 65)
#         best_r, best_lam = -np.inf, params[0]
#         for lam in params:
#             X = self._reconstruct(B1, B2, y1, y2, phi, lam, eigvals,
#                                   bad_channels=bad)
#             r = np.corrcoef(X, fmri)[0, 1]
#             if r > best_r:
#                 best_r, best_lam = r, lam
#         return best_lam
#
#     # ------------------------------------------------------------------
#     # NEW L‑curve picker (helpers embedded for brevity)
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _curv(xm, x, xp, ym, y, yp):
#         dx1, dx2 = x - xm, xp - x;  dy1, dy2 = y - ym, yp - y
#         dx = .5*(dx1+dx2);  dy = .5*(dy1+dy2)
#         ddx = dx2 - dx1;    ddy = dy2 - dy1
#         return abs(ddx*dy - ddy*dx) / ((dx*dx+dy*dy)**1.5 + 1e-12)
#
#     def _find_best_lambda_lcurve(self, B1, B2, y1, y2,
#                                  phi, eigvals, verts, fmri, bad):
#         # build dual system once (same as in _reconstruct)
#         εO1, εR1, εO2, εR2 = 586, 1548.52, 1058, 691.32
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((M1, M2))
#         y  = np.concatenate((y1.flatten(order='F'), y2.flatten(order='F')))
#         if bad:                              # drop rows for bad channels twice
#             m      = B1.shape[0]
#             mask   = np.ones(B.shape[0], bool)
#             mask[bad] = False;  mask[m+np.array(bad)] = False
#             B, y  = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size; I = np.eye(n); rho, mu = -0.6, 0.1
#         C = mu*np.block([[I, -rho*I], [-rho*I, rho**2*I]])
#
#         lam_grid = np.logspace(-5, 5, 65)
#         res, sol = [], []
#         for lam in lam_grid:
#             A   = B.T@B + lam*Λ + C
#             α   = np.linalg.solve(A, B.T@y)
#             res.append(np.linalg.norm(B@α - y))
#             sol.append(np.linalg.norm(np.sqrt(Λ)@α))
#
#         log_r, log_s = np.log10(res), np.log10(sol)
#         curv = [0]+[self._curv(log_r[i-1],log_r[i],log_r[i+1],
#                                log_s[i-1],log_s[i],log_s[i+1])
#                     for i in range(1,len(lam_grid)-1)]+[0]
#         return float(lam_grid[int(np.argmax(curv))])
#
#     # ------------------------------------------------------------------
#     # choose picker based on user flag
#     # ------------------------------------------------------------------
#     def _find_best_lambda(self, *args, **kw):
#         pick = self._find_best_lambda_lcurve if self.lambda_selection=="lcurve" \
#                else self._find_best_lambda_corr
#         return pick(*args, **kw)
#
#
#     def _do_process(self, session: "Session") -> None:
#         # --- 1) load mesh + eigenmodes ---
#         verts = session.patient.mesh.vertices
#         if session.patient.mesh.eigenmodes is None:
#             raise ValueError("Mesh has no eigenmodes.")
#         start, end = 2, 2 + self.num_eigenmodes
#         # stack then transpose → (n_vertices, M)
#         phi = np.vstack(session.patient.mesh.eigenmodes[start:end]).T
#         eigvals = np.array([e.eigenvalue for e in session.patient.mesh.eigenmodes[start:end]])
#         eigvals[0] = 0.0
#
#         # --- 2) sample Jacobians + build Bmn blocks ---
#         vj1 = session.jacobians[0].sample_at_vertices(verts)
#         vj2 = session.jacobians[1].sample_at_vertices(verts)
#         B1 = self._compute_Bmn(vj1, phi)
#         B2 = self._compute_Bmn(vj2, phi)
#
#         # --- 3) grab bad_channels (must have run DetectBadChannels first) ---
#         bad = session.processed_data.get("bad_channels", [])
#
#         # --- 4) iterate tasks ---
#         tO = session.processed_data["t_HbO"]
#         tR = session.processed_data["t_HbR"]
#         session.processed_data["t_HbO_reconstructed"] = {}
#         session.processed_data["t_HbR_reconstructed"] = {}
#
#         from lys.utils.mri_tstat import get_mri_tstats
#         for task in session.protocol.tasks:
#             y1 = tO[task]
#             y2 = tR[task]
#             fmri = get_mri_tstats(session.patient.name, task)
#
#             lam = self._find_best_lambda(B1, B2, y1, y2, phi, eigvals, verts, fmri, bad)
#             print(f"  Optimal regularization parameter: {lam:.6f}\n")
#             rec = self._reconstruct(
#                 B1, B2, y1, y2, phi, lam, eigvals,
#                 bad_channels=bad
#             )
#
#             session.processed_data["t_HbO_reconstructed"][task] = rec[0]
#             session.processed_data["t_HbR_reconstructed"][task] = rec[1]
#
#         # cleanup
#         del session.processed_data["t_HbO"]
#         del session.processed_data["t_HbR"]
#
#     def _compute_Bmn(self, vj: np.ndarray, phi: np.ndarray) -> np.ndarray:
#         """
#         Contract Jacobian (N_vertices, S, D) with phi (N_vertices, M)
#         → Bmn (S*D, M) in Fortran order.
#         """
#         B = np.einsum("nsd,nm->sdm", vj, phi)
#         S, D, M = B.shape
#         return B.reshape(S * D, M, order="F")
#
#     def _reconstruct(self,
#                      B1, B2,
#                      y1, y2,
#                      phi, lam, eigvals,
#                      *,
#                      ext=(586, 1548.52, 1058, 691.32),
#                      w=(1., 1.), rho=-0.6, mu=0.1,
#                      eps=0.5,  # ← tolerance for eigen‑mode filter
#                      bad_channels: list[int] = []):
#         """
#         Dual‑wavelength inversion with optional *eigen‑mode* anti‑correlation filter.
#
#         eps : float or None
#             If a float (0‥1), zero‑out any eigen‑mode i whose mismatch
#             |α_R[i] – ρ α_O[i]| / √(α_O²+α_R²)  >  eps.
#             Set to None to disable the filter.
#         """
#         εO1, εR1, εO2, εR2 = ext
#         # ------------ forward model blocks ------------
#         M1 = np.hstack((εO1 * B1, εR1 * B1))
#         M2 = np.hstack((εO2 * B2, εR2 * B2))
#         B = np.vstack((w[0] * M1, w[1] * M2))
#         y = np.concatenate((w[0] * y1.flatten(order="F"),
#                             w[1] * y2.flatten(order="F")))
#
#         # ------------ drop bad channels ------------
#         m = B1.shape[0]
#         if bad_channels:
#             mask = np.ones(B.shape[0], bool)
#             mask[bad_channels] = False
#             mask[m + np.array(bad_channels)] = False
#             B, y = B[mask], y[mask]
#
#         # ------------ regularised solve ------------
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size
#         I = np.eye(n)
#         C = mu * np.block([[rho**2 * I, -rho * I],
#                            [-rho * I, I]])
#
#         α = np.linalg.solve(B.T @ B + lam * Λ + C, B.T @ y)
#         α_O, α_R = α[:n], α[n:]
#
#         # ------------ eigen‑mode filter ------------
#         if eps is not None:
#             mismatch = np.abs(α_R - rho * α_O) / (np.sqrt(α_O ** 2 + α_R ** 2) + 1e-12)
#             keep = mismatch <= eps
#             α_O = α_O * keep  # zero out un‑matched modes
#             α_R = α_R * keep
#
#         # ------------ back‑projection ------------
#         return phi @ α_O, phi @ α_R
#
#     def _reconstruct_withoutanticorrelationcheck(self,
#                      B1, B2,
#                      y1, y2,
#                      phi, lam, eigvals,
#                      *,
#                      ext=(586, 1548.52, 1058, 691.32),
#                      w=(1., 1.), rho=-0.6, mu=0,
#                      bad_channels: list[int] = []
#                      ) -> np.ndarray:
#         """
#         Soft‐tied dual‐wavelength inversion, dropping bad_channels in both B and y.
#         Returns vertex‐wise HbO map.
#         """
#         εO1, εR1, εO2, εR2 = ext
#         # build dual forward model
#         M1 = np.hstack((εO1 * B1, εR1 * B1))
#         M2 = np.hstack((εO2 * B2, εR2 * B2))
#         B = np.vstack((w[0] * M1, w[1] * M2))
#         y = np.concatenate((w[0] * y1.flatten(order="F"),
#                             w[1] * y2.flatten(order="F")))
#
#         # drop bad rows
#         m = B1.shape[0]
#         if bad_channels:
#             mask = np.ones(B.shape[0], dtype=bool)
#             mask[bad_channels] = False
#             mask[m + np.array(bad_channels)] = False
#             B = B[mask, :]
#             y = y[mask]
#
#         # regularisation
#         Λ = np.diag(np.tile(eigvals, 2))
#         I = np.eye(eigvals.size)
#         C = mu * np.block([[I, -rho * I],
#                            [-rho * I, rho ** 2 * I]])
#         A = B.T @ B + lam * Λ + C
#
#         α = np.linalg.solve(A, B.T @ y)
#         return phi @ α[:eigvals.size], phi @ α[eigvals.size:]


# class ReconstructDualWithoutBadChannels(ProcessingStep):
#     """
#     Dual-wavelength eigen-mode reconstruction with bad-channel pruning.
#     Supported hyper-parameter pickers (keyword *lambda_selection*):
#         "corr"   – maximise fNIRS–fMRI correlation            (1-D)
#         "lcurve" – classic 2-term L-curve corner              (1-D)
#         "pareto" – 3-term Pareto-surface corner               (2-D)
#         "gcv"    – two-parameter Generalised Cross-Validation (2-D)
#     All pickers now return a pair (λ*, μ*).  If μ* is None the
#     solver falls back to the default coupling μ=0.1.
#     """
#
#     # ---------- constructor -------------------------------------------------
#     def __init__(self, num_eigenmodes: int, lambda_selection: str = "lcurve"):
#         if lambda_selection not in ("corr", "lcurve", "pareto", "gcv"):
#             raise ValueError("lambda_selection must be "
#                              "'corr', 'lcurve', 'pareto' or 'gcv'")
#         self.num_eigenmodes   = num_eigenmodes
#         self.lambda_selection = lambda_selection
#
#     # =======================================================================
#     # 1-D pickers kept from the original code
#     # =======================================================================
#     def _find_best_lambda_corr(self, B1, B2, y1, y2,
#                                phi, eigvals, verts, fmri, bad):
#         params  = np.logspace(-5, 5, 65)
#         best_r, best = -np.inf, params[0]
#         for lam in params:
#             X, _ = self._reconstruct(B1, B2, y1, y2, phi, lam, eigvals,
#                                      bad_channels=bad)
#             r = np.corrcoef(X, fmri)[0, 1]
#             if r > best_r:
#                 best_r, best = r, lam
#         return best, None          # (λ*, μ*)
#
#     @staticmethod
#     def _curv(xm, x, xp, ym, y, yp):
#         dx1, dx2 = x - xm, xp - x;  dy1, dy2 = y - ym, yp - y
#         dx = .5*(dx1+dx2);  dy = .5*(dy1+dy2)
#         ddx = dx2 - dx1;    ddy = dy2 - dy1
#         return abs(ddx*dy - ddy*dx) / ((dx*dx+dy*dy)**1.5 + 1e-12)
#
#     def _find_best_lambda_lcurve(self, B1, B2, y1, y2,
#                                  phi, eigvals, verts, fmri, bad):
#         εO1, εR1, εO2, εR2 = 586, 1548.52, 1058, 691.32
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((M1, M2))
#         y  = np.concatenate((y1.flatten(order='F'), y2.flatten(order='F')))
#         if bad:
#             m = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False; mask[m+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size; I = np.eye(n); rho, mu = -0.6, 0.1
#         C = mu*np.block([[I, -rho*I], [-rho*I, rho**2*I]])
#
#         lam_grid = np.logspace(-5, 5, 65)
#         res, sol = [], []
#         for lam in lam_grid:
#             α = np.linalg.solve(B.T@B + lam*Λ + C, B.T@y)
#             res.append(np.linalg.norm(B@α - y))
#             sol.append(np.linalg.norm(np.sqrt(Λ)@α))
#
#         log_r, log_s = np.log10(res), np.log10(sol)
#         curv = [0]+[self._curv(log_r[i-1],log_r[i],log_r[i+1],
#                                log_s[i-1],log_s[i],log_s[i+1])
#                     for i in range(1,len(lam_grid)-1)]+[0]
#         return float(lam_grid[int(np.argmax(curv))]), None
#
#     # =======================================================================
#     # 2-D pickers – NEW
#     # =======================================================================
#     def _find_best_param_pareto(self, B1, B2, y1, y2,
#                                 phi, eigvals, verts, fmri, bad,
#                                 lam_grid=np.logspace(-5,5,31),
#                                 mu_grid =np.logspace(-5,5,31),
#                                 ext=(586,1548.52,1058,691.32),
#                                 w=(1.,1.), rho=-.6):
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order='F'),
#                              w[1]*y2.flatten(order='F')))
#         if bad:
#             m = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False; mask[m+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         n = eigvals.size
#         Λ = np.diag(np.tile(eigvals, 2))
#         I = np.eye(n)
#         P = np.block([[ I, -rho*I], [-rho*I, rho**2*I]])
#
#         R = np.empty((len(lam_grid), len(mu_grid)))
#         S = np.empty_like(R)
#         C = np.empty_like(R)
#         for i, lam in enumerate(lam_grid):
#             for j, mu in enumerate(mu_grid):
#                 A  = B.T @ B + lam*Λ + mu*P
#                 α  = np.linalg.solve(A, B.T@y)
#                 R[i,j] = np.linalg.norm(B@α - y)
#                 S[i,j] = np.linalg.norm(np.sqrt(Λ)@α)
#                 C[i,j] = np.linalg.norm(P@α)
#
#         r, s, c = np.log10(R), np.log10(S), np.log10(C)
#         dr_dλ = np.gradient(r, np.log10(lam_grid), axis=0)
#         dr_dμ = np.gradient(r, np.log10(mu_grid),  axis=1)
#         ds_dλ = np.gradient(s, np.log10(lam_grid), axis=0)
#         ds_dμ = np.gradient(s, np.log10(mu_grid),  axis=1)
#         dc_dλ = np.gradient(c, np.log10(lam_grid), axis=0)
#         dc_dμ = np.gradient(c, np.log10(mu_grid),  axis=1)
#
#         tx = np.stack([dr_dλ, ds_dλ, dc_dλ], axis=-1)
#         ty = np.stack([dr_dμ, ds_dμ, dc_dμ], axis=-1)
#         κ  = np.linalg.norm(np.cross(tx, ty), axis=-1) \
#              / (np.linalg.norm(tx, axis=-1)**2 *
#                 np.linalg.norm(ty, axis=-1)**2)**0.5
#         i, j = np.unravel_index(np.argmax(κ), κ.shape)
#         return float(lam_grid[i]), float(mu_grid[j])
#
#     def _find_best_param_gcv(self, B1, B2, y1, y2,
#                              phi, eigvals, verts, fmri, bad,
#                              lam_grid=np.logspace(-5,5,31),
#                              mu_grid =np.logspace(-5,5,31),
#                              ext=(586,1548.52,1058,691.32),
#                              w=(1.,1.), rho=-.6):
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order='F'),
#                              w[1]*y2.flatten(order='F')))
#         if bad:
#             m = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False; mask[m+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         m  = B.shape[0]
#         Λ  = np.diag(np.tile(eigvals, 2))
#         n  = eigvals.size
#         I  = np.eye(n)
#         P  = np.block([[ I, -rho*I], [-rho*I, rho**2*I]])
#
#         best, best_gcv = (lam_grid[0], mu_grid[0]), np.inf
#         for lam in lam_grid:
#             for mu in mu_grid:
#                 A       = B.T@B + lam*Λ + mu*P
#                 A_invBT = np.linalg.solve(A, B.T)
#                 α       = A_invBT @ y
#                 res     = y - B @ α
#                 rss     = res @ res
#                 trH     = np.trace(B @ A_invBT)
#                 #gcv     = rss / (m - trH)**2 if m > trH else np.inf
#                 # inside _find_best_param_gcv  (two lines)
#                 df_target = 0.2 * m  # 20 % effective d.o.f.
#                 penalty = (trH - df_target) ** 2 / m  # new
#
#                 gcv = rss / m ** 2 + penalty  # replace old gcv
#
#                 if gcv < best_gcv:
#                     best_gcv, best = gcv, (lam, mu)
#         return float(best[0]), float(best[1])
#
#     # =======================================================================
#     # selector returning the chosen pair
#     # =======================================================================
#     def _pick_params(self, *args, **kw):
#         sel = self.lambda_selection
#         if   sel == "corr":   f = self._find_best_lambda_corr
#         elif sel == "lcurve": f = self._find_best_lambda_lcurve
#         elif sel == "pareto": f = self._find_best_param_pareto
#         elif sel == "gcv":    f = self._find_best_param_gcv
#         else: raise RuntimeError
#         return f(*args, **kw)
#
#     # =======================================================================
#     # remaining helpers (unchanged) – _compute_Bmn and _reconstruct
#     # =======================================================================
#     def _compute_Bmn(self, vj: np.ndarray, phi: np.ndarray) -> np.ndarray:
#         B = np.einsum("nsd,nm->sdm", vj, phi)
#         S, D, M = B.shape
#         return B.reshape(S * D, M, order="F")
#
#     def _reconstruct(self, B1, B2, y1, y2, phi, lam, eigvals,
#                      *, ext=(586,1548.52,1058,691.32),
#                      w=(1.,1.), rho=-0.6, mu=0.1,
#                      eps=0.5, bad_channels:list[int] = []):
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1))
#         M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order="F"),
#                              w[1]*y2.flatten(order="F")))
#         if bad_channels:
#             m = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad_channels] = False
#             mask[m+np.array(bad_channels)] = False
#             B, y = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size
#         I = np.eye(n)
#         C = mu * np.block([[rho**2*I, -rho*I],
#                            [-rho*I,    I   ]])
#
#         α = np.linalg.solve(B.T@B + lam*Λ + C, B.T@y)
#         α_O, α_R = α[:n], α[n:]
#         # if eps is not None:
#         #     mismatch = np.abs(α_R - rho*α_O)/(np.sqrt(α_O**2+α_R**2)+1e-12)
#         #     keep = mismatch <= eps
#         #     α_O, α_R = α_O*keep, α_R*keep
#         return phi @ α_O, phi @ α_R
#
#     # =======================================================================
#     # main entry point (same logic, now two parameters)
#     # =======================================================================
#     def _do_process(self, session:"Session") -> None:
#         verts = session.patient.mesh.vertices
#         if session.patient.mesh.eigenmodes is None:
#             raise ValueError("Mesh has no eigenmodes.")
#         s, e = 2, 2+self.num_eigenmodes
#         phi   = np.vstack(session.patient.mesh.eigenmodes[s:e]).T
#         eigvals = np.array([em.eigenvalue for em in
#                             session.patient.mesh.eigenmodes[s:e]])
#         eigvals[0] = 0.0
#
#         vj1 = session.jacobians[0].sample_at_vertices(verts)
#         vj2 = session.jacobians[1].sample_at_vertices(verts)
#         B1  = self._compute_Bmn(vj1, phi)
#         B2  = self._compute_Bmn(vj2, phi)
#         bad = session.processed_data.get("bad_channels", [])
#
#         tO = session.processed_data["t_HbO"]
#         tR = session.processed_data["t_HbR"]
#         session.processed_data["t_HbO_reconstructed"] = {}
#         session.processed_data["t_HbR_reconstructed"] = {}
#
#         from lys.utils.mri_tstat import get_mri_tstats
#         for task in session.protocol.tasks:
#             y1, y2 = tO[task], tR[task]
#             fmri   = get_mri_tstats(session.patient.name, task)
#
#             lam, mu = self._pick_params(B1, B2, y1, y2,
#                                         phi, eigvals, verts, fmri, bad)
#             if mu is None:   # 1-D pickers
#                 mu = 0.1
#             print(f"Task {task:<15}  λ*={lam:.3g}  μ*={mu:.3g}")
#
#             HbO, HbR = self._reconstruct(B1, B2, y1, y2, phi,
#                                          lam, eigvals, mu=mu,
#                                          bad_channels=bad)
#             session.processed_data["t_HbO_reconstructed"][task] = HbO
#             session.processed_data["t_HbR_reconstructed"][task] = HbR
#
#         del session.processed_data["t_HbO"]
#         del session.processed_data["t_HbR"]
# class ReconstructDualWithoutBadChannels(ProcessingStep):
#     """
#     Dual-wavelength eigen-mode reconstruction with bad-channel pruning.
#
#     lambda_selection choices
#         "corr"     – maximise fNIRS–fMRI correlation (1-D λ)
#         "lcurve"   – 2-term L-curve corner           (1-D λ)
#         "pareto"   – 3-term Pareto curvature         (2-D λ,μ)
#         "gcv"      – two-parameter GCV               (2-D λ,μ)
#         "evidence" – Bayesian evidence (type-II ML)  (1-D λ)
#
#     If `mu_fixed` is supplied μ never varies – every picker reduces to a
#     1-D search in λ.
#     """
#
#     # ------------------------------------------------------------------ #
#     # constructor
#     # ------------------------------------------------------------------ #
#     def __init__(self,
#                  num_eigenmodes: int,
#                  lambda_selection: str = "lcurve",
#                  mu_fixed: float | None = None):
#         if lambda_selection not in ("corr", "lcurve", "pareto", "gcv", "evidence"):
#             raise ValueError("lambda_selection must be "
#                              "'corr', 'lcurve', 'pareto', 'gcv' or 'evidence'")
#         self.num_eigenmodes   = num_eigenmodes
#         self.lambda_selection = lambda_selection
#         self.mu_fixed         = mu_fixed        # None → free μ, else constant
#
#     # ------------------------------------------------------------------ #
#     # helpers
#     # ------------------------------------------------------------------ #
#     def _mu(self) -> float:
#         """Current fixed-μ value or the historical default 0.1."""
#         return 0.1 if self.mu_fixed is None else self.mu_fixed
#
#     @staticmethod
#     def _curv(xm, x, xp, ym, y, yp):
#         dx1, dx2 = x - xm, xp - x;  dy1, dy2 = y - ym, yp - y
#         dx = .5*(dx1+dx2);  dy = .5*(dy1+dy2)
#         ddx = dx2 - dx1;    ddy = dy2 - dy1
#         return abs(ddx*dy - ddy*dx) / ((dx*dx+dy*dy)**1.5 + 1e-12)
#
#     # ------------------------------------------------------------------ #
#     # 1-D pickers (λ only)
#     # ------------------------------------------------------------------ #
#     def _find_best_lambda_corr(self, B1, B2, y1, y2,
#                                phi, eigvals, verts, fmri, bad):
#         lam_grid = np.logspace(-5, 5, 65)
#         best_r, best_lam = -np.inf, lam_grid[0]
#         mu = self._mu()
#         for lam in lam_grid:
#             X, _ = self._reconstruct(B1, B2, y1, y2, phi, lam, eigvals,
#                                      mu=mu, bad_channels=bad)
#             r = np.corrcoef(X, fmri)[0, 1]
#             if r > best_r:
#                 best_r, best_lam = r, lam
#         return best_lam, mu
#
#     def _find_best_lambda_lcurve(self, B1, B2, y1, y2,
#                                  phi, eigvals, verts, fmri, bad):
#         εO1, εR1, εO2, εR2 = 586, 1548.52, 1058, 691.32
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((M1, M2))
#         y  = np.concatenate((y1.flatten(order='F'), y2.flatten(order='F')))
#         if bad:
#             m = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False;  mask[m+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size; I = np.eye(n)
#         mu = self._mu(); rho = -0.6
#         C = mu*np.block([[I, -rho*I], [-rho*I, rho**2*I]])
#
#         lam_grid = np.logspace(-5, 5, 65)
#         res, sol = [], []
#         for lam in lam_grid:
#             α = np.linalg.solve(B.T@B + lam*Λ + C, B.T@y)
#             res.append(np.linalg.norm(B@α - y))
#             sol.append(np.linalg.norm(np.sqrt(Λ)@α))
#
#         log_r, log_s = np.log10(res), np.log10(sol)
#         curv = [0]+[self._curv(log_r[i-1],log_r[i],log_r[i+1],
#                                log_s[i-1],log_s[i],log_s[i+1])
#                     for i in range(1,len(lam_grid)-1)]+[0]
#         return float(lam_grid[int(np.argmax(curv))]), mu
#
#     # ------------------------------------------------------------------ #
#     # 2-D pickers (λ,μ) – collapse to 1-D if mu_fixed set
#     # ------------------------------------------------------------------ #
#     def _find_best_param_gcv(self, B1, B2, y1, y2,
#                              phi, eigvals, verts, fmri, bad,
#                              lam_grid=np.logspace(-5,5,31),
#                              mu_grid =np.logspace(-5,5,31),
#                              ext=(586,1548.52,1058,691.32),
#                              w=(1.,1.), rho=-.6):
#         if self.mu_fixed is not None:
#             mu_grid = np.asarray([self.mu_fixed])
#
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order='F'),
#                              w[1]*y2.flatten(order='F')))
#         if bad:
#             m0 = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False; mask[m0+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         m  = B.shape[0]
#         Λ  = np.diag(np.tile(eigvals, 2))
#         n  = eigvals.size
#         I  = np.eye(n); best, best_gcv = (lam_grid[0], mu_grid[0]), np.inf
#
#         for lam in lam_grid:
#             for mu in mu_grid:
#                 P  = np.block([[ I, -rho*I], [-rho*I, rho**2*I]]) * mu
#                 A  = B.T@B + lam*Λ + P
#                 A_invBT = np.linalg.solve(A, B.T)
#                 α   = A_invBT @ y
#                 res = y - B @ α
#                 rss = res @ res
#                 trH = np.trace(B @ A_invBT)
#                 if m <= trH:      # avoid division by zero / negative
#                     continue
#                 gcv = rss / (m - trH)**2
#                 if gcv < best_gcv:
#                     best_gcv, best = gcv, (lam, mu)
#         return float(best[0]), float(best[1])
#
#     def _find_best_param_pareto(self, B1, B2, y1, y2,
#                                 phi, eigvals, verts, fmri, bad,
#                                 lam_grid=np.logspace(-5,5,31),
#                                 mu_grid =np.logspace(-5,5,31),
#                                 ext=(586,1548.52,1058,691.32),
#                                 w=(1.,1.), rho=-.6):
#         if self.mu_fixed is not None:
#             mu_grid = np.asarray([self.mu_fixed])
#
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order='F'),
#                              w[1]*y2.flatten(order='F')))
#         if bad:
#             m0 = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False;  mask[m0+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size; I = np.eye(n)
#
#         R = np.empty((len(lam_grid), len(mu_grid)))
#         S = np.empty_like(R);   Cmat = np.empty_like(R)
#
#         for i, lam in enumerate(lam_grid):
#             for j, mu in enumerate(mu_grid):
#                 P = np.block([[ I, -rho*I], [-rho*I, rho**2*I]]) * mu
#                 A = B.T@B + lam*Λ + P
#                 α = np.linalg.solve(A, B.T@y)
#                 R[i,j] = np.linalg.norm(B@α - y)
#                 S[i,j] = np.linalg.norm(np.sqrt(Λ)@α)
#                 Cmat[i,j] = np.linalg.norm(P@α)
#
#         r, s, c = np.log10(R), np.log10(S), np.log10(Cmat)
#         dr_dλ = np.gradient(r, np.log10(lam_grid), axis=0)
#         dr_dμ = np.gradient(r, np.log10(mu_grid), axis=1)
#         ds_dλ = np.gradient(s, np.log10(lam_grid), axis=0)
#         ds_dμ = np.gradient(s, np.log10(mu_grid), axis=1)
#         dc_dλ = np.gradient(c, np.log10(lam_grid), axis=0)
#         dc_dμ = np.gradient(c, np.log10(mu_grid), axis=1)
#
#         tx = np.stack([dr_dλ, ds_dλ, dc_dλ], axis=-1)
#         ty = np.stack([dr_dμ, ds_dμ, dc_dμ], axis=-1)
#         κ  = np.linalg.norm(np.cross(tx, ty), axis=-1) \
#              / (np.linalg.norm(tx, axis=-1)**2 *
#                 np.linalg.norm(ty, axis=-1)**2)**0.5
#         i, j = np.unravel_index(np.argmax(κ), κ.shape)
#         return float(lam_grid[i]), float(mu_grid[j])
#
#     # ------------------------------------------------------------------ #
#     # Bayesian evidence picker (λ only, μ may be fixed)
#     # ------------------------------------------------------------------ #
#     def _find_best_lambda_evidence(self, B1, B2, y1, y2,
#                                    phi, eigvals, verts, fmri, bad,
#                                    lam_grid=np.logspace(-5,5,65),
#                                    ext=(586,1548.52,1058,691.32),
#                                    w=(1.,1.), rho=-.6):
#         mu = self._mu()
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order='F'),
#                              w[1]*y2.flatten(order='F')))
#         if bad:
#             m0 = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad] = False; mask[m0+np.array(bad)] = False
#             B, y = B[mask], y[mask]
#
#         m   = B.shape[0]
#         Λ   = np.diag(np.tile(eigvals, 2))
#         n   = eigvals.size; I = np.eye(n)
#         P   = mu * np.block([[ I, -rho*I], [-rho*I, rho**2*I]])
#
#         best_logE, best_lam = -np.inf, lam_grid[0]
#         for lam in lam_grid:
#             A = B.T @ B + lam*Λ + P
#             α = np.linalg.solve(A, B.T @ y)
#             residual = y - B @ α
#             rss = residual @ residual
#             Σp = np.linalg.pinv(lam*Λ + P)                  # prior cov, take Moore-Penrose inverse
#             tr_SigmaBB = np.trace(Σp @ (B.T @ B))
#             sigma2 = (rss + tr_SigmaBB) / m
#
#             C  = B @ Σp @ B.T + sigma2 * np.eye(m)
#
#             sign, logdetC = np.linalg.slogdet(C)
#             if sign <= 0:      # numerical issue → skip
#                 continue
#             logE = -0.5*(m*np.log(rss/m) + logdetC)
#             if logE > best_logE:
#                 best_logE, best_lam = logE, lam
#
#         return float(best_lam), mu
#
#     # ------------------------------------------------------------------ #
#     # universal dispatcher
#     # ------------------------------------------------------------------ #
#     def _pick_params(self, *args, **kw):
#         sel = self.lambda_selection
#         if   sel == "corr":     f = self._find_best_lambda_corr
#         elif sel == "lcurve":   f = self._find_best_lambda_lcurve
#         elif sel == "pareto":   f = self._find_best_param_pareto
#         elif sel == "gcv":      f = self._find_best_param_gcv
#         elif sel == "evidence": f = self._find_best_lambda_evidence
#         else:                   raise RuntimeError
#         return f(*args, **kw)   # returns (λ*, μ*)
#
#     # ------------------------------------------------------------------ #
#     # linear-algebra helpers (unchanged except for μ being passed in)
#     # ------------------------------------------------------------------ #
#     def _compute_Bmn(self, vj: np.ndarray, phi: np.ndarray) -> np.ndarray:
#         B = np.einsum("nsd,nm->sdm", vj, phi)
#         S, D, M = B.shape
#         return B.reshape(S * D, M, order="F")
#
#     def _reconstruct(self,
#                      B1, B2, y1, y2, phi, lam, eigvals,
#                      *, ext=(586,1548.52,1058,691.32),
#                      w=(1.,1.), rho=-0.6, mu=0.1,
#                      eps=0.5, bad_channels:list[int] = []):
#         εO1, εR1, εO2, εR2 = ext
#         M1 = np.hstack((εO1*B1, εR1*B1))
#         M2 = np.hstack((εO2*B2, εR2*B2))
#         B  = np.vstack((w[0]*M1, w[1]*M2))
#         y  = np.concatenate((w[0]*y1.flatten(order="F"),
#                              w[1]*y2.flatten(order="F")))
#         if bad_channels:
#             m0 = B1.shape[0]
#             mask = np.ones(B.shape[0], bool)
#             mask[bad_channels] = False
#             mask[m0+np.array(bad_channels)] = False
#             B, y = B[mask], y[mask]
#
#         Λ = np.diag(np.tile(eigvals, 2))
#         n = eigvals.size
#         I = np.eye(n)
#         C = mu * np.block([[rho**2*I, -rho*I],
#                            [-rho*I,    I   ]])
#
#         α = np.linalg.solve(B.T@B + lam*Λ + C, B.T@y)
#         α_O, α_R = α[:n], α[n:]
#
#         if eps is not None:
#             mismatch = np.abs(α_R - rho*α_O)/(np.sqrt(α_O**2+α_R**2)+1e-12)
#             keep = mismatch <= eps
#             α_O, α_R = α_O*keep, α_R*keep
#
#         return phi @ α_O, phi @ α_R
#
#     # ------------------------------------------------------------------ #
#     # main entry point
#     # ------------------------------------------------------------------ #
#     def _do_process(self, session:"Session") -> None:
#         verts = session.patient.mesh.vertices
#         if session.patient.mesh.eigenmodes is None:
#             raise ValueError("Mesh has no eigenmodes.")
#         s, e = 2, 2 + self.num_eigenmodes
#         phi   = np.vstack(session.patient.mesh.eigenmodes[s:e]).T
#         eigvals = np.array([em.eigenvalue for em in
#                             session.patient.mesh.eigenmodes[s:e]])
#         eigvals[0] = 0.0
#
#         vj1 = session.jacobians[0].sample_at_vertices(verts)
#         vj2 = session.jacobians[1].sample_at_vertices(verts)
#         B1  = self._compute_Bmn(vj1, phi)
#         B2  = self._compute_Bmn(vj2, phi)
#         bad = session.processed_data.get("bad_channels", [])
#
#         tO = session.processed_data["t_HbO"]
#         tR = session.processed_data["t_HbR"]
#         session.processed_data["t_HbO_reconstructed"] = {}
#         session.processed_data["t_HbR_reconstructed"] = {}
#
#         from lys.utils.mri_tstat import get_mri_tstats
#         for task in session.protocol.tasks:
#             y1, y2 = tO[task], tR[task]
#             fmri   = get_mri_tstats(session.patient.name, task)
#
#             lam, mu = self._pick_params(B1, B2, y1, y2,
#                                         phi, eigvals, verts, fmri, bad)
#             HbO, HbR = self._reconstruct(
#                 B1, B2, y1, y2, phi, lam, eigvals,
#                 mu=mu, bad_channels=bad
#             )
#             print(f"Task {task:<15}  λ*={lam:.3g}  μ*={mu:.3g}")
#
#             session.processed_data["t_HbO_reconstructed"][task] = HbO
#             session.processed_data["t_HbR_reconstructed"][task] = HbR
#
#         del session.processed_data["t_HbO"]
#         del session.processed_data["t_HbR"]

import numpy as np
from typing import List, Tuple


class ReconstructDualWithoutBadChannels(ProcessingStep):
    """
    Dual‑wavelength eigen‑mode reconstruction with bad‑channel pruning.

    lambda_selection choices
        "manual"      – use user‑supplied λ
        "corr"        – maximise fNIRS–fMRI correlation
        "lcurve"      – 2‑term L‑curve corner (λ only)
        "pareto"      – 3‑term Pareto‑surface curvature (λ, μ)
        "gcv"         – generalised cross‑validation (λ, μ)
        "evidence"    – Bayesian evidence (type‑II ML)        (λ only)
        "sure"        – Stein’s unbiased risk estimate / C_p  (λ only)
        "discrepancy" – Morozov discrepancy principle         (λ only)
        "quasiopt"    – quasi‑optimality (Hanke–Raus)         (λ only)
        "cv"          – K‑fold row‑wise cross‑validation      (λ only)

    If `mu_fixed` is supplied, μ never varies – every picker collapses to
    a 1‑D search in λ.
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(self,
                 num_eigenmodes : int,
                 lambda_selection : str = "lcurve",
                 mu_fixed        : float | None = None,
                 manual_lambda   : float | None = None,
                 noise_sigma     : float | None = None,
                 cv_folds        : int = 5):
        valid = {"manual","corr","lcurve","pareto","gcv","evidence",
                 "sure","discrepancy","quasiopt","cv"}
        if lambda_selection not in valid:
            raise ValueError(f"lambda_selection must be one of {sorted(valid)}")
        if lambda_selection == "manual" and manual_lambda is None:
            raise ValueError("manual_lambda must be given for manual mode")
        if lambda_selection in ("sure","discrepancy") and noise_sigma is None:
            raise ValueError("noise_sigma must be supplied for 'sure' or "
                             "'discrepancy' mode")

        self.num_eigenmodes   = num_eigenmodes
        self.lambda_selection = lambda_selection
        self.mu_fixed         = mu_fixed
        self.manual_lambda    = manual_lambda
        self.noise_sigma      = noise_sigma
        self.cv_folds         = cv_folds

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _mu(self) -> float:
        """Fixed μ value or default 0.1."""
        return 0.1 if self.mu_fixed is None else self.mu_fixed

    @staticmethod
    def _curv(xm, x, xp, ym, y, yp):
        dx1, dx2 = x - xm, xp - x;  dy1, dy2 = y - ym, yp - y
        dx = .5*(dx1+dx2);  dy = .5*(dy1+dy2)
        ddx = dx2 - dx1;    ddy = dy2 - dy1
        return abs(ddx*dy - ddy*dx) / ((dx*dx+dy*dy)**1.5 + 1e-12)

    # ------------------------------------------------------------------ #
    # 0) MANUAL λ
    # ------------------------------------------------------------------ #
    def _find_best_lambda_manual(self,*a,**k):
        return float(self.manual_lambda), self._mu()

    # ------------------------------------------------------------------ #
    # 1) correlation picker
    # ------------------------------------------------------------------ #
    def _find_best_lambda_corr(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad):
        lam_grid = np.logspace(-5,5,65); best_r=-np.inf; best=lam_grid[0]
        mu=self._mu()
        for lam in lam_grid:
            X,_ = self._reconstruct(B1,B2,y1,y2,phi,lam,eig,
                                    mu=mu,bad_channels=bad)
            r=np.corrcoef(X,fmri)[0,1]
            if r>best_r: best_r, best=r, lam
        return float(best), mu

    # ------------------------------------------------------------------ #
    # 2) L‑curve picker
    # ------------------------------------------------------------------ #
    def _find_best_lambda_lcurve(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad):
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        Λ=np.diag(np.tile(eig,2)); n=eig.size; I=np.eye(n)
        mu=self._mu(); rho=-.6
        C=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])

        lam_grid=np.logspace(-5,5,65); res,sol=[],[]
        for lam in lam_grid:
            α=np.linalg.solve(B.T@B+lam*Λ+C,B.T@y)
            res.append(np.linalg.norm(B@α - y))
            sol.append(np.linalg.norm(np.sqrt(Λ)@α))
        lr,ls=np.log10(res),np.log10(sol)
        curv=[0]+[self._curv(lr[i-1],lr[i],lr[i+1],
                             ls[i-1],ls[i],ls[i+1])
                  for i in range(1,len(lam_grid)-1)]+[0]
        lam=float(lam_grid[int(np.argmax(curv))])
        return lam, mu

    # ------------------------------------------------------------------ #
    # 3) GCV picker  (λ,μ)
    # ------------------------------------------------------------------ #
    def _find_best_param_gcv(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                             lam_grid=np.logspace(-5,5,31),
                             mu_grid=np.logspace(-5,5,31),
                             rho=-.6):
        if self.mu_fixed is not None:
            mu_grid=np.asarray([self.mu_fixed])
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        m=B.shape[0]; Λ=np.diag(np.tile(eig,2)); n=eig.size; I=np.eye(n)
        best=(lam_grid[0],mu_grid[0]); best_gcv=np.inf
        for lam in lam_grid:
            for mu in mu_grid:
                P=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])
                A=B.T@B+lam*Λ+P
                A_invBT=np.linalg.solve(A,B.T)
                α=A_invBT@y
                rss=float(y@y - y@B@α)
                trH=float(np.trace(B@A_invBT))
                if trH>=m: continue
                gcv=rss/(m-trH)**2
                if gcv<best_gcv: best_gcv, best = gcv,(lam,mu)
        return float(best[0]), float(best[1])

    # ------------------------------------------------------------------ #
    # 4) Pareto picker  (λ,μ)
    # ------------------------------------------------------------------ #
    def _find_best_param_pareto(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                                lam_grid=np.logspace(-5,5,31),
                                mu_grid=np.logspace(-5,5,31),
                                rho=-.6):
        if self.mu_fixed is not None:
            mu_grid=np.asarray([self.mu_fixed])
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        Λ=np.diag(np.tile(eig,2)); n=eig.size; I=np.eye(n)
        R,S,C=np.empty((len(lam_grid),len(mu_grid))),\
               np.empty_like(R),np.empty_like(R)
        for i,lam in enumerate(lam_grid):
            for j,mu in enumerate(mu_grid):
                P=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])
                A=B.T@B+lam*Λ+P; α=np.linalg.solve(A,B.T@y)
                R[i,j]=np.linalg.norm(B@α - y)
                S[i,j]=np.linalg.norm(np.sqrt(Λ)@α)
                C[i,j]=np.linalg.norm(P@α)
        r,s,c=np.log10(R),np.log10(S),np.log10(C)
        dr_dλ=np.gradient(r,np.log10(lam_grid),axis=0)
        dr_dμ=np.gradient(r,np.log10(mu_grid),axis=1)
        ds_dλ=np.gradient(s,np.log10(lam_grid),axis=0)
        ds_dμ=np.gradient(s,np.log10(mu_grid),axis=1)
        dc_dλ=np.gradient(c,np.log10(lam_grid),axis=0)
        dc_dμ=np.gradient(c,np.log10(mu_grid),axis=1)
        tx=np.stack([dr_dλ,ds_dλ,dc_dλ],axis=-1)
        ty=np.stack([dr_dμ,ds_dμ,dc_dμ],axis=-1)
        κ=np.linalg.norm(np.cross(tx,ty),axis=-1) / \
          (np.linalg.norm(tx,axis=-1)**2 * np.linalg.norm(ty,axis=-1)**2)**0.5
        i,j=np.unravel_index(np.argmax(κ),κ.shape)
        return float(lam_grid[i]), float(mu_grid[j])

    # ------------------------------------------------------------------ #
    # 5) Bayesian evidence picker
    # ------------------------------------------------------------------ #
    def _find_best_lambda_evidence(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                                   lam_grid=np.logspace(-5,5,65),
                                   rho=-.6):
        mu=self._mu()
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        m=B.shape[0]; Λ=np.diag(np.tile(eig,2)); n=eig.size; I=np.eye(n)
        P=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])
        best_logE=-np.inf; best_lam=lam_grid[0]
        for lam in lam_grid:
            Prior=lam*Λ+P; Σp=np.linalg.pinv(Prior)
            A=B.T@B+Prior; α=np.linalg.solve(A,B.T@y)
            rss=float(np.linalg.norm(y-B@α)**2)
            tr_SigmaBB=float(np.trace(Σp@(B.T@B)))
            sigma2=(rss+tr_SigmaBB)/m
            C=B@Σp@B.T+sigma2*np.eye(m)
            sign,ldet=np.linalg.slogdet(C)
            if sign<=0: continue
            logE=-0.5*(m*np.log(sigma2)+ldet)
            if logE>best_logE: best_logE, best_lam = logE, lam
        return float(best_lam), mu

    # ------------------------------------------------------------------ #
    # 6) SURE picker
    # ------------------------------------------------------------------ #
    def _find_best_lambda_sure(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                               lam_grid=np.logspace(-5,5,65),rho=-.6):
        σ2=self.noise_sigma**2; mu=self._mu()
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        m=B.shape[0]; Λ=np.diag(np.tile(eig,2)); n=eig.size; I=np.eye(n)
        best_lam=lam_grid[0]; best_sure=np.inf
        for lam in lam_grid:
            P=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])
            A=B.T@B+lam*Λ+P; A_invBT=np.linalg.solve(A,B.T)
            α=A_invBT@y; rss=float(np.linalg.norm(y-B@α)**2)
            trH=float(np.trace(B@A_invBT))
            sure=rss - m*σ2 + 2*σ2*trH
            if sure<best_sure: best_sure,sure,best_lam=sure,sure,lam
        return float(best_lam), mu

    # ------------------------------------------------------------------ #
    # 7) Morozov discrepancy
    # ------------------------------------------------------------------ #
    def _find_best_lambda_discrepancy(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                                      lam_grid=np.logspace(-5,5,65),rho=-.6):
        target=B1.shape[0]*self.noise_sigma**2
        mu=self._mu()
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        Λ=np.diag(np.tile(eig,2))
        for lam in lam_grid:                # increasing λ ⇒ larger residual
            α,_=self._solve(B,y,Λ,lam,mu)
            rss=float(np.linalg.norm(y-B@α)**2)
            if rss>=target: return float(lam), mu
        return float(lam_grid[-1]), mu

    # ------------------------------------------------------------------ #
    # 8) quasi‑optimality
    # ------------------------------------------------------------------ #
    def _find_best_lambda_quasiopt(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                                   lam_grid=np.logspace(-5,5,65),rho=-.6):
        mu=self._mu()
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        Λ=np.diag(np.tile(eig,2)); sols=[]
        for lam in lam_grid:
            α,_=self._solve(B,y,Λ,lam,mu)
            sols.append(α)
        diffs=[np.linalg.norm(sols[i]-sols[i+1]) for i in range(len(sols)-1)]
        idx=int(np.argmin(diffs)); return float(lam_grid[idx+1]), mu

    # ------------------------------------------------------------------ #
    # 9) K‑fold cross‑validation
    # ------------------------------------------------------------------ #
    def _find_best_lambda_cv(self,B1,B2,y1,y2,phi,eig,verts,fmri,bad,
                             lam_grid=np.logspace(-5,5,31),rho=-.6):
        mu=self._mu()
        εO1,εR1,εO2,εR2=586,1548.52,1058,691.32
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((M1,M2)); y=np.concatenate((y1.flatten('F'),y2.flatten('F')))
        if bad:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad]=False; mask[m0+np.array(bad)]=False; B,y=B[mask],y[mask]
        m=B.shape[0]; Λ=np.diag(np.tile(eig,2))
        idx=np.arange(m); np.random.shuffle(idx)
        folds=np.array_split(idx,self.cv_folds)
        best_lam=lam_grid[0]; best_err=np.inf
        for lam in lam_grid:
            cv_err=0.0
            for f in folds:
                train=np.setdiff1d(idx,f); Bt,yt=B[train],y[train]
                α,_=self._solve(Bt,yt,Λ,lam,mu)
                cv_err += np.linalg.norm(B[f]@α - y[f])**2
            if cv_err<best_err: best_err, best_lam=cv_err,lam
        return float(best_lam), mu

    # ------------------------------------------------------------------ #
    # universal dispatcher
    # ------------------------------------------------------------------ #
    def _pick_params(self,*a,**k):
        d={
            "manual":      self._find_best_lambda_manual,
            "corr":        self._find_best_lambda_corr,
            "lcurve":      self._find_best_lambda_lcurve,
            "pareto":      self._find_best_param_pareto,
            "gcv":         self._find_best_param_gcv,
            "evidence":    self._find_best_lambda_evidence,
            "sure":        self._find_best_lambda_sure,
            "discrepancy": self._find_best_lambda_discrepancy,
            "quasiopt":    self._find_best_lambda_quasiopt,
            "cv":          self._find_best_lambda_cv,
        }
        return d[self.lambda_selection](*a,**k)   # → (λ*, μ*)

    # ------------------------------------------------------------------ #
    # tiny linear‑algebra helpers
    # ------------------------------------------------------------------ #
    def _solve(self,B,y,Λ,lam,mu,rho=-.6)->Tuple[np.ndarray,float]:
        n=Λ.shape[0]//2; I=np.eye(n)
        C=mu*np.block([[I,-rho*I],[-rho*I,rho**2*I]])
        A=B.T@B+lam*Λ+C
        α=np.linalg.solve(A,B.T@y); res=y-B@α
        return α, float(res@res)

    def _compute_Bmn(self,vj:np.ndarray,phi:np.ndarray)->np.ndarray:
        B=np.einsum("nsd,nm->sdm",vj,phi); S,D,M=B.shape
        return B.reshape(S*D,M,order="F")

    def _reconstruct(self,B1,B2,y1,y2,phi,lam,eigvals,*,
                     ext=(586,1548.52,1058,691.32),w=(1.,1.),
                     rho=-.6,mu=0.1,eps=0.5,bad_channels:List[int]=[]):
        εO1,εR1,εO2,εR2=ext
        M1=np.hstack((εO1*B1,εR1*B1));M2=np.hstack((εO2*B2,εR2*B2))
        B=np.vstack((w[0]*M1,w[1]*M2))
        y=np.concatenate((w[0]*y1.flatten('F'),
                          w[1]*y2.flatten('F')))
        if bad_channels:
            m0=B1.shape[0]; mask=np.ones(B.shape[0],bool)
            mask[bad_channels]=False
            mask[m0+np.array(bad_channels)]=False
            B,y=B[mask],y[mask]
        Λ=np.diag(np.tile(eigvals,2)); n=eigvals.size; I=np.eye(n)
        C=mu*np.block([[rho**2*I,-rho*I],[-rho*I, I]])
        α=np.linalg.solve(B.T@B+lam*Λ+C,B.T@y)
        α_O,α_R=α[:n],α[n:]
        if eps is not None:
            mismatch=np.abs(α_R-rho*α_O)/(np.sqrt(α_O**2+α_R**2)+1e-12)
            keep=mismatch<=eps; α_O*=keep; α_R*=keep
        return phi@α_O, phi@α_R

    # ------------------------------------------------------------------ #
    # main entry point
    # ------------------------------------------------------------------ #
    def _do_process(self,session:"Session")->None:
        verts=session.patient.mesh.vertices
        if session.patient.mesh.eigenmodes is None:
            raise ValueError("Mesh has no eigenmodes.")
        s,e=2,2+self.num_eigenmodes
        phi=np.vstack(session.patient.mesh.eigenmodes[s:e]).T
        eigvals=np.array([eigen.eigenvalue
                          for eigen in session.patient.mesh.eigenmodes[s:e]])
        eigvals[0]=0.0

        vj1=session.jacobians[0].sample_at_vertices(verts)
        vj2=session.jacobians[1].sample_at_vertices(verts)
        B1=self._compute_Bmn(vj1,phi)
        B2=self._compute_Bmn(vj2,phi)
        bad=session.processed_data.get("bad_channels",[])

        tO=session.processed_data["t_HbO"]
        tR=session.processed_data["t_HbR"]
        session.processed_data["t_HbO_reconstructed"]={}
        session.processed_data["t_HbR_reconstructed"]={}

        from lys.utils.mri_tstat import get_mri_tstats
        for task in session.protocol.tasks:
            y1,y2=tO[task],tR[task]
            fmri=get_mri_tstats(session.patient.name,task)
            lam,mu=self._pick_params(B1,B2,y1,y2,phi,eigvals,
                                     verts,fmri,bad)
            HbO,HbR=self._reconstruct(B1,B2,y1,y2,phi,lam,eigvals,
                                      mu=mu,bad_channels=bad)
            print(f"Task {task:<15} λ*={lam:.3g} μ*={mu:.3g}")
            session.processed_data["t_HbO_reconstructed"][task]=HbO
            session.processed_data["t_HbR_reconstructed"][task]=HbR

        del session.processed_data["t_HbO"]
        del session.processed_data["t_HbR"]


# ─── convert_to_tstats.py ──────────────────────────────────────────────

# ─── steps.py ──────────────────────────────────────────────────────────────
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
import pywt                                 # only needed for the wavelet mode
from lys.abstract_interfaces.processing_step import ProcessingStep
from typing import Tuple, Dict, Literal, Optional

# ─── steps.py  (append at the end of the file) ──────────────────────────
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import windows
from typing import Dict, List, Optional

from lys.abstract_interfaces.processing_step import ProcessingStep
from lys.objects import Session

# Make sure the canonical_hrf function is in scope; it lives in the same
# module that already defines ExtractHRFviaCanonicalFit.
# from your_module import canonical_hrf

# --------------------------------------------------------------------- #
#  ReconstructSpatioTemporalEvolution – task-wise spatio-temporal solver
# --------------------------------------------------------------------- #
class ReconstructSpatioTemporalEvolutionOLD(ProcessingStep):
    """
    Task-wise dual-wavelength eigen-mode reconstruction of the neural
    *time-course* on the cortical mesh.

    Parameters
    ----------
    num_eigenmodes : int
        Number of Laplace eigen-modes to use (after skipping the first 2).
    lambda_reg : float
        Spatial Tikhonov weight λ that multiplies the eigen-values.
    mu : float, optional
        Coupling strength μ between HbO and HbR (default 0.1).
    rho : float, optional
        Expected HbR/HbO ratio (default −0.6).
    tmin, tmax : float, optional
        Window [s] relative to each onset that is reconstructed
        (default −5 … +30 s).
    window : {"hann", "tukey", None}, optional
        Optional time-domain taper applied before the FFT (default "hann").
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 num_eigenmodes: int,
                 lambda_reg: float = 1e-2,
                 *,
                 mu: float = 0.1,
                 rho: float = -0.6,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 window: Optional[str] = "hann") -> None:
        self.num_eigenmodes = int(num_eigenmodes)
        self.lambda_reg     = float(lambda_reg)
        self.mu             = float(mu)
        self.rho            = float(rho)
        self.tmin           = float(tmin)
        self.tmax           = float(tmax)
        self.window         = window

    # ================================================================== #
    # helpers
    # ================================================================== #
    @staticmethod
    def _compute_Bmn(vertex_jac: np.ndarray,
                     phi: np.ndarray) -> np.ndarray:
        """
        Contract vertex-Jacobian (N,S,D) with eigen-basis (N,M) to
        Bmn (S·D, M), using Fortran ordering so *sources vary fastest*.
        """
        B_sdm = np.einsum("nsd,nm->sdm", vertex_jac, phi)     # (S,D,M)
        S, D, M = B_sdm.shape
        return B_sdm.reshape(S * D, M, order="F")             # (S·D,M)

    # ================================================================== #
    # main entry
    # ================================================================== #
    def _do_process(self, session: "Session") -> None:        # noqa: N802
        # ---------------- raw data ------------------------------------
        HbO = session.processed_data["HbO"]        # (T,C)
        HbR = session.processed_data["HbR"]
        fs  = globals().get("fs", 3.4722)          # Hz
        C_tot = HbO.shape[1]                       # S·D channels

        # ---------------- eigen-basis ---------------------------------
        verts = session.patient.mesh.vertices
        eig_modes = session.patient.mesh.eigenmodes
        if eig_modes is None:
            raise RuntimeError("Mesh eigen-modes are missing.")

        start, end = 2, 2 + self.num_eigenmodes
        phi = np.vstack(eig_modes[start:end]).T                # (V,M)
        eigvals = np.array([em.eigenvalue for em in eig_modes[start:end]],
                           dtype=float)
        eigvals[0] = 0.0                                        # λ₀ = 0

        # ---------------- Jacobian → Bmn -----------------------------
        vj1 = session.jacobians[0].sample_at_vertices(verts)
        vj2 = session.jacobians[1].sample_at_vertices(verts)
        B1  = self._compute_Bmn(vj1, phi)                       # (C_tot,M)
        B2  = self._compute_Bmn(vj2, phi)

        # ---------------- extinction coefficients --------------------
        eps_O1, eps_R1 = 586.0,   1548.52
        eps_O2, eps_R2 = 1058.0,   691.32

        # Wavelength blocks
        M1 = np.hstack((eps_O1 * B1, eps_R1 * B1))              # (C_tot,2M)
        M2 = np.hstack((eps_O2 * B2, eps_R2 * B2))              # (C_tot,2M)
        B_full = np.vstack((M1, M2))                            # (2C_tot,2M)
        BtB = B_full.T @ B_full
        Bt  = B_full.T

        # ---------------- regularisers -------------------------------
        Lambda = np.diag(np.tile(eigvals, 2)) * self.lambda_reg
        n = eigvals.size
        I = np.eye(n)
        C_cpl = self.mu * np.block([[ self.rho ** 2 * I, -self.rho * I],
                                    [-self.rho * I,        I        ]])

        # Constant system matrix (2M × 2M) & its inverse
        A_mat = BtB + Lambda + C_cpl
        A_inv = np.linalg.inv(A_mat)                            # (2M,2M)

        # ---------------- HRF spectrum -------------------------------
        hrf_dict = session.processed_data["hrf"]
        hrf_kernel = canonical_hrf(hrf_dict["time"],
                                   tau   = hrf_dict.get("tau",   1.0),
                                   delay = hrf_dict.get("delay", 0.0),
                                   ratio = hrf_dict.get("ratio", 1/6))

        # ---------------- outputs ------------------------------------
        neural_HbO: Dict[str, np.ndarray] = {}
        neural_HbR: Dict[str, np.ndarray] = {}
        t_axis_out: Optional[np.ndarray] = None

        # =============================================================
        # iterate over tasks
        # =============================================================
        for task in session.protocol.tasks:

            # ---------- collect (onset,offset) pairs -----------------
            blocks = [(s, e) for s, e, lbl in session.protocol.intervals
                      if lbl == task]
            if not blocks:
                continue

            epoch_O: List[np.ndarray] = []
            epoch_R: List[np.ndarray] = []

            # --------------------------------------------------------
            # loop over blocks
            # --------------------------------------------------------
            for on, _ in blocks:
                start_idx = int(round((on + self.tmin) * fs))
                L_epoch   = int(round((self.tmax - self.tmin) * fs))

                if start_idx < 0 or start_idx + L_epoch > HbO.shape[0]:
                    continue   # skip truncated block

                seg_O = HbO[start_idx:start_idx + L_epoch].copy()  # (L,C)
                seg_R = HbR[start_idx:start_idx + L_epoch].copy()

                # ----- baseline correction (same as HRF extraction)
                base_start = int(round((on + self.tmin) * fs))
                base_end   = int(round(on * fs))
                if base_start >= 0 and base_end > base_start:
                    base_O = HbO[base_start:base_end].mean(axis=0)
                    base_R = HbR[base_start:base_end].mean(axis=0)
                    seg_O -= base_O
                    seg_R -= base_R

                # ----- optional taper --------------------------------
                if self.window == "hann":
                    taper = windows.hann(L_epoch, sym=False)[:, None]
                    seg_O *= taper
                    seg_R *= taper
                elif self.window == "tukey":
                    taper = windows.tukey(L_epoch, alpha=0.2)[:, None]
                    seg_O *= taper
                    seg_R *= taper

                # ----- FFT ------------------------------------------
                Y1 = rfft(seg_O, axis=0)                        # (F,C)
                Y2 = rfft(seg_R, axis=0)

                # HRF spectrum (same length n_fft)
                H = rfft(hrf_kernel, n=L_epoch)                 # (F,)
                H_safe = np.where(np.abs(H) < 1e-6, 1e-6, H)    # avoid /0

                # ----- frequency-wise solve ------------------------
                F_bins = Y1.shape[0]
                alpha_O = np.zeros((F_bins, n), dtype=complex)
                alpha_R = np.zeros_like(alpha_O)

                for k in range(F_bins):
                    y_vec = np.concatenate((Y1[k] / H_safe[k],
                                            Y2[k] / H_safe[k]))   # (2C,)

                    alpha_k = A_inv @ (Bt @ y_vec)                 # (2M,)
                    alpha_O[k] = alpha_k[:n]
                    alpha_R[k] = alpha_k[n:]

                # ----- IFFT back to time domain -------------------
                a_O_time = irfft(alpha_O, n=L_epoch, axis=0)       # (L,M)
                a_R_time = irfft(alpha_R, n=L_epoch, axis=0)

                map_O = phi @ a_O_time.T                           # (V,L)
                map_R = phi @ a_R_time.T

                epoch_O.append(map_O)
                epoch_R.append(map_R)

                if t_axis_out is None:
                    t_axis_out = np.arange(L_epoch) / fs + self.tmin

            # ---------- grand-average across blocks -----------------
            if epoch_O:
                neural_HbO[task] = np.mean(epoch_O, axis=0)        # (V,L)
                neural_HbR[task] = np.mean(epoch_R, axis=0)

        # ---------------- stash in session ---------------------------
        session.processed_data["neural_time_axis"] = t_axis_out
        session.processed_data.setdefault("neural_recon", {})
        session.processed_data["neural_recon"]["HbO"] = neural_HbO
        session.processed_data["neural_recon"]["HbR"] = neural_HbR



# class ReconstructSpatioTemporalEvolution(ProcessingStep):
#     """
#     Spatio‑temporal (neural) reconstruction.
#
#     Idea
#     ----
#     In the frequency domain the forward model factorises ::
#
#         Y(f)  =  B · H(f) · α(f)                    (1)
#
#         Y(f)   – channel data  (complex, C × 1)
#         B      – *time‑independent* sensitivity matrix (C × M)
#         H(f)   – complex Fourier spectrum of the HRF (scalar)
#         α(f)   – eigen‑mode coefficients of the **neural** signal (M × 1)
#
#     For every Fourier bin *k* we solve the Tikhonov‑regularised inverse
#
#         α̂_k  =  argmin_α ‖B·H_k·α − Y_k‖²  +  λ‖Λ^{½}α‖²          (2)
#
#     with Λ = diag(eigenvalues) and *λ* a **fixed** hyper‑parameter.
#
#     Optionally, the same principle is applied in a *wavelet* basis in
#     which H(f) is approximated by the mean gain of the HRF spectrum in the
#     pass‑band of each wavelet scale.
#
#     Results
#     -------
#     Adds to ``session.processed_data``
#
#         "HbO_spatiotemporal"  : (T, N_vertices) ndarray  – HbO μM
#         "HbR_spatiotemporal"  : (T, N_vertices) ndarray  – HbR μM
#         "alpha_O" / "alpha_R" : dict(freq_bin → complex α_k)   (optional)
#
#     Parameters
#     ----------
#     num_eigenmodes : int
#         How many Laplace eigen‑modes to keep (same meaning as elsewhere).
#     lam            : float
#         Fixed spatial Tikhonov regularisation weight λ.
#     basis          : {"fourier", "wavelet"}
#         Temporal basis used for the de‑convolution.
#     wavelet        : str | None
#         PyWavelets identifier (only if *basis="wavelet"*).  Defaults to
#         "db4", which has proven robust in fNIRS work :contentReference[oaicite:0]{index=0}.
#     """
#
#     # ------------------------------------------------------------------ #
#     def __init__(self,
#                  num_eigenmodes : int,
#                  lam            : float = 1e-1,
#                  basis          : Literal["fourier", "wavelet"] = "fourier",
#                  wavelet        : Optional[str] = None):
#         self.num_eigenmodes = int(num_eigenmodes)
#         self.lam            = float(lam)
#         self.basis          = basis
#         self.wavelet        = wavelet or "db4"
#
#     # ======================================================================
#     # helpers – spectral solvers
#     # ======================================================================
#     def _solve_frequency_bin(self,
#                              B    : np.ndarray,
#                              Hk   : complex,
#                              Yk   : np.ndarray,
#                              Λ    : np.ndarray,
#                              lam  : float) -> np.ndarray:
#         """
#         Solve (2) for one Fourier bin (complex maths, but real code).
#         """
#         if np.abs(Hk) < 1e-12:                 # HRF has (almost) zero gain
#             return np.zeros(Λ.shape[0], complex)
#
#         BtB = (np.abs(Hk)**2) * (B.T @ B)      #   |H_k|² BᵀB
#         rhs =  Hk.conjugate() * (B.T @ Yk)     # H*_k  Bᵀ Y_k
#         A   = BtB + lam * Λ                    # add Tikhonov term
#
#         return np.linalg.solve(A, rhs)         # complex α_k
#
#     # ======================================================================
#     # main entry
#     # ======================================================================
#     def _do_process(self, session: "Session") -> None:          # noqa: N802
#         # -------- (1) geometry ------------------------------------------------
#         mesh   = session.patient.mesh
#         verts  = mesh.vertices
#         start, end = 2, 2 + self.num_eigenmodes
#         phi    = np.vstack(mesh.eigenmodes[start:end]).T        # (N,M)
#         eigval = np.array([em.eigenvalue
#                            for em in mesh.eigenmodes[start:end]])
#         eigval[0] = 0.0
#         Λ = np.diag(eigval)
#
#         # -------- (2) sensitivity (Jacobian) → Bmn ----------------------------
#         vj = session.jacobians[0].sample_at_vertices(verts)     # HbO only
#         B  = np.einsum("nsd,nm->sdm", vj, phi) \
#                 .reshape(vj.shape[1]*vj.shape[2], -1, order="F")
#
#         # -------- (3) data + HRF ---------------------------------------------
#         HbO = session.processed_data["HbO"]      # shape (T,C)
#         HbR = session.processed_data["HbR"]
#         if HbO.ndim == 3:                        # (T,S,D) → (T,C)
#             HbO = HbO.reshape(HbO.shape[0], -1, order="F")
#             HbR = HbR.reshape(HbR.shape[0], -1, order="F")
#         Y_O = HbO.T                              # (C,T)
#         Y_R = HbR.T
#
#         fs   = globals().get("fs", 3.4722)       # Hz
#
#         hrf = session.processed_data["hrf"]
#         hrf_kernel = canonical_hrf(hrf["time"],
#                                    tau   = hrf.get("tau", 1.0),
#                                    delay = hrf.get("delay", 0.0),
#                                    ratio = hrf.get("ratio", 1/6))
#
#         # ========== A) FOURIER  =============================================
#         if self.basis == "fourier":
#             F = rfft(hrf_kernel, axis=0)                     # (F,)
#             Yf_O = rfft(Y_O, axis=1)                         # (C,F)
#             Yf_R = rfft(Y_R, axis=1)
#
#             αf_O = np.zeros((F.size, eigval.size), complex)
#             αf_R = np.zeros_like(αf_O)
#
#             for k, Hk in enumerate(F):
#                 αf_O[k] = self._solve_frequency_bin(B, Hk, Yf_O[:, k],
#                                                     Λ, self.lam)
#                 αf_R[k] = self._solve_frequency_bin(B, Hk, Yf_R[:, k],
#                                                     Λ, self.lam)
#
#             α_t_O = irfft(αf_O, n=Y_O.shape[1], axis=0)      # (T,M)
#             α_t_R = irfft(αf_R, n=Y_R.shape[1], axis=0)
#
#         # ========== B) WAVELET  ============================================
#         else:   # self.basis == "wavelet"
#             wave = pywt.Wavelet(self.wavelet)
#             max_level = pywt.dwt_max_level(HbO.shape[0], wave.dec_len)
#             coeffs_O  = pywt.wavedec(Y_O, wave, axis=1, level=max_level)
#             coeffs_R  = pywt.wavedec(Y_R, wave, axis=1, level=max_level)
#
#             # pre‑compute band‑centre HRF gains
#             band_gains = []
#             freqs = rfftfreq(HbO.shape[0], d=1/fs)
#             H = rfft(hrf_kernel, axis=0)
#             for level in range(max_level):
#                 f_lo, f_hi = pywt.scale2frequency(wave, level+1)*fs/2, \
#                              pywt.scale2frequency(wave, level)*fs/2
#                 band = (freqs >= f_lo) & (freqs < f_hi)
#                 band_gains.append(np.mean(np.abs(H[band])) if band.any() else 0.)
#
#             α_list_O, α_list_R = [], []
#             for lvl, gain in enumerate(band_gains):
#                 if gain < 1e-9:                               # skip if HRF ≈ 0
#                     α_list_O.append(np.zeros((coeffs_O[lvl].shape[0],
#                                               eigval.size)))
#                     α_list_R.append(np.zeros_like(α_list_O[-1]))
#                     continue
#
#                 α_lvl_O = np.vstack([
#                     self._solve_frequency_bin(B, gain, coeffs_O[lvl][:, t],
#                                               Λ, self.lam).real
#                     for t in range(coeffs_O[lvl].shape[1])
#                 ])
#                 α_lvl_R = np.vstack([
#                     self._solve_frequency_bin(B, gain, coeffs_R[lvl][:, t],
#                                               Λ, self.lam).real
#                     for t in range(coeffs_R[lvl].shape[1])
#                 ])
#                 α_list_O.append(α_lvl_O)
#                 α_list_R.append(α_lvl_R)
#
#             α_t_O = pywt.waverec(α_list_O, wave, axis=0)
#             α_t_R = pywt.waverec(α_list_R, wave, axis=0)
#
#         # -------- (4) back‑projection ----------------------------------------
#         HbO_rec = (phi @ α_t_O.T).T     # (T,N_vertices)
#         HbR_rec = (phi @ α_t_R.T).T
#
#         session.processed_data["HbO_spatiotemporal"] = HbO_rec
#         session.processed_data["HbR_spatiotemporal"] = HbR_rec
#         # store eigen‑coefficients for optional post‑hoc diagnostics
#         session.processed_data["alpha_O"] = α_t_O
#         session.processed_data["alpha_R"] = α_t_R


import numpy as np
from scipy.signal import convolve
from lys.abstract_interfaces.processing_step import ProcessingStep
from lys.objects import Session


# grab the two HRF helpers that already live in your namespace
#   canonical_hrf()            – data-driven, needs (t, tau, delay, ratio)
#   canonical_double_gamma_hrf – SPM-like default
# plus the global sampling rate ‘fs’ that is defined at the top of the file
# ----------------------------------------------------------------------

class ConvertToTStatsWithExtractedHRF(ProcessingStep):
    """
    GLM-based conversion of HbO/HbR to per-condition *t*-statistics.

    Uses the session-specific HRF (τ*, δ*, ρ*) that was found during
    *ExtractHRFviaCanonicalFit*.  If those parameters are missing it falls
    back to the canonical double-gamma HRF.
    """

    # ------------------------------------------------------------------
    def __init__(self, max_iter: int = 20, tol: float = 1e-2):
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    # ==================================================================
    # main entry
    # ==================================================================
    def _do_process(self, session: Session) -> None:

        HbO = session.processed_data["HbO"]  # (T, C)
        HbR = session.processed_data["HbR"]
        fs_ = globals().get("fs", 3.4722)  # Hz
        T, _ = HbO.shape

        # --------------------------------------------------------------
        # 1) build the HRF kernel (data-driven if possible)
        # --------------------------------------------------------------
        hrf_info = session.processed_data.get("hrf", {})
        have_fit = {"tau", "delay", "ratio", "time"} <= hrf_info.keys()

        if have_fit:  # preferred path
            tau = hrf_info["tau"]
            delay = hrf_info["delay"]
            ratio = hrf_info["ratio"]
            t_vec = hrf_info["time"]  # (L,)
            hrf_kernel = canonical_hrf(t_vec,
                                       tau=tau,
                                       delay=delay,
                                       ratio=ratio)
        else:  # backward-compatible
            tr = 1.0 / fs_
            hrf_kernel = canonical_double_gamma_hrf(tr=tr, duration=30.0)
            print("falling back to default gamma, params not extracted\n")

        # --------------------------------------------------------------
        # 2) run channel-wise GLMs → t-maps
        # --------------------------------------------------------------
        t_HbO, t_HbR = self._get_t_stats(
            HbO, HbR, session.protocol,
            fs_, T, hrf_kernel
        )

        session.processed_data["t_HbO"] = t_HbO
        session.processed_data["t_HbR"] = t_HbR
        # leave HbO/HbR intact – other steps may still need them

    # ==================================================================
    # helpers
    # ==================================================================
    def _get_t_stats(self,
                     HbO: np.ndarray,
                     HbR: np.ndarray,
                     protocol,
                     fs: float,
                     n_time: int,
                     hrf_kernel: np.ndarray
                     ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:

        X, conditions = self._create_design_matrix(
            protocol, fs, n_time, hrf_kernel
        )

        n_channels = HbO.shape[1]
        S, D = num_sources, num_detectors

        t_HbO_arr = np.zeros((n_channels, len(conditions)))
        t_HbR_arr = np.zeros_like(t_HbO_arr)

        for ch in range(n_channels):
            beta_o, t_o = self._glm_single_channel_tstats(
                HbO[:, ch], X, self.max_iter, self.tol
            )
            t_HbO_arr[ch] = t_o

            beta_r, t_r = self._glm_single_channel_tstats(
                -HbR[:, ch], X, self.max_iter, self.tol
            )
            t_HbR_arr[ch] = t_r

        # reshape flat-channel vectors → (S, D) matrices
        t_HbO = {cond: t_HbO_arr[:, i].reshape(S, D, order='F')
                 for i, cond in enumerate(conditions)}
        t_HbR = {cond: t_HbR_arr[:, i].reshape(S, D, order='F')
                 for i, cond in enumerate(conditions)}
        return t_HbO, t_HbR

    # ------------------------------------------------------------------
    def _create_design_matrix(self,
                              protocol,
                              fs: float,
                              n_time: int,
                              hrf_kernel: np.ndarray
                              ) -> tuple[np.ndarray, list[str]]:

        conditions = sorted(list(protocol.tasks))
        X = np.zeros((n_time, len(conditions)))

        for i, cond in enumerate(conditions):
            for onset, offset, lbl in protocol.intervals:
                if lbl != cond:
                    continue
                on = int(round(onset * fs))
                off = int(round(offset * fs))
                X[on:off, i] = 1.0

        # HRF convolution (per column)
        for i in range(X.shape[1]):
            X[:, i] = convolve(X[:, i], hrf_kernel, mode='full')[:n_time]

        return X, conditions

    # ------------------------------------------------------------------
    def _glm_single_channel_tstats(self,
                                   y: np.ndarray,
                                   X: np.ndarray,
                                   max_iter: int,
                                   tol: float
                                   ) -> tuple[np.ndarray, np.ndarray]:

        n_time, n_reg = X.shape
        beta = np.zeros(n_reg)
        resid = y.copy()

        for _ in range(max_iter):
            beta_old = beta.copy()

            # AR(1) estimate
            rho = 0.0
            if n_time > 1:
                rho = np.corrcoef(resid[:-1], resid[1:])[0, 1]
                rho = np.clip(rho, -0.99, 0.99)

            # whitening
            W = np.eye(n_time)
            for t in range(1, n_time):
                W[t, t - 1] = -rho
            y_w = W @ y
            X_w = W @ X

            beta = np.linalg.lstsq(X_w, y_w, rcond=None)[0]
            resid = y - X @ beta

            if np.max(np.abs(beta - beta_old)) < tol:
                break

        sigma2 = np.var(resid, ddof=n_reg)
        XtX_inv = np.linalg.inv(X_w.T @ X_w)
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        t_vals = beta / (se_beta + 1e-12)
        return beta, t_vals


class ConvertToTStats(ProcessingStep):
    """
    A processing step that converts HbO and HbR data to t-statistics using GLM analysis.
    
    This step performs AR(1)+IRLS (Autoregressive model with Iteratively Reweighted Least Squares)
    analysis on each channel to compute t-statistics for each experimental condition.
    """
    
    def __init__(self, max_iter: int = 20, tol: float = 1e-2):
        """
        Initialize the t-statistics conversion step.
        
        Args:
            max_iter: Maximum number of iterations for IRLS
            tol: Tolerance for convergence in IRLS
        """
        self.max_iter = max_iter
        self.tol = tol

    def _do_process(self, session: Session) -> None:
        """
        Convert HbO and HbR data to t-statistics.
        
        Args:
            session: The session to process (modified in-place)
        """
        HbO = session.processed_data["HbO"]
        HbR = session.processed_data["HbR"]
        
        t_HbO, t_HbR = self._get_t_stats(HbO, HbR, session.protocol)
        
        session.processed_data["t_HbO"] = t_HbO
        session.processed_data["t_HbR"] = t_HbR
        #del session.processed_data["HbO"]
        #del session.processed_data["HbR"]

    def _get_t_stats(self, HbO: np.ndarray, HbR: np.ndarray, protocol) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Compute t-statistics for HbO and HbR data using GLM analysis.
        
        Args:
            HbO: Oxyhemoglobin data with shape (T, n_channels)
            HbR: Deoxyhemoglobin data with shape (T, n_channels)
            protocol: Protocol object containing timing information
            
        Returns:
            A tuple of two dictionaries, one for t_HbO and one for t_HbR.
            Each dictionary maps a condition name to a numpy array of t-statistics 
            with shape (S, D).
        """
        T, n_channels = HbO.shape  # n_channels = S*D
        S, D = num_sources, num_detectors  # Use the global constants defined at the top
        
        # 1) Build the same design matrix X that you used in GLM
        X, conditions = self._create_design_matrix(protocol, fs, T)
        
        # Initialize outputs
        t_HbO_array = np.zeros((n_channels, len(conditions)))
        t_HbR_array = np.zeros((n_channels, len(conditions)))
        
        # 2) For each channel, run AR(1)+IRLS and compute t-stats
        for ch in range(n_channels):
            y_o = HbO[:, ch]
            y_r = -HbR[:, ch] #-HbR should be fitted by HbO's response function
            
            # Fit and get (beta, tvals) for oxy
            _beta_o, tvals_o = self._glm_single_channel_tstats(
                y_o, X, max_iter=self.max_iter, tol=self.tol
            )
            t_HbO_array[ch, :] = tvals_o
            
            # Fit and get (beta, tvals) for deoxy
            _beta_r, tvals_r = self._glm_single_channel_tstats(
                y_r, X, max_iter=self.max_iter, tol=self.tol
            )
            t_HbR_array[ch, :] = tvals_r

        # Reshape back to (S, D) format for each condition
        t_HbO = {condition: t_HbO_array[:, i].reshape(S, D) for i, condition in enumerate(conditions)}
        t_HbR = {condition: t_HbR_array[:, i].reshape(S, D) for i, condition in enumerate(conditions)}

        return t_HbO, t_HbR

    def _create_design_matrix(self, protocol, fs: float, n_time_points: int) -> tuple[np.ndarray, list[str]]:
        """
        Create a design matrix for GLM analysis based on the protocol.
        
        Args:
            protocol: Protocol object containing timing information
            fs: Sampling frequency in Hz
            n_time_points: Number of time points in the data
            
        Returns:
            A tuple containing:
            - Design matrix with shape (n_time_points, n_conditions)
            - A list of condition names.
        """
        # Get unique task conditions
        conditions = sorted(list(protocol.tasks))
        n_conditions = len(conditions)
        
        # Initialize design matrix
        X = np.zeros((n_time_points, n_conditions))
        
        # Create time vector
        time_vector = np.arange(n_time_points) / fs
        
        # For each condition, create a regressor
        for i, condition in enumerate(conditions):
            # Find all intervals for this condition
            condition_intervals = [
                (start, end) for start, end, label in protocol.intervals 
                if label == condition
            ]
            
            # Create boxcar function for this condition
            for start_time, end_time in condition_intervals:
                start_idx = int(start_time * fs)
                end_idx = int(end_time * fs)
                
                # Ensure indices are within bounds
                start_idx = max(0, start_idx)
                end_idx = min(n_time_points, end_idx)
                
                if start_idx < end_idx:
                    X[start_idx:end_idx, i] = 1.0
        tr = 1.0 / fs
        hrf = canonical_double_gamma_hrf(tr=tr)  # or your choice
        for col_idx in range(n_conditions):
            # convolve
            col_convolved = convolve(X[:, col_idx], hrf, mode='full')[:n_time_points]
            X[:, col_idx] = col_convolved
        return X, conditions

    def _glm_single_channel_tstats(self, y: np.ndarray, X: np.ndarray, max_iter: int = 20, tol: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform GLM analysis with AR(1) noise model and IRLS for a single channel.
        
        Args:
            y: Time series data for a single channel
            X: Design matrix
            max_iter: Maximum number of iterations for IRLS
            tol: Tolerance for convergence in IRLS
            
        Returns:
            Tuple of (beta, tvals) where beta are the regression coefficients and tvals are t-statistics
        """
        n_time_points, n_conditions = X.shape
        
        # Initialize
        beta = np.zeros(n_conditions)
        residuals = y.copy()
        
        for iteration in range(max_iter):
            # Store previous beta for convergence check
            beta_prev = beta.copy()
            
            # Estimate AR(1) coefficient from residuals
            if len(residuals) > 1:
                # Compute AR(1) coefficient using autocorrelation
                autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                rho = np.clip(autocorr, -0.99, 0.99)  # Clip to avoid numerical issues
            else:
                rho = 0.0
            
            # Create AR(1) whitening matrix
            W = np.eye(n_time_points)
            for i in range(1, n_time_points):
                W[i, i-1] = -rho
            
            # Apply whitening
            y_whitened = W @ y
            X_whitened = W @ X
            
            # Solve least squares problem
            beta = np.linalg.lstsq(X_whitened, y_whitened, rcond=None)[0]
            
            # Update residuals
            y_pred = X @ beta
            residuals = y - y_pred
            
            # Check convergence
            if np.max(np.abs(beta - beta_prev)) < tol:
                break
        
        # Compute t-statistics
        # Estimate noise variance from residuals
        sigma2 = np.var(residuals)
        
        # Compute covariance matrix of beta
        XtX_inv = np.linalg.inv(X_whitened.T @ X_whitened)
        cov_beta = sigma2 * XtX_inv
        
        # Compute t-statistics
        tvals = beta / np.sqrt(np.diag(cov_beta))
        
        return beta, tvals

# --------------------------------------------------------------------- #
#  ReconstructSpatioTemporalEvolution
#  (single-alpha formulation – HRFs inside the forward model)
# --------------------------------------------------------------------- #
import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import windows
from lys.abstract_interfaces.processing_step import ProcessingStep
from lys.objects import Session


class ReconstructSpatioTemporalEvolution(ProcessingStep):
    """
    Dual-wavelength, task-wise spatio–temporal reconstruction that
    • assumes *one* neural-activity coefficient vector  α( t )  shared by
      HbO and HbR,
    • embeds the chromophore-specific HRFs directly in the frequency-domain
      forward matrix.

    Stored in  ``session.processed_data`` ::

        "neural_time_axis"             – (L,)
        "neural_recon" : {
            "neural" : {task: (V,L)},  # pure neural activity
            "HbO"    : {task: (V,L)},  # HbO  = neural ⊗ HRF_O
            "HbR"    : {task: (V,L)},  # HbR  = neural ⊗ HRF_R
        }
    """

    # ------------------------------------------------------------------ #
    def __init__(self,
                 num_eigenmodes: int,
                 lambda_reg: float = 1e-2,
                 *,
                 tmin: float = -5.0,
                 tmax: float = 30.0,
                 window: str | None = "hann") -> None:
        self.num_eigenmodes = int(num_eigenmodes)
        self.lambda_reg     = float(lambda_reg)
        self.tmin           = float(tmin)
        self.tmax           = float(tmax)
        self.window         = window

    # ------------------------------------------------------------------ #
    @staticmethod
    def _Bmn(vj: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Jacobian contraction → (S·D, M) with Fortran ordering."""
        B = np.einsum("nsd,nm->sdm", vj, phi)          # (S,D,M)
        S, D, M = B.shape
        return B.reshape(S * D, M, order="F")

    # ================================================================== #
    def _do_process(self, session: "Session") -> None:      # noqa: N802
        fs = globals().get("fs", 3.4722)                    # Hz

        # ---------- raw channel data -----------------------------------
        HbO = session.processed_data["HbO"]                 # (T,C_tot)
        HbR = session.processed_data["HbR"]
        T , C_tot = HbO.shape

        # ---------- eigen-basis & B-matrices ---------------------------
        mesh  = session.patient.mesh
        verts = mesh.vertices
        modes = mesh.eigenmodes
        if modes is None:
            raise RuntimeError("mesh.eigenmodes missing")

        s, e   = 2, 2 + self.num_eigenmodes
        phi    = np.vstack(modes[s:e]).T                    # (V,M)
        eigval = np.array([m.eigenvalue for m in modes[s:e]])
        eigval[0] = 0.0
        M = eigval.size
        Λ = np.diag(eigval) * self.lambda_reg               # (M,M)

        vj1 = session.jacobians[0].sample_at_vertices(verts)
        vj2 = session.jacobians[1].sample_at_vertices(verts)
        B1  = self._Bmn(vj1, phi)                           # (C,M)
        B2  = self._Bmn(vj2, phi)

        # ---------- extinction coefficients ----------------------------
        εO1, εR1 =  586.0, 1548.52
        εO2, εR2 = 1058.0,  691.32

        # ---------- HRFs -----------------------------------------------
        hrf_dat = session.processed_data["hrf"]
        hrf_O   = canonical_hrf(hrf_dat["time"],
                                tau   = hrf_dat.get("tau",   1.0),
                                delay = hrf_dat.get("delay", 0.0),
                                ratio = hrf_dat.get("ratio", 1/6))
        hrf_R   = -0.1 * hrf_O

        # ---------- outputs --------------------------------------------
        neural, HbO_maps, HbR_maps = {}, {}, {}
        t_axis_out = None

        # ===============================================================
        for task in session.protocol.tasks:

            onoff = [(on, off) for on, off, lbl in session.protocol.intervals
                     if lbl == task]
            if not onoff:
                continue

            ep_neu, ep_O, ep_R = [], [], []

            for on, _ in onoff:
                start = int(round((on + self.tmin) * fs))
                L     = int(round((self.tmax - self.tmin) * fs))
                if start < 0 or start + L > T:
                    continue

                seg_O = HbO[start:start+L].copy()
                seg_R = HbR[start:start+L].copy()

                # ---------- baseline correction -----------------------
                b0 = int(round((on + self.tmin) * fs))
                b1 = int(round(on * fs))
                if b0 >= 0 and b1 > b0:
                    seg_O -= HbO[b0:b1].mean(axis=0)
                    seg_R -= HbR[b0:b1].mean(axis=0)

                # ---------- optional taper ----------------------------
                if self.window == "hann":
                    seg_O *= windows.hann(L, sym=False)[:, None]
                    seg_R *= windows.hann(L, sym=False)[:, None]
                elif self.window == "tukey":
                    seg_O *= windows.tukey(L, .2)[:, None]
                    seg_R *= windows.tukey(L, .2)[:, None]

                # ---------- FFT ---------------------------------------
                Y1 = rfft(seg_O, axis=0)                    # (F,C)
                Y2 = rfft(seg_R, axis=0)
                H_O = rfft(hrf_O, n=L)                      # (F,)
                H_R = rfft(hrf_R, n=L)

                F = Y1.shape[0]
                αf = np.zeros((F, M), complex)

                for k in range(F):
                    # forward matrix for this frequency bin
                    g1 = εO1 * H_O[k] + εR1 * H_R[k]        # scalar
                    g2 = εO2 * H_O[k] + εR2 * H_R[k]
                    # if both gains ~0 the data carry no information → skip
                    if abs(g1) < 1e-9 and abs(g2) < 1e-9:
                        continue

                    Bk  = np.vstack((g1 * B1,
                                     g2 * B2))             # (2C,M)
                    yk  = np.concatenate((Y1[k], Y2[k]))    # (2C,)

                    A   = Bk.T @ Bk + Λ
                    αf[k] = np.linalg.solve(A, Bk.T @ yk)   # (M,)

                α_t = irfft(αf, n=L, axis=0).T              # (M,L)

                neu_map = (phi @ α_t).T                     # (L,V)
                HbO_map = neu_map * hrf_O[:, None]
                HbR_map = neu_map * hrf_R[:, None]

                ep_neu.append(neu_map.T)                    # store as (V,L)
                ep_O  .append(HbO_map.T)
                ep_R  .append(HbR_map.T)

                if t_axis_out is None:
                    t_axis_out = np.arange(L) / fs + self.tmin

            if ep_neu:
                neural [task] = np.mean(ep_neu, axis=0)
                HbO_maps[task] = np.mean(ep_O , axis=0)
                HbR_maps[task] = np.mean(ep_R , axis=0)

        # ---------- stash in session ----------------------------------
        session.processed_data["neural_time_axis"] = t_axis_out
        rec = session.processed_data.setdefault("neural_recon", {})
        rec["neural"] = neural
        rec["HbO"]    = HbO_maps
        rec["HbR"]    = HbR_maps




class ConvertWavelengthsToOD(ProcessingStep):
    def _do_process(self, session: Session) -> None:
        """
        Convert the wavelengths to HbO and HbR.
        """
        session.processed_data["wl1"] = self.convert_to_OD(session.processed_data["wl1"])
        session.processed_data["wl2"] = self.convert_to_OD(session.processed_data["wl2"])

    def convert_to_OD(self, raw_data):
        """
        Handles 1-D or N-D input, guards against zero baseline.
        OD = –ln(I / I₀), I₀ = mean(raw[200:300]).
        """
        idx  = np.arange(200, 300)
        I0   = np.mean(raw_data[idx], axis=0)
        if np.isscalar(I0):
            if I0 == 0: I0 = np.finfo(float).eps
        else:
            I0[I0 == 0] = np.finfo(float).eps
        return -np.log(raw_data / I0)


class ConvertODtoHbOandHbR(ProcessingStep):
    """
    A processing step that converts optical density data to hemoglobin concentrations.
    
    Converts OD data from two wavelengths to HbO (oxyhemoglobin) and HbR (deoxyhemoglobin)
    concentrations using the Beer-Lambert law and extinction coefficients.
    """
    
    def _do_process(self, session: Session) -> None:
        """
        Convert OD data to HbO and HbR concentrations.
        
        Args:
            session: The session to process (modified in-place)
        """
        od_wl1 = session.processed_data["wl1"]
        od_wl2 = session.processed_data["wl2"]
        
        HbO, HbR = self._convert_od_to_hemo(od_wl1, od_wl2)
        # ---------- NEW: quick HbO/HbR polarity sanity-check -------------
        # During a typical activation HbO ↑ while HbR ↓  ⇒  corr ≈ −1.
        # If most channels show a **positive** correlation something is
        # very likely flipped (wrong Beer–Lambert sign, swapped rows, …).
        median_corr = np.nanmedian([
            np.corrcoef(HbO[:, ch], HbR[:, ch])[0, 1]
            for ch in range(HbO.shape[1])
        ])
        if median_corr > 0:
            print("[OD→Hb]   WARNING: median HbO/HbR correlation is > 0 "
                  "(expected negative) – check extinction-matrix sign!")
            # OPTIONAL auto-fix (uncomment if you prefer auto-correction)
            # HbR *= -1.0
            # print("[OD→Hb]   Auto-flipped HbR sign to restore polarity.")

        session.processed_data["HbO"] = HbO
        session.processed_data["HbR"] = HbR
        del session.processed_data["wl1"]
        del session.processed_data["wl2"]

    def _convert_od_to_hemo(self, od_wl1: np.ndarray, od_wl2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert optical density data to hemoglobin concentrations.
        
        Handles both (T, S, D) and (T, S*D) shaped data. For (T, S*D) data,
        the transformation is applied to each time point and channel pair.
        
        Args:
            od_wl1: Optical density data for wavelength 1
            od_wl2: Optical density data for wavelength 2
            
        Returns:
            Tuple of (HbO, HbR) concentration arrays with same shape as input
        """
        if od_wl1.shape != od_wl2.shape:
            raise ValueError("od_wl1 and od_wl2 must have the same shape")
        
        if len(od_wl1.shape) == 3:
            # Original (T, S, D) format
            return self._convert_3d_od_to_hemo(od_wl1, od_wl2)
        elif len(od_wl1.shape) == 2:
            # Flattened (T, S*D) format
            return self._convert_2d_od_to_hemo(od_wl1, od_wl2)
        else:
            raise ValueError(f"Expected 2D or 3D data, got shape {od_wl1.shape}")

    def _convert_3d_od_to_hemo(self, od_wl1: np.ndarray, od_wl2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert 3D OD data (T, S, D) to hemoglobin concentrations.
        
        Args:
            od_wl1: Optical density data for wavelength 1, shape (T, S, D)
            od_wl2: Optical density data for wavelength 2, shape (T, S, D)
            
        Returns:
            Tuple of (HbO, HbR) concentration arrays, each with shape (T, S, D)
        """
        T, S, D = od_wl1.shape
        HbO = np.zeros((T, S, D))
        HbR = np.zeros((T, S, D))

        # Solve for each time x source x detector
        for t in range(T):
            for s in range(S):
                for d in range(D):
                    od_vec = np.array([od_wl1[t, s, d], od_wl2[t, s, d]])
                    hbo, hbr = A_inv.dot(od_vec)
                    HbO[t, s, d] = hbo
                    HbR[t, s, d] = hbr

        return HbO, HbR

    def _convert_2d_od_to_hemo(self, od_wl1: np.ndarray, od_wl2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert 2D OD data (T, S*D) to hemoglobin concentrations.
        
        Args:
            od_wl1: Optical density data for wavelength 1, shape (T, S*D)
            od_wl2: Optical density data for wavelength 2, shape (T, S*D)
            
        Returns:
            Tuple of (HbO, HbR) concentration arrays, each with shape (T, S*D)
        """
        T, channels = od_wl1.shape
        HbO = np.zeros((T, channels))
        HbR = np.zeros((T, channels))

        # Solve for each time x channel
        for t in range(T):
            for c in range(channels):
                od_vec = np.array([od_wl1[t, c], od_wl2[t, c]])
                hbo, hbr = A_inv.dot(od_vec)
                HbO[t, c] = hbo
                HbR[t, c] = hbr

        return HbO, HbR


class RemoveScalpEffect(ProcessingStep):
    """
    A processing step that removes scalp effects from HbO and HbR data using PCA.
    
    This step fits a PCA model on baseline data and removes the first principal
    component contribution from the entire time series, which is assumed to
    represent scalp effects.
    """
    
    def _do_process(self, session: Session) -> None:
        """
        Remove scalp effects from HbO and HbR data.
        
        Args:
            session: The session to process (modified in-place)
        """
        HbO = session.processed_data["HbO"]
        HbR = session.processed_data["HbR"]
        
        HbO_clean = self._subtract_first_pc(HbO, session.protocol)
        HbR_clean = self._subtract_first_pc(HbR, session.protocol)
        
        session.processed_data["HbO"] = HbO_clean
        session.processed_data["HbR"] = HbR_clean
    
    def _subtract_first_pc(self, Hb: np.ndarray, protocol) -> np.ndarray:
        """
        Subtract the first principal component contribution from the data.
        
        Args:
            Hb: Hemoglobin data with shape (T, S*D)
            protocol: Protocol object containing timing information
            
        Returns:
            Cleaned hemoglobin data with same shape as input
        """
        # 1) Fit PCA on baseline
        baseline_2D = self.extract_init_baseline_data(Hb, protocol)
        pca_1 = PCA(n_components=1)
        pca_1.fit(baseline_2D)

        # 2) Project entire time series on that PC and reconstruct
        scores = pca_1.transform(Hb)
        pc1_contribution = pca_1.inverse_transform(scores)

        # 3) Subtract PC1 contribution
        cleaned_2D = Hb - pc1_contribution
        return cleaned_2D
    
    def extract_init_baseline_data(self, Hb: np.ndarray, protocol) -> np.ndarray:
        """
        Extract baseline data from the beginning of the time series.
        
        Args:
            Hb: Hemoglobin data with shape (T, S*D)
            protocol: Protocol object containing timing information
            
        Returns:
            Baseline data with shape (T_baseline, S*D)
        """
        idx_start, idx_end = self.get_baseline_indices(protocol)
        baseline_data = Hb[idx_start:idx_end]  # shape: (T_baseline, S*D)
        return baseline_data

    def get_baseline_indices(self, protocol) -> tuple[int, int]:
        """
        Get the start and end indices for baseline data.
        
        Baseline is defined as the period from time 0 to the start of the first task.
        
        Args:
            protocol: Protocol object containing timing information
            
        Returns:
            Tuple of (start_index, end_index) for baseline period
        """
        baseline_start_time = 0.0
        baseline_end_time = min(interval[0] for interval in protocol.intervals)

        # Convert times to indices
        idx_start = int(baseline_start_time)
        idx_end = int(baseline_end_time)

        return idx_start, idx_end

class HemoToOD(ProcessingStep):
    def hemo_to_od(self,
            HbO,
            HbR,
    ):
        T = len(HbO)
        HbO = np.reshape(HbO, (T,num_sources,num_detectors))
        HbR = np.reshape(HbR, (T, num_sources, num_detectors))
        S = num_sources
        D = num_detectors
        od_wl1 = np.zeros((T, S, D))
        od_wl2 = np.zeros((T, S, D))
        for t in range(T):
            for s_idx in range(S):
                for d_idx in range(D):
                    hemo_vec = np.array([HbO[t, s_idx, d_idx], HbR[t, s_idx, d_idx]])
                    od_vec = A.dot(hemo_vec)
                    od_wl1[t, s_idx, d_idx] = od_vec[0]
                    od_wl2[t, s_idx, d_idx] = od_vec[1]

        return od_wl1, od_wl2

    def _do_process(self, session: 'Session') -> None:
        HbO = session.processed_data["t_HbO"]
        HbR = session.processed_data["t_HbR"]
        od_wl1, od_wl2 = self.hemo_to_od(HbO,HbR)
        session.processed_data["wl1"] = od_wl1
        session.processed_data["wl2"] = od_wl2
        del session.processed_data["t_HbO"]
        del session.processed_data["t_HbR"]


# class Zscore(ProcessingStep):
#     def _do_process(self, session: 'Session') -> None:
#         processed_data = session.processed_data("HbO)
#         processed_data = processed_data - np.mean(processed_data)

class BandpassFilter(ProcessingStep):
    """
    A processing step that applies a bandpass filter to wavelength data.
    
    Uses a 3rd order Butterworth filter to filter data between the specified
    frequency bounds. Applies the filter to both wl1 and wl2 data channels.
    """
    
    def __init__(self, upper_bound: float, lower_bound: float):
        """
        Initialize the bandpass filter.
        
        Args:
            upper_bound: Upper frequency bound in Hz
            lower_bound: Lower frequency bound in Hz
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def _do_process(self, session: Session) -> None:
        """
        Apply bandpass filter to both wl1 and wl2 data channels.
        
        Args:
            session: The session to process (modified in-place)
        """
        session.processed_data["HbO"] = self._apply_filter(session.processed_data["HbO"])
        session.processed_data["HbR"] = self._apply_filter(session.processed_data["HbR"])
    
    def _apply_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a 3rd order Butterworth bandpass filter to the data.
        
        Args:
            data: Input data array to filter
            
        Returns:
            Filtered data array
        """
        # Design the Butterworth filter
        nyquist = 0.5 * fs # Assuming normalized frequency
        low = self.lower_bound / nyquist
        high = self.upper_bound / nyquist
        
        # Create 3rd order Butterworth bandpass filter
        b, a = signal.butter(3, [low, high], btype='band')
        
        # Apply the filter
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        
        return filtered_data


class CroppingStep(ProcessingStep):
    """
    A processing step that crops raw wavelength data and adjusts protocol intervals.
    
    This step removes a specified number of samples from the beginning and end of
    the time series and updates the protocol intervals to match the new time base.
    """
    
    def __init__(self, crop_samples: int = 50):
        """
        Initialize the cropping step.
        
        Args:
            crop_samples: Number of samples to remove from both beginning and end
        """
        self.crop_samples = crop_samples

    def _do_process(self, session: Session) -> None:
        """
        Crop the raw wavelength data and adjust protocol intervals.
        
        Args:
            session: The session to process (modified in-place)
        """
        # Crop the raw wavelength data
        session.processed_data["wl1"] = session.processed_data["wl1"][self.crop_samples:-self.crop_samples, ...]
        session.processed_data["wl2"] = session.processed_data["wl2"][self.crop_samples:-self.crop_samples, ...]
        
        # Adjust protocol intervals
        session.protocol = self._adjust_protocol(session.protocol, self.crop_samples)
    
    def _adjust_protocol(self, protocol, crop_samples: int):
        """
        Adjust protocol intervals to account for cropped data.
        
        Args:
            protocol: Original protocol object
            crop_samples: Number of samples cropped from beginning
            
        Returns:
            New protocol object with adjusted intervals
        """
        # Convert crop_samples to time units using sampling frequency
        crop_time = crop_samples / fs
        
        # Create new intervals with adjusted timing
        new_intervals = []
        for start_time, end_time, label in protocol.intervals:
            # Adjust times by subtracting crop_time
            new_start = start_time - crop_time
            new_end = end_time - crop_time
            
            # Only keep intervals that are still within the valid time range
            if new_end > 0:  # Interval ends after the cropped start
                # Clamp start time to 0 if it goes negative
                clamped_start = max(0, new_start)
                new_intervals.append((clamped_start, new_end, label))
        
        # Create new protocol with adjusted intervals
        from lys.objects.protocol import Protocol
        return Protocol(intervals=new_intervals)