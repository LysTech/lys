import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import convolve
from scipy.stats import gamma
from lys.objects import Session
from lys.interfaces.processing_step import ProcessingStep

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
    peak1, peak2 = 4, 14
    ratio = 1/6

    hrf1 = gamma.pdf(times, peak1)
    hrf2 = gamma.pdf(times, peak2)
    hrf = hrf1 - ratio * hrf2
    return hrf

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



class ReconstructDualWithoutBadChannels(ProcessingStep):
    """
    Dual‐wavelength eigenmode reconstruction that first drops any flat/noisy
    channels flagged in session.processed_data["bad_channels"].
    """
    def __init__(self, num_eigenmodes: int,
                 lambda_selection: str = "lcurve"):   # "lcurve"  or  "corr"
        self.num_eigenmodes   = num_eigenmodes
        self.lambda_selection = lambda_selection      # ← new

    # ------------------------------------------------------------------
    # Pick parameter that maximizes correlation with fMRI
    # ------------------------------------------------------------------
    def _find_best_lambda_corr(self, B1, B2, y1, y2,
                               phi, eigvals, verts, fmri, bad):
        params  = np.logspace(-5, 5, 65)
        best_r, best_lam = -np.inf, params[0]
        for lam in params:
            X = self._reconstruct(B1, B2, y1, y2, phi, lam, eigvals,
                                  bad_channels=bad)
            r = np.corrcoef(X, fmri)[0, 1]
            if r > best_r:
                best_r, best_lam = r, lam
        return best_lam

    # ------------------------------------------------------------------
    # NEW L‑curve picker (helpers embedded for brevity)
    # ------------------------------------------------------------------
    @staticmethod
    def _curv(xm, x, xp, ym, y, yp):
        dx1, dx2 = x - xm, xp - x;  dy1, dy2 = y - ym, yp - y
        dx = .5*(dx1+dx2);  dy = .5*(dy1+dy2)
        ddx = dx2 - dx1;    ddy = dy2 - dy1
        return abs(ddx*dy - ddy*dx) / ((dx*dx+dy*dy)**1.5 + 1e-12)

    def _find_best_lambda_lcurve(self, B1, B2, y1, y2,
                                 phi, eigvals, verts, fmri, bad):
        # build dual system once (same as in _reconstruct)
        εO1, εR1, εO2, εR2 = 586, 1548.52, 1058, 691.32
        M1 = np.hstack((εO1*B1, εR1*B1));  M2 = np.hstack((εO2*B2, εR2*B2))
        B  = np.vstack((M1, M2))
        y  = np.concatenate((y1.flatten(order='F'), y2.flatten(order='F')))
        if bad:                              # drop rows for bad channels twice
            m      = B1.shape[0]
            mask   = np.ones(B.shape[0], bool)
            mask[bad] = False;  mask[m+np.array(bad)] = False
            B, y  = B[mask], y[mask]

        Λ = np.diag(np.tile(eigvals, 2))
        n = eigvals.size; I = np.eye(n); rho, mu = -0.6, 0.1
        C = mu*np.block([[I, -rho*I], [-rho*I, rho**2*I]])

        lam_grid = np.logspace(-5, 5, 65)
        res, sol = [], []
        for lam in lam_grid:
            A   = B.T@B + lam*Λ + C
            α   = np.linalg.solve(A, B.T@y)
            res.append(np.linalg.norm(B@α - y))
            sol.append(np.linalg.norm(np.sqrt(Λ)@α))

        log_r, log_s = np.log10(res), np.log10(sol)
        curv = [0]+[self._curv(log_r[i-1],log_r[i],log_r[i+1],
                               log_s[i-1],log_s[i],log_s[i+1])
                    for i in range(1,len(lam_grid)-1)]+[0]
        return float(lam_grid[int(np.argmax(curv))])

    # ------------------------------------------------------------------
    # choose picker based on user flag
    # ------------------------------------------------------------------
    def _find_best_lambda(self, *args, **kw):
        pick = self._find_best_lambda_lcurve if self.lambda_selection=="lcurve" \
               else self._find_best_lambda_corr
        return pick(*args, **kw)


    def _do_process(self, session: "Session") -> None:
        # --- 1) load mesh + eigenmodes ---
        verts = session.patient.mesh.vertices
        if session.patient.mesh.eigenmodes is None:
            raise ValueError("Mesh has no eigenmodes.")
        start, end = 2, 2 + self.num_eigenmodes
        # stack then transpose → (n_vertices, M)
        phi = np.vstack(session.patient.mesh.eigenmodes[start:end]).T
        eigvals = np.array([e.eigenvalue for e in session.patient.mesh.eigenmodes[start:end]])
        eigvals[0] = 0.0

        # --- 2) sample Jacobians + build Bmn blocks ---
        vj1 = session.jacobians[0].sample_at_vertices(verts)
        vj2 = session.jacobians[1].sample_at_vertices(verts)
        B1 = self._compute_Bmn(vj1, phi)
        B2 = self._compute_Bmn(vj2, phi)

        # --- 3) grab bad_channels (must have run DetectBadChannels first) ---
        bad = session.processed_data.get("bad_channels", [])

        # --- 4) iterate tasks ---
        tO = session.processed_data["t_HbO"]
        tR = session.processed_data["t_HbR"]
        session.processed_data["t_HbO_reconstructed"] = {}
        session.processed_data["t_HbR_reconstructed"] = {}

        from lys.utils.mri_tstat import get_mri_tstats
        for task in session.protocol.tasks:
            y1 = tO[task]
            y2 = tR[task]
            fmri = get_mri_tstats(session.patient.name, task)

            lam = self._find_best_lambda(B1, B2, y1, y2, phi, eigvals, verts, fmri, bad)
            print(f"  Optimal regularization parameter: {lam:.6f}\n")
            rec = self._reconstruct(
                B1, B2, y1, y2, phi, lam, eigvals,
                bad_channels=bad
            )

            session.processed_data["t_HbO_reconstructed"][task] = rec
            session.processed_data["t_HbR_reconstructed"][task] = rec

        # cleanup
        del session.processed_data["t_HbO"]
        del session.processed_data["t_HbR"]

    def _compute_Bmn(self, vj: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Contract Jacobian (N_vertices, S, D) with phi (N_vertices, M)
        → Bmn (S*D, M) in Fortran order.
        """
        B = np.einsum("nsd,nm->sdm", vj, phi)
        S, D, M = B.shape
        return B.reshape(S * D, M, order="F")


    def _reconstruct(self,
                     B1, B2,
                     y1, y2,
                     phi, lam, eigvals,
                     *,
                     ext=(586, 1548.52, 1058, 691.32),
                     w=(1., 1.), rho=-0.6, mu=0.1,
                     bad_channels: list[int] = []
                     ) -> np.ndarray:
        """
        Soft‐tied dual‐wavelength inversion, dropping bad_channels in both B and y.
        Returns vertex‐wise HbO map.
        """
        εO1, εR1, εO2, εR2 = ext
        # build dual forward model
        M1 = np.hstack((εO1 * B1, εR1 * B1))
        M2 = np.hstack((εO2 * B2, εR2 * B2))
        B = np.vstack((w[0] * M1, w[1] * M2))
        y = np.concatenate((w[0] * y1.flatten(order="F"),
                            w[1] * y2.flatten(order="F")))

        # drop bad rows
        m = B1.shape[0]
        if bad_channels:
            mask = np.ones(B.shape[0], dtype=bool)
            mask[bad_channels] = False
            mask[m + np.array(bad_channels)] = False
            B = B[mask, :]
            y = y[mask]

        # regularisation
        Λ = np.diag(np.tile(eigvals, 2))
        I = np.eye(eigvals.size)
        C = mu * np.block([[I, -rho * I],
                           [-rho * I, rho ** 2 * I]])
        A = B.T @ B + lam * Λ + C

        α = np.linalg.solve(A, B.T @ y)
        return phi @ α[:eigvals.size]


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
        del session.processed_data["HbO"]
        del session.processed_data["HbR"]

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
        hrf = canonical_double_gamma_hrf(tr=tr, duration=30.0)  # or your choice
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
        nyquist = 0.5  # Assuming normalized frequency
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