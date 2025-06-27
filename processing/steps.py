import numpy as np
from scipy import signal
from sklearn.decomposition import PCA
from scipy import stats

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

#Vertex jacobian should have shape (num_vertices, num_sources, num_detectors) (or some transpose of that)
#num sources: 16, num detectors: 24


class ReconstructEigenmodes(ProcessingStep):
    def __init__(self, num_eigenmodes: int, regularisation_param: float = 0.01):
        self.num_eigenmodes = num_eigenmodes
        self.regularisation_param = regularisation_param

    def _do_process(self, session: Session) -> None:
        """
        Reconstruct the data.
        """
        vertices = session.patient.mesh.vertices
        phi = np.array([e for e in session.patient.mesh.eigenmodes]).T #Shape: (n_vertices, n_eigenmodes)
        vertex_jacobian = session.jacobians[0].sample_at_vertices(vertices)
        Bmn = self.compute_Bmn(vertex_jacobian, phi)
        eigenvals = np.array([e.eigenvalue for e in session.patient.mesh.eigenmodes])
        # we have session.processed_data["t_HbO"] and session.processed_data["t_HbR"]
        # they both have shape: (n_channels, n_tasks)
        # we want the reconstructed t-stat maps for each task, both HbO and HbR
        
        # Initialize the reconstructed data dictionaries
        n_tasks = session.processed_data["t_HbO"].shape[1]
        session.processed_data["t_HbO_reconstructed"] = {}
        session.processed_data["t_HbR_reconstructed"] = {}
        
        # Reconstruct for each task
        for task_idx in range(n_tasks):
            session.processed_data["t_HbO_reconstructed"][task_idx] = self.reconstruct(Bmn, session.processed_data["t_HbO"][:, task_idx], phi, eigenvals)
            session.processed_data["t_HbR_reconstructed"][task_idx] = self.reconstruct(Bmn, session.processed_data["t_HbR"][:, task_idx], phi, eigenvals)
        
        # Clean up intermediate data
        del session.processed_data["t_HbO"]
        del session.processed_data["t_HbR"]


    def reconstruct(self, Bmn, y_m, phi, eigenvals):
        L = np.diag(self.regularisation_param * eigenvals)
        A = Bmn.T @ Bmn + L
        b = Bmn.T @ y_m
        alphas = np.linalg.solve(A, b)
        X = phi @ alphas
        return X




    def compute_Bmn(self, vertex_jacobian: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        vertex_jacobian : (D, S, N) array
            Jacobian evaluated at the selected vertices
            (D = n_detectors, S = n_sources, N = n_vertices).

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
        B_sdm = np.einsum('dsn,nm->sdm', vertex_jacobian, phi)

        # Reshape to (S·D, M) with Fortran ordering so sources vary fastest.
        S, D, M = B_sdm.shape
        Bmn = B_sdm.reshape(S * D, M, order='F')
        return Bmn



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

    def _get_t_stats(self, HbO: np.ndarray, HbR: np.ndarray, protocol) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute t-statistics for HbO and HbR data using GLM analysis.
        
        Args:
            HbO: Oxyhemoglobin data with shape (T, n_channels)
            HbR: Deoxyhemoglobin data with shape (T, n_channels)
            protocol: Protocol object containing timing information
            
        Returns:
            Tuple of (t_HbO, t_HbR) t-statistics arrays, each with shape (n_channels, n_conditions)
        """
        T, n_channels = HbO.shape  # n_channels = S*D
        # 1) Build the same design matrix X that you used in GLM
        X = self._create_design_matrix(protocol, fs, T)
        n_conditions = X.shape[1]
        
        # Initialize outputs
        t_HbO = np.zeros((n_channels, n_conditions))
        t_HbR = np.zeros((n_channels, n_conditions))
        
        # 2) For each channel, run AR(1)+IRLS and compute t-stats
        for ch in range(n_channels):
            y_o = HbO[:, ch]
            y_r = HbR[:, ch]
            
            # Fit and get (beta, tvals) for oxy
            beta_o, tvals_o = self._glm_single_channel_tstats(
                y_o, X, max_iter=self.max_iter, tol=self.tol
            )
            t_HbO[ch, :] = tvals_o
            
            # Fit and get (beta, tvals) for deoxy
            beta_r, tvals_r = self._glm_single_channel_tstats(
                y_r, X, max_iter=self.max_iter, tol=self.tol
            )
            t_HbR[ch, :] = tvals_r
        
        return t_HbO, t_HbR

    def _create_design_matrix(self, protocol, fs: float, n_time_points: int) -> np.ndarray:
        """
        Create a design matrix for GLM analysis based on the protocol.
        
        Args:
            protocol: Protocol object containing timing information
            fs: Sampling frequency in Hz
            n_time_points: Number of time points in the data
            
        Returns:
            Design matrix with shape (n_time_points, n_conditions)
        """
        # Get unique task conditions
        conditions = list(protocol.tasks)
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
        
        return X

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
        
        HbO, HbR = self._convert_od_to_hemo_concentrations(od_wl1, od_wl2)
        
        session.processed_data["HbO"] = HbO
        session.processed_data["HbR"] = HbR
        del session.processed_data["wl1"]
        del session.processed_data["wl2"]

    def _convert_od_to_hemo_concentrations(self, od_wl1: np.ndarray, od_wl2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        session.processed_data["wl1"] = self._apply_filter(session.processed_data["wl1"])
        session.processed_data["wl2"] = self._apply_filter(session.processed_data["wl2"])
    
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