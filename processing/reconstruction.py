import numpy as np
from lys.interfaces import ProcessingStep
from lys.objects import Session

#TODO: docstring for compute_Bmn
#TODO: hardcode the number of sources and detectors is BAD!!!

class ReconstructionStep(ProcessingStep):
    def __init__(self, num_eigenmodes: int):
        self.num_eigenmodes = num_eigenmodes

    def _do_process(self, session: Session) -> None:
        """ this modifies session.processed_data inplace """
        eigenmodes = session.patient.mesh.eigenmodes
        tasks = session.protocol.tasks
        vertex_jacobian_wl1 = session.jacobians[0].sample_at_vertices(session.patient.mesh.vertices)
        num_sources = 16
        num_detectors = 24
        Bmn_wl1 = compute_Bmn(vertex_jacobian_wl1,
                        eigenmodes, 
                        num_sources, #TODO: where does this come from?
                        num_detectors, #TODO: where does this come from?
                        self.num_eigenmodes)
        reconstructed = reconstruct(Bmn_wl1, session.processed_data["wl1"], eigenmodes, 0.01)
        session.processed_data = reconstructed


def compute_Bmn(vertex_jacobian, phi, num_sources, num_detectors, num_eigenmodes):
    # this is 1000x faster than naive version                                   
    Bsdn = np.tensordot(vertex_jacobian, phi, axes=(0, 0))                      
    Bmn = Bsdn.reshape((num_sources * num_detectors, num_eigenmodes))
    return Bmn    


def reconstruct(Bmn, y_sd, eigenmodes, regularisation_param):
    eigenvals = np.array([eigenmode.eigenvalue for eigenmode in eigenmodes])
    y_m = y_sd.flatten(order="F")
    L = np.diag(regularisation_param * eigenvals)
    A = Bmn.T @ Bmn + L
    b = Bmn.T @ y_m
    alphas = np.linalg.solve(A, b)
    #TODO: would be nice to just do eigenmodes@alphas but haven't quite figured out how to do that
    X = np.array([e.vector for e in eigenmodes]) @ alphas
    return X