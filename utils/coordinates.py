import numpy as np


"""
We have some notes on coordinate systems here:

https://www.notion.so/Lys-Technologies-5d72f2c92d98453a9b5cd2f866164f6b?p=19e1eff4030180edab3ccdc7c9717e74&pm=s
"""

def align_with_csf(vertices, vol, tissue):
    """
    Align mesh vertices with CSF points in the segmentation volume.
    Args:
        vertices (np.ndarray): Mesh vertices (already affine transformed and shifted).
        vol (np.ndarray): Segmentation volume.
        tissue (int): Tissue label to align with.
    Returns:
        np.ndarray: Vertices after alignment and translation.
    """
    mesh_min_x = np.argmin([v[0] for v in vertices])
    mesh_min_x_point = vertices[mesh_min_x]
    mesh_max_x = np.argmax([v[0] for v in vertices])
    mesh_max_x_point = vertices[mesh_max_x]

    csf_points = []
    for i in range(vol.shape[0]):
        for j in range(vol.shape[1]):
            for k in [94, 95, 96, 97, 98]:
                if vol[i, j, k] == tissue:
                    csf_points.append([i + 0.5, j + 0.5, k + 0.5])
                else:
                    continue

    csf_min_x = np.argmin([v[0] for v in csf_points])
    csf_min_x_point = csf_points[csf_min_x]
    csf_max_x = np.argmax([v[0] for v in csf_points])
    csf_max_x_point = csf_points[csf_max_x]

    L1_length = csf_max_x_point[0] - csf_min_x_point[0]  # length of medium
    L2_length = mesh_max_x_point[0] - mesh_min_x_point[0]  # length of mesh

    print(f"L2: {L2_length}")
    print(f"L1: {L1_length}")

    # Why is an extra translation required:
    # it is because the volume is only approximately centered on AC
    # and the mesh is also only approximately centered on AC
    # -> errors add up and lead to visible misalignment which this extra
    #    translation corrects for.

    t1 = np.array(csf_min_x_point) - np.array(mesh_min_x_point)
    t2 = np.array(csf_max_x_point) - np.array(mesh_max_x_point)
    t = (t1 + t2) / 2.0
    print(f"Translation vector: {t}")

    # extra shift by 12 because in reality AC point is between the hemispheres
    # not on the lh.
    tvertices = vertices + t + np.array([0, 0, 12])
    return tvertices


def undo_the_scaling(coords, x_scales, y_scales, z_scales):
    #undo the scaling
    points = coords.copy()
    neg_x = points[:,0] < 0
    points[:,0][neg_x]  /= x_scales[0]
    points[:,0][~neg_x] /= x_scales[1]

    neg_y = points[:,1] < 0
    points[:,1][neg_y]  /= y_scales[0]
    points[:,1][~neg_y] /= y_scales[1]

    neg_z = points[:,2] <0
    points[:,2][neg_z]  /= z_scales[0]
    points[:,2][~neg_z] /= z_scales[1]

    return points


def undo_affine_transformation(vertices, affine_matrix):
    """
    Apply affine transformation to mesh vertices.
    Args:
        vertices (np.ndarray): Mesh vertices.
        affine_matrix (np.ndarray): Affine transformation matrix.
    Returns:
        np.ndarray: Transformed vertices.
    """
    points = vertices.copy()
    ones = np.ones((points.shape[0], 1))
    hom_coords = np.hstack((points, ones))
    changed = (affine_matrix @ hom_coords.T).T
    changed = changed[:, :3]
    return changed


def read_adjBBX_file(patient: str):
    #TODO: make this take in a _adjBBX file and parse it, I currently don't have a copy of such a file
    print("***WARNING: NOT YET GENERAL, ONLY FOR P03***")
    affine_matrix = np.array([                                                      
                    [0.8542155027389526, -0.3071072399616241, -0.0305903572589159, -6.6516065597534180],
                    [0.4039888083934784, 0.7342166304588318, 0.0154890427365899, -7.3568248748779297],
                    [0.0245080236345530, -0.0236067622900009, 0.8310761451721191, 0.5436074733734131],
                    [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]
                ])                                                              
                                                                                
    x_scales = [1.05517, 1.0186]                                                    
    y_scales = [1.02484, 1.11429]                                                   
    z_scales = [1.05594, 1.05517] 
    return affine_matrix, x_scales, y_scales, z_scales

