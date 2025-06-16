import numpy as np
from lys.objects.mesh import from_mat, StaticMeshData
from lys.visualization.plot3d import VTKScene

mesh_file = "/Users/thomasrialan/Documents/code/Geometric-Eigenmodes/inverse_fnirs/P03_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_unMNI_D32k.mat"
mesh = from_mat(mesh_file)

scene = VTKScene()
static_mesh_data = StaticMeshData(mesh, np.random.rand(mesh.vertices.shape[0])*3)
scene.add(static_mesh_data).show()

# Show normalized data (values between 0 and 1)
#scene.style(static_mesh_data, normalize=True)

# Show original data again
#scene.style(static_mesh_data, normalize=False)

