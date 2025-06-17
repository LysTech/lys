import numpy as np
from lys.objects.mesh import from_mat, StaticMeshData, TimeSeriesMeshData
from lys.objects.volume import from_jnii
from lys.objects.atlas import Atlas
from lys.visualization.plot3d import VTKScene

# Construct a volume
volume_file = "../Geometric-Eigenmodes/data/P03/anat/volumes/P03_7tissues.jnii"
volume = from_jnii(volume_file)
atlas = Atlas(volume.array.astype(np.int32))

"""
# Construct a mesh
mesh_file = "/Users/thomasrialan/Documents/code/Geometric-Eigenmodes/inverse_fnirs/P03_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_unMNI_D32k.mat"
mesh = from_mat(mesh_file)

# Plot a timeseries on the mesh
scene = VTKScene()
data = np.random.rand(mesh.vertices.shape[0], 100) * 1./3
static_mesh_data = TimeSeriesMeshData(mesh, data)
scene.add(static_mesh_data).show()
"""



