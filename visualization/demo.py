import numpy as np
from lys.objects.mesh import from_mat, StaticMeshData, TimeSeriesMeshData
from lys.objects.volume import from_jnii
from lys.objects.atlas import Atlas
from lys.objects.optodes import Points
from lys.visualization.plot3d import VTKScene

# Construct a volume
volume_file = "../Geometric-Eigenmodes/data/P03/anat/volumes/P03_7tissues.jnii"
volume = from_jnii(volume_file)
atlas = Atlas(volume.array.astype(np.int32))

# Construct a mesh
mesh_file = "../Geometric-Eigenmodes/data/P03/anat/meshes/P03_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_D32k.mat"
mesh = from_mat(mesh_file)

# Plot a timeseries on the mesh
scene = VTKScene()
data = np.random.rand(mesh.vertices.shape[0], 100) * 1./3
static_mesh_data = TimeSeriesMeshData(mesh, data)
scene.add(static_mesh_data).show()

coordinates = [
(97.709549, 70.279701, 53.722301),
(151.368797, 86.386452, 79.984344),
(119.707191, 52.001999, 85.864594),
(164.997999, 111.524994, 108.707054),
(144.374435, 65.484200, 106.892235),
(96.487450, 43.001999, 103.500153),
(160.977913, 86.979912, 138.252289),
(120.589859, 50.591854, 133.964661),
(165.700596, 114.902275, 165.297409),
(140.858078, 66.860077, 160.304810),
(95.074440, 49.001999, 160.519211),
(149.928448, 93.048210, 190.117767),
(113.717392, 60.001999, 182.531372),
(76.019394, 60.879463, 182.896866),
(93.407166, 81.606064, 210.604065),
(117.611732, 110.079796, 221.998001)]

scene.remove(static_mesh_data)
points = Points(coordinates)
scene.add(points).show()





