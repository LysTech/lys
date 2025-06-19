# Lys


## Project Structure

```
lys/
├── objects/
│   ├── atlas.py   
│   ├── mesh.py       
│   ├── volume.py     
│   └── __init__.py
├── utils/
│   ├── paths.py   
│   └── strings.py
├── visualization/
│   ├── plot3d.py 
│   └── utils.py
└── tests/  
```


## Usage Examples

### Visualization

The idea is to respect the Open-Closed principle + have composability. Whenever we define a new object that we want to be able to plot, it just needs to have a `to_vtk` method that returns a vtkActor and the `VTKScene` class can plot it. 

A few examples:

```python
from lys.objects.mesh import from_mat, StaticMeshData
from lys.visualization.plot3d import VTKScene

# Load a mesh from a MATLAB file
mesh = from_mat("/path/to/mesh.mat")

# Create and show the visualization
scene = VTKScene()
scene.add(mesh).show()
scene.remove(mesh)

# If you want data on the mesh, e.g. t-stats
data = np.random.rand(mesh.vertices.shape[0])) * 3 #these are between 0 and 3
static_data_mesh = StaticMeshData(mesh, data) 
scene.add(static_data_mesh) .show()
```

By default we use the full range of the colormap, so if you pass values between [0,0.3] then the cmap is normalised to use the full range of colors. If we so please, we could re-write this as a `normalize=True/False` argument but I think we always (?) want to use the full range of color.

```
# Update the colormap to any allowed matplotlib colormap (or custom!)
scene.style(static_data_mesh, cmap="inferno")
```

### Working with Time-Series Data
```python
scene = VTKScene()
scene.add(time_series_mesh)
scene.show()
scene.style(time_series_mesh, opacity=0.5, cmap="viridis") #change the opacity and/or cmap
```

StaticDataMesh and TimeSeriesDataMesh styles can be changed with: new opacities, new colormaps.

### Style Options by Object Type

| Type                | Style Updates (arguments to `apply_style`)                |
|---------------------|----------------------------------------------------------|
| **Atlas**           | `opacity`, `colors`                                       |
| **Points**          | `color`, `radius`, `opacity`                              |
| **Optodes**         | `radius`                                                  |
| **Volume**          | `opacity`, `color`, `data_range`, `cmap`                  |
| **Mesh**            | `opacity`                                                 |
| **StaticMeshData**  | `opacity`, `cmap`                                         |
| **TimeSeriesMeshData** | (delegates to StaticMeshData and Mesh, so: `opacity`, `cmap`) |

- For `Atlas`, you can update the opacity of all regions and provide a custom color mapping for regions.
- For `Points`, you can update the color (RGB tuple), radius, and opacity of the points.
- For `Optodes`, you can update the radius of the optode spheres (color is fixed: sources are red, detectors are blue).
- For `Volume`, you can update opacity, color, data range, and colormap.
- For `Mesh`, you can update opacity.
- For `StaticMeshData`, you can update opacity and colormap.
- For `TimeSeriesMeshData`, you can update opacity and colormap (applies to the current timepoint's data).


## Data Directory Setup

Before using Lys, you must set the `LYS_DATA_DIR` environment variable in your shell configuration (e.g., `.bashrc`, `.zshrc`). This variable should point to the root directory where your data is stored. For example:

```bash
LYS_DATA_DIR="/Users/thomasrialan/Documents/code/Geometric-Eigenmodes/data"
export LYS_DATA_DIR
```

Make sure to restart your terminal or source your shell configuration after making this change:

```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Data Folder Structure

Your data directory should be organized as follows:

```
subject/
├── anat/
│   ├── volumes/
│   │   ├── MRI/
│   │   └── segmentations/
│   └── meshes/
├── derivatives/
│   └── jacobians/
│       └── sub-001_jacobian.h5  # Actual file stored once
└── nirs/
    └── experiment/
        ├── session1/
        │   ├── (symlink) sub-001_jacobian.h5 -> ../../derivatives/jacobians/sub-001_jacobian.h5 
        │   ├── sub-001_optodes.some_format
        │   ├── data.snirf
        │   ├── protocol.prt
        │   └── processed_session1_v1
        └── session2/
            ├── (symlink) sub-001_jacobian.h5 -> ../../derivatives/jacobians/sub-001_jacobian.h5
            ├── sub-001_optodes.some_format
            ├── data.snirf
            └── protocol.prt
```

