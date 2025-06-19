# Lys


## Project Structure

```
lys/
├── objects/
│   ├── atlas.py        # Atlas-related functionality
│   ├── mesh.py         # Core mesh classes and functionality
│   ├── volume.py       # Volume data handling
│   └── __init__.py
├── visualization/
│   ├── plot3d.py       # VTK-based 3D visualization
│   └── utils.py        # Visualization utilities
└── tests/              # Test suite
```


## Usage Examples

### Basic Mesh Loading and Visualization
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

### Atlases
Atlases work the same, their opacity can be updated. 

### Points / Optodes
Their radius can be updated, as can their color (with an RGB tuple like `(0,0,1)`).

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

