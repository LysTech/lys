# Lys


## Project Structure

```
lys/
├── objects/
│   ├── mesh.py         # Core mesh classes and functionality
│   └── __init__.py
├── visualization/
│   └── plot3d.py       # VTK-based 3D visualization
└── tests/              # Test suite
```

## Core Features

### Mesh Processing
- Load meshes from MATLAB (.mat) files
- Basic mesh operations and validation
- Support for static and time-series data on meshes
- Mesh downsampling (planned feature)

### Visualization
- Interactive 3D visualization using VTK and Qt
- Support for both static and time-series data visualization
- Customizable colormaps and styling
- Time-series playback with slider control
- Screenshot capability
- Jupyter/IPython integration

## Usage Examples

### Basic Mesh Loading and Visualization
```python
from lys.objects.mesh import from_mat
from lys.visualization.plot3d import VTKScene

# Load a mesh from a MATLAB file
mesh = from_mat("/path/to/mesh.mat")

# Create and show the visualization
VTKScene().add(mesh).show()
```

### Working with Time-Series Data
```python
# Create a scene
scene = VTKScene()
scene.add(time_series_mesh)
scene.show()
scene.style(time_series_mesh, opacity=0.5) #change the opacity
scene.style(time_series_mesh, cmap="viridis") #change the cmap, currently only "viridis" is defined

```

