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
from lys.objects.mesh import from_mat
from lys.visualization.plot3d import VTKScene

# Load a mesh from a MATLAB file
mesh = from_mat("/path/to/mesh.mat")

# Create and show the visualization
VTKScene().add(mesh).show()
```

### Working with Time-Series Data
```python
scene = VTKScene()
scene.add(time_series_mesh)
scene.show()
scene.style(time_series_mesh, opacity=0.5, cmap="viridis") #change the opacity and/or cmap
```

