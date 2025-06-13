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

# Add a time-series mesh
scene.add(time_series_mesh)

# Show the visualization with time control
scene.show()

# Update to a specific timepoint
scene.set_timepoint(5)
```

## Dependencies
- VTK
- PyQt5 (or PySide6)
- NumPy
- SciPy

## Development Guidelines

1. Default to plotting everything automatically
2. If a function was written by GPT and not carefully studied, mention this in the comment
3. Follow the established code structure and patterns
4. Add tests for new functionality
5. Document new features and changes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add license information here]
