# Lys

## Table of Contents

- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Visualization](#visualization)
- [Patient Class](#patient-class)
- [Jacobian](#jacobian)
- [Data Directory Setup](#data-directory-setup)
- [Data Folder Structure](#data-folder-structure)

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


## Documentation

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

```python
# Update the colormap to any allowed matplotlib colormap (or custom!)
scene.style(static_data_mesh, cmap="inferno")
```

**Working with Time-Series Data**

```python
scene = VTKScene()
scene.add(time_series_mesh)
scene.show()
scene.style(time_series_mesh, opacity=0.5, cmap="viridis") #change the opacity and/or cmap
```

StaticDataMesh and TimeSeriesDataMesh styles can be changed with: new opacities, new colormaps.

**Style Options by Object Type**

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

**Snapshot Testing for Visualization**

To ensure that visualization outputs remain consistent over time, Lys uses **snapshot tests**. These tests render objects (such as meshes, atlases, and optodes), save a reference image (snapshot) on the first run, and compare future renders pixel-by-pixel against this snapshot. If a rendering changes unexpectedly, the test will fail, helping to catch regressions or unintended changes in visualization. You can find these tests in `tests/test_plot3d_snapshot.py` and the reference images in `tests/snapshots/`.


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

This is cool because now we can call `lys.utils.paths.lys_data_dir()` and even if we have different computers it'll work for us all.

### Patient Class

The `Patient` class represents a subject with their associated brain segmentation and surface mesh. It is designed as an immutable (frozen) dataclass to ensure that once a `Patient` object is created, its attributes cannot be changed. This immutability helps prevent accidental modification of patient data, making the codebase safer and easier to reason about, especially when passing patient objects between functions or threads.

**How to Instantiate a Patient**

To create a `Patient` object, use the `from_name` class method, which automatically loads the relevant segmentation and mesh data for the given patient identifier:

```python
from lys.objects import Patient

# Instantiate a patient by their identifier (e.g., "P03")
patient = Patient.from_name("P03")

# Access patient attributes
print(patient.name)           # 'P03'
print(patient.segmentation)   # Atlas object
print(patient.mesh)           # Mesh object
```

- `name`: The patient identifier string (e.g., "P03").
- `segmentation`: An `Atlas` object representing the patient's brain segmentation.
- `mesh`: A `Mesh` object representing the patient's brain surface mesh in native space.

**Why Immutability?**

The `Patient` class is defined as a frozen dataclass (`@dataclass(frozen=True)`), which means its attributes cannot be modified after creation. This design choice:
- Prevents accidental changes to patient data after loading.
- Makes `Patient` objects hashable and safe to use as dictionary keys or in sets.
- Encourages a functional programming style, improving code reliability and maintainability.

### Jacobian

The `Jacobian` class and its associated functions provide a way to load, represent, and work with Jacobian matrices, which are typically used in neuroimaging and optical modeling to describe how measurements relate to changes in tissue properties at different locations in the brain.

**Key Features:**

  - The class provides a method `sample_at_vertices(vertices)` to interpolate values from the Jacobian at arbitrary 3D coordinates using linear interpolation.

- **Loading Jacobians:**
  - Use `load_jacobians(patient, experiment, session)` to automatically find and load all Jacobian files for a given subject and session. This returns a list of `Jacobian` objects.
  - Use `load_jacobian_from(path)` to load a Jacobian from a specific file path. Currently, only MATLAB `.mat` files are supported (with more formats planned).
  - The function `load_jacobian_from_mat(path)` loads a Jacobian from a MATLAB `.mat` file, handling the necessary transposition to convert MATLAB's storage order to the expected Python/NumPy order. A warning is issued to remind users of this transformation.

- **File Discovery:**
  - The helper function `_jacobian_paths(patient, experiment, session)` constructs the expected file paths for Jacobian files based on the data directory structure and returns all matching files in the session directory.
  - Jacobian files are expected to have 'jacobian' in their filename and be located in the appropriate session directory under the data root.

- **Example Usage:**

```python
from lys.objects.jacobian import load_jacobians

# Load all Jacobians for a given patient/session
jacobians = load_jacobians('P03', 'fnirs_8classes', 'session-01')

# Interpolate values at specific 3D coordinates
vertices = np.array([[10, 20, 30], [40, 50, 60]])
sampled = jacobians[0].sample_at_vertices(vertices)
print(sampled)
```

- **Notes:**
  - Only `.mat` files are currently supported for Jacobian loading. Attempting to load other formats will raise an error.
  - The code is designed to be extensible for future file types.
  - If no Jacobian file is found for a session, a `FileNotFoundError` is raised with a helpful message.


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

