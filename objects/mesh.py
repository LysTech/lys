import numpy as np
from scipy.io import loadmat
import vtk
from typing import List, Tuple

from lys.visualization.plot3d import VTKScene

"""
Defines classes related to meshes:
- Mesh
- StaticMeshData
- TimeSeriesMeshData

You can currently try this by doing

mesh = from_mat("/path/to/mesh.mat")
VTKScene().add(mesh).show()
"""

#TODO: 
# - add plotting to the __init__ function of Mesh
# - implement the downsample function
# - create our own mesh from a segmentation Volume

class Mesh:
    def __init__(self, vertices, faces, show = True):
        """ Construct mesh and show it if show is True """
        self.faces = faces
        self.vertices = vertices
        if show:
            VTKScene().add(self).show()
        self._check_mesh()

    def downsample(self, n_vertices: int) -> 'Mesh':
        """ Return a new mesh with only a subset of the vertices """
        raise NotImplementedError

    def _check_mesh(self):
        """ Check mesh properties:
            - 0-indexing
        """
        min_face_idx = min([min(f) for f in self.faces])
        assert min_face_idx == 0, f"Expected 0-indexed faces, got {min_face_idx}"

    def to_vtk(self, cmap: str = "viridis", opacity: float = 1.0, **kw) -> vtk.vtkActor:
        """Convert mesh to VTK actor."""
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for v in self.vertices:
            pts.InsertNextPoint(v)
        poly.SetPoints(pts)

        polys = vtk.vtkCellArray()
        for face in self.faces:
            polys.InsertNextCell(len(face), face)
        poly.SetPolys(polys)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        return actor
    
    def apply_style(self, actor: vtk.vtkActor, cmap: str = None, opacity: float = None, **kw):
        """Apply styling to existing VTK actor."""
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)


def from_mat(mat_file_path: str) -> Mesh:
    """ Mesh constructor: Load a Mesh from a MATLAB .mat file """
    mdata = loadmat(mat_file_path)
    vertices = mdata["vertices"].astype(float)  # shape (N,3)
    faces = mdata["faces"] - 1
    min_face_idx = min([min(f) for f in faces])
    assert min_face_idx == 0, f"Expected 1-indexed MATLAB faces, but minimum index after conversion is {min_face_idx}"
    return Mesh(vertices, faces)


class StaticMeshData:
    """ Mesh with one value per vertex, useful for plotting. """
    def __init__(self, mesh: Mesh, data: np.ndarray):
        self.mesh = mesh
        self.data = data
        self.data_range = (data.min(), data.max())
        
        # Check data dimensions
        assert len(data.shape) == 1, f"Expected 1D data array, got shape {data.shape}"
        assert data.shape[0] == mesh.vertices.shape[0], f"Data length {data.shape[0]} must match number of vertices {mesh.vertices.shape[0]}"

    def to_vtk(self, cmap: str = "viridis", opacity: float = 1.0, **kw) -> List[vtk.vtkProp]:
        """Convert static mesh data to VTK actors (mesh + scalar bar)."""
        # Get the base mesh actor
        actor = self.mesh.to_vtk(opacity=opacity, **kw)
        mapper = actor.GetMapper()
        poly = mapper.GetInput()

        # Add scalar data to the mesh
        scalars = vtk.vtkFloatArray()
        scalars.SetName("StaticData")
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfTuples(len(self.data))
        for i, v in enumerate(self.data):
            scalars.SetValue(i, float(v))
        poly.GetPointData().SetScalars(scalars)

        # Set scalar range
        mapper.SetScalarRange(*self.data_range)

        # Apply colormap
        if cmap == "viridis":
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.7, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(0.0, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)

        # Create scalar bar
        scalar_bar = _make_scalar_bar(mapper.GetLookupTable())
        actor._scalar_bar = scalar_bar  # Store reference for later use
        
        return [actor, scalar_bar]

    def apply_style(self, actor: vtk.vtkActor, cmap: str = None, opacity: float = None, 
                   data_range: Tuple[float, float] = None, **kw):
        """Apply styling to existing VTK actor."""
        # Apply base mesh styling first
        self.mesh.apply_style(actor, cmap=cmap, opacity=opacity, **kw)
        
        # Apply data-specific styling
        if data_range is not None:
            self.data_range = data_range
            mapper = actor.GetMapper()
            if mapper:
                mapper.SetScalarRange(*data_range)
        
        # Update colormap if requested
        if cmap is not None and cmap == "viridis":
            mapper = actor.GetMapper()
            if mapper:
                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.7, 0.0)
                lut.SetSaturationRange(1.0, 1.0)
                lut.SetValueRange(0.0, 1.0)
                lut.Build()
                mapper.SetLookupTable(lut)

    def _make_scalar_bar(self, lut: vtk.vtkLookupTable, title: str = "", n_labels: int = 5):
        """Helper to create scalar bar."""
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lut)
        bar.SetTitle(title)
        bar.SetNumberOfLabels(n_labels)
        bar.UnconstrainedFontSizeOn()
        return bar


class TimeSeriesMeshData:
    """ Mesh with one timeseries per vertex, useful for plotting. """
    def __init__(self, mesh: Mesh, timeseries: np.ndarray, current_timepoint: int = 0):
        self.mesh = mesh
        self.timeseries = timeseries
        self.current_timepoint = current_timepoint
        
        # Check timeseries dimensions
        assert len(timeseries.shape) == 2, f"Expected 2D timeseries array, got shape {timeseries.shape}"
        assert timeseries.shape[0] == mesh.vertices.shape[0], f"Timeseries vertices dimension {timeseries.shape[0]} must match number of vertices {mesh.vertices.shape[0]}"
    
    def to_vtk(self, cmap: str = "viridis", opacity: float = 1.0, **kw) -> List[vtk.vtkProp]:
        """Convert time-series mesh to VTK actors (mesh + scalar bar)."""
        actor = self.mesh.to_vtk(opacity=opacity, **kw)
        self._apply_timeseries_data(actor, self.timeseries[:, self.current_timepoint])

        mapper = actor.GetMapper()
        mapper.SetScalarRange(float(self.timeseries.min()), float(self.timeseries.max()))

        if cmap == "viridis":
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.7, 0.0)
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(0.0, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)

        scalar_bar = _make_scalar_bar(mapper.GetLookupTable())
        actor._scalar_bar = scalar_bar
        return [actor, scalar_bar]
    
    def apply_style(self, actor: vtk.vtkActor, **kw):
        """Apply styling to time-series mesh actor."""
        self.mesh.apply_style(actor, **kw)
    
    def update_timeseries_actor(self, actor: vtk.vtkActor):
        """Update actor with current timepoint data."""
        if hasattr(self, "timeseries") and hasattr(self, "current_timepoint"):
            data = self.timeseries[:, self.current_timepoint]
            self._apply_timeseries_data(actor, data)
    
    def _apply_timeseries_data(self, actor: vtk.vtkActor, data: np.ndarray):
        """Helper to apply timeseries data to actor."""
        mapper = actor.GetMapper()
        poly = mapper.GetInput() if mapper else None
        if poly is None:
            return

        scalars = vtk.vtkFloatArray()
        scalars.SetName("TimeSeries")
        scalars.SetNumberOfComponents(1)
        scalars.SetNumberOfTuples(len(data))
        for i, v in enumerate(data):
            scalars.SetValue(i, float(v))
        poly.GetPointData().SetScalars(scalars)
        poly.Modified()


# Standalone function for creating scalar bars
def _make_scalar_bar(lut: vtk.vtkLookupTable, title: str = "", n_labels: int = 5):
    """Helper to create scalar bar."""
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(title)
    bar.SetNumberOfLabels(n_labels)
    bar.UnconstrainedFontSizeOn()
    return bar


