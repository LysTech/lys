import numpy as np
from scipy.io import loadmat
import vtk
from typing import List, Tuple

from lys.visualization.plot3d import VTKScene
from lys.visualization.utils import _make_scalar_bar, get_vtk_colormap

"""
Defines classes related to meshes:
- Mesh
- StaticMeshData
- TimeSeriesMeshData
"""

#TODO: 
# - implement the downsample function
# - create our own mesh from a segmentation Volume

class Mesh:
    def __init__(self, vertices, faces, show = True):
        """ Construct mesh and show it if show is True """
        self.faces = faces
        self.vertices = vertices
        if show:
            _scene = VTKScene()
            _scene.add(self).show()
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
    
    def apply_style(self, actor: vtk.vtkActor, opacity: float = None, **kw):
        """Apply styling to existing VTK actor. This violates Open-Closed Principle because
        when we add a new style to a mesh we need to change this method, but I doubt this will happen much."""
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
        if cmap is not None:
            lut = get_vtk_colormap(cmap)
            mapper.SetLookupTable(lut)

        # Create scalar bar
        scalar_bar = _make_scalar_bar(mapper.GetLookupTable())
        actor._scalar_bar = scalar_bar  # Store reference for later use
        
        return [actor, scalar_bar]

    def apply_style(self, actor: vtk.vtkActor, cmap: str = None, opacity: float = None, **kw):
        """Apply styling to existing VTK actor."""
        # This method is called for each actor associated with the object.
        # We only want to act when called with the main mesh actor, which has a PolyDataMapper.
        if not isinstance(actor.GetMapper(), vtk.vtkPolyDataMapper):
            return

        # Apply base mesh styling first
        self.mesh.apply_style(actor, opacity=opacity, **kw)
        
        mapper = actor.GetMapper()
        
        # Update colormap if requested
        if cmap is not None:
            if mapper:
                lut = get_vtk_colormap(cmap)
                mapper.SetLookupTable(lut)
                
                # Also update the scalar bar
                if hasattr(actor, '_scalar_bar') and actor._scalar_bar:
                    actor._scalar_bar.SetLookupTable(lut)
        
        if mapper:
            poly = mapper.GetInput()
            scalars = vtk.vtkFloatArray()
            scalars.SetName("StaticData")
            scalars.SetNumberOfComponents(1)
            scalars.SetNumberOfTuples(len(self.data))
            
            for i, v in enumerate(self.data):
                scalars.SetValue(i, float(v))
            mapper.SetScalarRange(*self.data_range)
            
            poly.GetPointData().SetScalars(scalars)
            poly.Modified()

class TimeSeriesMeshData:
    """ Mesh with one timeseries per vertex, useful for plotting. """
    def __init__(self, mesh: Mesh, timeseries: np.ndarray, current_timepoint: int = 0):
        self.mesh = mesh
        self.timeseries = timeseries
        
        # Check timeseries dimensions
        assert len(timeseries.shape) == 2, f"Expected 2D timeseries array, got shape {timeseries.shape}"
        assert timeseries.shape[0] == mesh.vertices.shape[0], f"Timeseries vertices dimension {timeseries.shape[0]} must match number of vertices {mesh.vertices.shape[0]}"

        # Internal StaticMeshData for the current time slice
        self._static_view = StaticMeshData(mesh, timeseries[:, 0])
        # The colormap range must span the entire timeseries
        self._static_view.data_range = (timeseries.min(), timeseries.max())
        
        self.set_timepoint(current_timepoint)
    
    def set_timepoint(self, t: int):
        """Set the current timepoint."""
        if not (0 <= t < self.timeseries.shape[1]):
            raise IndexError(f"Timepoint {t} is out of bounds.")
        self.current_timepoint = t
        self._static_view.data = self.timeseries[:, t]

    def to_vtk(self, cmap: str = "viridis", opacity: float = 1.0, **kw) -> List[vtk.vtkProp]:
        """Convert time-series mesh to VTK actors by delegating to the static view."""
        return self._static_view.to_vtk(cmap=cmap, opacity=opacity, **kw)
    
    def apply_style(self, actor: vtk.vtkActor, **kw):
        """Apply styling by delegating to the static view."""
        # The mesh part of styling
        self.mesh.apply_style(actor, **kw)
        
        # The data part of styling
        self._static_view.apply_style(actor, **kw)
    
    def update_timeseries_actor(self, actor: vtk.vtkActor):
        """Update actor with current timepoint data by re-applying style."""
        # We assume the style is cached on the scene's handle and not passed here.
        # We need to re-apply the style to update the scalars.
        # A bit of a hack: apply_style will be called with no new style arguments,
        # so it will just re-apply the current data.
        self._static_view.apply_style(actor)

def _make_scalar_bar(lut: vtk.vtkLookupTable, title: str = "", n_labels: int = 5):
    """Helper to create scalar bar."""
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(title)
    bar.SetNumberOfLabels(n_labels)
    bar.UnconstrainedFontSizeOn()
    return bar


