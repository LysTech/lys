import os
import warnings
import numpy as np
from scipy.io import loadmat
import vtk
from vtk.util import numpy_support
from typing import List, Optional
from pathlib import Path

from lys.utils.paths import lys_subjects_dir
from lys.objects.segmentation import load_charm_segmentation
from lys.objects.atlas import Atlas
from lys.utils.coordinates import align_with_csf, undo_the_scaling, undo_affine_transformation, read_adjBBX_file
from lys.abstract_interfaces.plottable import Plottable
from lys.objects.eigenmodes import Eigenmode, load_eigenmodes

"""
Defines classes related to meshes:
- Mesh
- StaticMeshData
- TimeSeriesMeshData
"""

#TODO: 
# - create our own mesh from a segmentation Volume
# - load_unMNI_mesh should just load the mesh from disk if it exists? NO: takes 1.65 seconds to load MNI+unMNI, its fast enough!


def load_unMNI_mesh(patient: str) -> Optional["Mesh"]:
    """ 
    patient: e.g. "P03"
    segmentation: Atlas object, required for alignment to native space

    If the mesh file does not exist, prints a warning and returns None.
    If the segmentation does not exist, prints a warning and returns None.
    """
    from lys.visualization.plot3d import VTKScene
    mesh_path = Path(mni_mesh_path(patient))
    if not mesh_path.exists(): #TODO: is this bad code? I'm not sure.
        warnings.warn(f"Mesh file not found for patient {patient} at {mesh_path}. Returning None.")
        return None
    mni_mesh = from_mat(str(mesh_path))
    mni_mesh.vertices = mni_mesh.vertices[:, ::-1] #flip XYZ -> ZYX for matlab fortran order compatibility
    segmentation = load_charm_segmentation(patient)
    if segmentation is None:
        warnings.warn(f"Segmentation not found for patient {patient}. Returning None.")
        return None
    nativespace_mesh = mni_to_nativespace(mni_mesh, segmentation, patient)
    eigenmodes = load_eigenmodes(patient)
    nativespace_mesh.eigenmodes = eigenmodes
    VTKScene().add(segmentation).add(nativespace_mesh).format(segmentation, opacity=0.02).show() # plot for visual check
    return nativespace_mesh



class Mesh(Plottable):
    def __init__(self, vertices, faces, eigenmodes: List[Eigenmode] | None = None):
        self.faces = faces
        self.vertices = vertices
        self.eigenmodes = eigenmodes
        self._check_mesh()

    def downsample(self, target_vertices: int) -> 'Mesh':
        """
        Return a new Mesh with a subset of vertices and corresponding faces.
        This method downsamples the mesh to approximately `target_vertices`.
        Args:
            target_vertices (int): The target number of vertices for the downsampled mesh.
        Returns:
            Mesh: The downsampled mesh.
        """
        if target_vertices >= len(self.vertices):
            return Mesh(self.vertices.copy(), self.faces.copy())

        decimate = vtk.vtkQuadricDecimation()
        
        # Convert our mesh to a VTK PolyData object
        poly_data = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(self.vertices))
        poly_data.SetPoints(points)

        cells = vtk.vtkCellArray()
        # VTK requires a flat array of (n_points, p1, p2, ..., pn, n_points, p1, ...)
        faces_flat = np.hstack((np.full((self.faces.shape[0], 1), 3), self.faces)).flatten()
        cells.SetCells(poly_data.GetNumberOfPoints(), numpy_support.numpy_to_vtkIdTypeArray(faces_flat))
        poly_data.SetPolys(cells)
        
        decimate.SetInputData(poly_data)
        
        # Calculate the decimation factor
        current_vertices = self.vertices.shape[0]
        reduction = (current_vertices - target_vertices) / current_vertices
        decimate.SetTargetReduction(reduction)
        
        decimate.Update()
        
        decimated_poly = decimate.GetOutput()
        
        # Extract vertices and faces from the decimated polydata
        new_vertices = numpy_support.vtk_to_numpy(decimated_poly.GetPoints().GetData())
        
        new_faces_vtk = decimated_poly.GetPolys().GetData()
        new_faces_numpy = numpy_support.vtk_to_numpy(new_faces_vtk)
        
        # Reshape the flat array from VTK back into a faces array
        new_faces = new_faces_numpy.reshape(-1, 4)[:, 1:]

        return Mesh(new_vertices, new_faces)

    def _check_mesh(self):
        """ Check mesh properties:
            - 0-indexing
            - not an empty mesh
        """
        assert len(self.faces) != 0, "Mesh has no faces"
        min_face_idx = min([min(f) for f in self.faces])
        assert min_face_idx == 0, f"Expected 0-indexed faces, got {min_face_idx}"
        if self.eigenmodes is None:
            warnings.warn("This mesh has no eigenmodes.")

    def to_vtk(self, cmap: str = "viridis", opacity: float = 1.0, **kw) -> vtk.vtkActor:
        """Convert mesh to VTK actor."""
        poly = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for v in self.vertices:
            pts.InsertNextPoint(v)
        poly.SetPoints(pts)

        polys = vtk.vtkCellArray()
        # The faces must be converted to a VTK-compatible format
        if self.faces.size > 0:
            # VTK requires a flat array of (n_points, p1, p2, ..., pn, n_points, p1, ...)
            faces_flat = np.hstack((np.full((self.faces.shape[0], 1), 3), self.faces)).flatten()
            id_type_array = numpy_support.numpy_to_vtkIdTypeArray(faces_flat)
            polys.SetCells(self.vertices.shape[0], id_type_array)
        poly.SetPolys(polys)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        return actor
    
    def apply_style(self, actor: vtk.vtkActor, opacity: float = 1.0, **kw):
        """Apply styling to existing VTK actor. This violates Open-Closed Principle because
        when we add a new style to a mesh we need to change this method, but I doubt this will happen much."""
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)


class StaticMeshData(Plottable):
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
        from lys.visualization.utils import _make_scalar_bar, get_vtk_colormap
        
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

    def apply_style(self, actor: vtk.vtkActor, cmap: str | None = None, opacity: float | None = None, **kw):
        """Apply styling to existing VTK actor."""
        from lys.visualization.utils import get_vtk_colormap
        
        # This method is called for each actor associated with the object.
        # We only want to act when called with the main mesh actor, which has a PolyDataMapper.
        if not isinstance(actor.GetMapper(), vtk.vtkPolyDataMapper):
            return

        # Apply base mesh styling first - only pass opacity if it's not None
        if opacity is not None:
            self.mesh.apply_style(actor, opacity=opacity, **kw)
        else:
            self.mesh.apply_style(actor, **kw)
        
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


class TimeSeriesMeshData(Plottable):
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


def mni_to_nativespace(mesh: Mesh, segmentation: Atlas, patient: str):
    """
    Transform mesh from MNI space to native space, assuming vertices are in ZYX order.
    All coordinate operations (translations, scaling, affine) are applied in ZYX order:
    - vertices[:, 0] is Z, vertices[:, 1] is Y, vertices[:, 2] is X
    - All translation and scaling vectors must be reordered accordingly
    """
    tissue = 2  # 2 is the white matter, 3 is CSF
    vol = segmentation.array.flatten()
    vol[vol != tissue] = 0
    vol = vol.reshape(segmentation.array.shape)
    vertices = mesh.vertices
    faces = mesh.faces

    affine_matrix, x_scales, y_scales, z_scales = read_adjBBX_file(patient)

    # Reorder translation vectors for ZYX
    vertices = vertices - np.array([128, 128, 128])  # shift AC to origin (ZYX order)
    # Reorder scales for ZYX: (z_scales, y_scales, x_scales)
    vertices = undo_the_scaling(vertices, z_scales, y_scales, x_scales)
    # Permute ZYX -> XYZ for affine, then back
    vertices_xyz = vertices[:, [2, 1, 0]]
    vertices_xyz = undo_affine_transformation(vertices_xyz, affine_matrix)
    vertices = vertices_xyz[:, [2, 1, 0]]  # back to ZYX
    vertices = vertices + np.array([96, 128, 128])  # AC point starts at O -> move it (ZYX order)
    #vertices = align_with_csf(vertices, vol, tissue)
    mesh.vertices = vertices
    return mesh


def mni_mesh_path(patient: str) -> str:
    root = lys_subjects_dir()
    return os.path.join(root, patient, "anat", "meshes", f"{patient}_EIGMOD_MPR_IIHC_MNI_WM_LH_edited_again_RECOSM_D32k.mat")


def from_mat(mat_file_path: str, **kwargs) -> Mesh:
    """ Mesh constructor: Load a Mesh from a MATLAB .mat file """
    mdata = loadmat(mat_file_path)
    vertices = mdata["vertices"].astype(float)  # shape (N,3)
    faces = mdata["faces"] - 1
    min_face_idx = min([min(f) for f in faces])
    assert min_face_idx == 0, f"Expected 1-indexed MATLAB faces, but minimum index after conversion is {min_face_idx}"
    return Mesh(vertices, faces, **kwargs)

