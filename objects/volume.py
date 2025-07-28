import vtk
import json
import numpy as np
import pprint

from lys.abstract_interfaces.plottable import Plottable


class Volume(Plottable):
    def __init__(self, array, metadata={}):
        """ Construct volume and show it if show is True """
        self.array = array
        self.metadata = metadata
        self._check_volume()
    
    def _check_volume(self):
        """ Check volume properties:
            - 3D array shape
            - Valid data type
        """
        assert len(self.array.shape) == 3, f"Expected 3D array, got shape {self.array.shape}"
        assert np.issubdtype(self.array.dtype, np.number), f"Expected numeric array, got {self.array.dtype}"
    
    def to_vtk(self, opacity: float = 0.3, **kw) -> list:
        """Convert volume to VTK actor (required by plotting code)"""
        # Create VTK image data from numpy array
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(self.array.shape)
        vtk_data.SetSpacing(1.0, 1.0, 1.0)  # Could be customized based on metadata
        vtk_data.SetOrigin(0.0, 0.0, 0.0)   # Could be customized based on metadata
        
        # Convert numpy array to VTK format
        flat_array = self.array.flatten(order='F')  # VTK uses Fortran ordering
        vtk_array = vtk.vtkFloatArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(len(flat_array))
        for i, val in enumerate(flat_array):
            vtk_array.SetValue(i, float(val))
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Use marching cubes to extract isosurface
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_data)
        print("We are plotting isosurfaces of the volume using marching cubes.")
        
        # Set iso value to mean
        iso_value = float(np.mean(self.array))
        marching_cubes.SetValue(0, iso_value)
        marching_cubes.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(marching_cubes.GetOutputPort())
        
        # Set scalar range for the mapper
        data_range = (float(self.array.min()), float(self.array.max()))
        mapper.SetScalarRange(*data_range)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        
        return [actor]
    
    def apply_style(self, actor: vtk.vtkActor, opacity: float = None, color: tuple = None, 
                   data_range: tuple = None, cmap: str = None, **kw):
        """Apply styling to existing VTK actor."""
        from lys.visualization.utils import get_vtk_colormap
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)
        if color is not None:
            actor.GetProperty().SetColor(*color)
        
        # Handle scalar range updates
        if data_range is not None:
            mapper = actor.GetMapper()
            if mapper:
                mapper.SetScalarRange(*data_range)
        
        # Handle colormap updates
        if cmap is not None:
            mapper = actor.GetMapper()
            if mapper:
                lut = get_vtk_colormap(cmap)
                mapper.SetLookupTable(lut)
                
                # Update scalar bar if it exists
                if hasattr(actor, '_scalar_bar'):
                    actor._scalar_bar.SetLookupTable(lut)

def from_jnii(file_path: str, **kwargs) -> Volume:
    with open(file_path, "r") as f:
        raw = json.load(f)
    
    print("NIFTI Header:")
    pprint.pprint(raw['struct']['NIFTIHeader'])

    #volumeData = np.array(raw["struct"]["NIFTIData"]["_ArrayData_"], dtype=float)
    #volumeData = volumeData.reshape(raw["struct"]["NIFTIHeader"]["Dim"])
    flat = np.asarray(raw["struct"]["NIFTIData"]["_ArrayData_"], dtype=float)
    vol = flat.reshape((192, 256, 256), order='F')
    # --------‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑ here ------------
    # use Fortran order when turning the 1‑D list back into (x,y,z)
    volumeData = flat.reshape((192, 256, 256), order='F') #try the same stuff as v1
    # ----------------------------------------------------------
    print("Volume data gets divide by 50.")
    volumeData = volumeData / 50.0
    return Volume(volumeData, **kwargs)
