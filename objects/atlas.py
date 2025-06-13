import vtk
import numpy as np
from typing import Dict, Optional, List, Tuple
import colorsys

from lys.visualization.plot3d import VTKScene


class Atlas:
    """A volume with discrete regions, each having a unique integer label."""
    
    def __init__(self, array: np.ndarray, metadata: dict, label_names: Optional[Dict[int, str]] = None, show: bool = True):
        """
        Construct an atlas volume.
        
        Parameters:
        -----------
        array : np.ndarray
            3D array with integer labels for each region
        metadata : dict
            Should contain 'spacing' and 'origin' keys
        label_names : dict, optional
            Mapping from label values to region names
        show : bool
            Whether to immediately display the atlas
        """
        self.array = array
        self.metadata = metadata
        self.label_names = label_names or {}
        
        # Get unique labels (excluding 0 which is typically background)
        self.unique_labels = np.unique(array)
        self.unique_labels = self.unique_labels[self.unique_labels != 0]
        
        # Generate default names for unnamed labels
        for label in self.unique_labels:
            if label not in self.label_names:
                self.label_names[label] = f"Region {label}"
        
        # Generate default colors
        self._generate_default_colors()
        
        self._check_atlas()
        
        if show:
            VTKScene().add(self).show()
    
    def _check_atlas(self):
        """Check atlas properties."""
        assert len(self.array.shape) == 3, f"Expected 3D array, got shape {self.array.shape}"
        assert np.issubdtype(self.array.dtype, np.integer), f"Expected integer array, got {self.array.dtype}"
    
    def _generate_default_colors(self):
        """Generate visually distinct colors for each region."""
        n_regions = len(self.unique_labels)
        self.colors = {}
        
        # Use HSV color space to generate evenly spaced hues
        for i, label in enumerate(self.unique_labels):
            hue = i / max(n_regions, 1)
            # Use high saturation and value for vivid colors
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            self.colors[label] = rgb
    
    def to_vtk(self, opacity: float = 0.8, colors: Optional[Dict[int, Tuple[float, float, float]]] = None, **kw) -> List[vtk.vtkProp]:
        """
        Convert atlas to VTK actors with legend.
        
        Parameters:
        -----------
        opacity : float
            Opacity for all regions
        colors : dict, optional
            Custom color mapping for regions {label: (r, g, b)}
        
        Returns:
        --------
        List containing the atlas actor and legend actor
        """
        if colors is not None:
            self.colors.update(colors)
        
        # Create VTK image data
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(self.array.shape)
        
        # Use metadata for spacing and origin if available
        spacing = self.metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = self.metadata.get('origin', (0.0, 0.0, 0.0))
        vtk_data.SetSpacing(spacing)
        vtk_data.SetOrigin(origin)
        
        # Convert numpy array to VTK format
        flat_array = self.array.flatten(order='F')
        vtk_array = vtk.vtkIntArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetNumberOfTuples(len(flat_array))
        for i, val in enumerate(flat_array):
            vtk_array.SetValue(i, int(val))
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Create lookup table for colors
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(int(self.array.max()) + 1)
        lut.Build()
        
        # Set background (label 0) to transparent
        lut.SetTableValue(0, 0, 0, 0, 0)
        
        # Set colors for each region
        for label in self.unique_labels:
            r, g, b = self.colors[label]
            lut.SetTableValue(int(label), r, g, b, 1.0)
        
        # Use marching cubes to extract surfaces for each region
        append_filter = vtk.vtkAppendPolyData()
        
        for label in self.unique_labels:
            # Threshold to isolate this region
            threshold = vtk.vtkImageThreshold()
            threshold.SetInputData(vtk_data)
            threshold.ThresholdBetween(label, label)
            threshold.SetInValue(label)
            threshold.SetOutValue(0)
            threshold.Update()
            
            # Extract surface
            marching_cubes = vtk.vtkMarchingCubes()
            marching_cubes.SetInputConnection(threshold.GetOutputPort())
            marching_cubes.SetValue(0, label - 0.5)
            marching_cubes.Update()
            
            # Add to combined polydata
            append_filter.AddInputConnection(marching_cubes.GetOutputPort())
        
        append_filter.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append_filter.GetOutputPort())
        mapper.SetLookupTable(lut)
        mapper.SetScalarRange(0, int(self.array.max()))
        mapper.ScalarVisibilityOn()
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetOpacity(opacity)
        
        # Store references for later use
        actor._lut = lut
        actor._colors = self.colors.copy()
        
        # Create legend
        legend = self._create_legend()
        
        return [actor, legend]
    
    def _create_legend(self) -> vtk.vtkLegendBoxActor:
        """Create a legend showing region colors and names."""
        legend = vtk.vtkLegendBoxActor()
        legend.SetNumberOfEntries(len(self.unique_labels))
        
        # Configure legend appearance
        legend.UseBackgroundOn()
        legend.SetBackgroundColor(0.1, 0.1, 0.1)
        legend.SetBackgroundOpacity(0.8)
        legend.BoxOn()
        
        # Add entries for each region
        for i, label in enumerate(self.unique_labels):
            # Create a small colored square for each region
            square = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(0, 0, 0)
            pts.InsertNextPoint(1, 0, 0)
            pts.InsertNextPoint(1, 1, 0)
            pts.InsertNextPoint(0, 1, 0)
            square.SetPoints(pts)
            
            cells = vtk.vtkCellArray()
            cells.InsertNextCell(4)
            cells.InsertCellPoint(0)
            cells.InsertCellPoint(1)
            cells.InsertCellPoint(2)
            cells.InsertCellPoint(3)
            square.SetPolys(cells)
            
            # Set color for the square
            colors = vtk.vtkUnsignedCharArray()
            colors.SetNumberOfComponents(3)
            r, g, b = self.colors[label]
            colors.InsertNextTuple([int(r*255), int(g*255), int(b*255)])
            square.GetCellData().SetScalars(colors)
            
            # Add to legend
            legend.SetEntry(i, square, self.label_names[label], (r, g, b))
        
        # Position legend on the right side
        legend.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        legend.GetPositionCoordinate().SetValue(0.85, 0.1)
        legend.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        legend.GetPosition2Coordinate().SetValue(0.15, 0.8)
        
        # Configure text properties
        legend.GetEntryTextProperty().SetFontSize(12)
        legend.GetEntryTextProperty().SetColor(1, 1, 1)
        legend.GetEntryTextProperty().SetFontFamilyToArial()
        
        return legend
    
    def apply_style(self, actor: vtk.vtkActor, opacity: Optional[float] = None, 
                   colors: Optional[Dict[int, Tuple[float, float, float]]] = None, **kw):
        """
        Apply styling to existing VTK actor.
        
        Parameters:
        -----------
        actor : vtk.vtkActor
            The actor to update
        opacity : float, optional
            New opacity value
        colors : dict, optional
            New color mapping for regions {label: (r, g, b)}
        """
        if opacity is not None:
            actor.GetProperty().SetOpacity(opacity)
        
        if colors is not None and hasattr(actor, '_lut'):
            # Update stored colors
            if hasattr(actor, '_colors'):
                actor._colors.update(colors)
            
            # Update lookup table
            lut = actor._lut
            for label, (r, g, b) in colors.items():
                if label in self.unique_labels:
                    lut.SetTableValue(int(label), r, g, b, 1.0)
            
            # Trigger update
            lut.Modified()
            actor.GetMapper().Modified()

if __name__ == "__main__":
    import nibabel as nib
    import numpy as np

    atlas_path = '/Users/thomasrialan/Documents/code/Geometric-Eigenmodes/data/P03/anat/volumes/BA_reproject_in_T1_rigid.nii.gz'
    atlas_img = nib.load(atlas_path)

    # Load atlas data
    atlas_data = atlas_img.get_fdata().astype(np.int32)
    metadata = {
        'spacing': atlas_img.header.get_zooms()[:3],
        'origin': atlas_img.affine[:3, 3]
    }

    # Create label names
    unique_labels = np.unique(atlas_data)
    label_names = {label: f"BA {label}" for label in unique_labels if label != 0}

    # Create and display atlas
    atlas = Atlas(atlas_data, metadata, label_names)