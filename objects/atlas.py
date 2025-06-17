import vtk
import numpy as np
from typing import Dict, Optional, List, Tuple
import colorsys
from vtk.util import numpy_support

from lys.visualization.plot3d import VTKScene


class Atlas:
    """A volume with discrete regions, each having a unique integer label."""
    
    def __init__(self, array: np.ndarray, metadata: dict = {}, label_names: Optional[Dict[int, str]] = None, show: bool = True):
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
        
        self.visible_labels = set(self.unique_labels)
        self.legend_items = {}  # Store legend item actors
        self.vtk_volume = None
        self.opacity_tf = None
        self.renderer = None
        
        # Generate default colors
        self._generate_default_colors()
        
        self._check_atlas()
        
        if show:
            self.scene = VTKScene()
            self.scene.add(self)
            self.scene.show()
    
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
    
    def _create_checkbox_actor(self, label: int, x_pos: float, y_pos: float, size: float) -> vtk.vtkActor2D:
        """Create a checkbox actor for the legend."""
        checkbox_points = vtk.vtkPoints()
        checkbox_lines = vtk.vtkCellArray()
        
        # Checkbox outline (square)
        checkbox_points.InsertNextPoint(0, 0, 0)
        checkbox_points.InsertNextPoint(1, 0, 0)
        checkbox_points.InsertNextPoint(1, 1, 0)
        checkbox_points.InsertNextPoint(0, 1, 0)
        
        # Create outline
        checkbox_lines.InsertNextCell(5)
        checkbox_lines.InsertCellPoint(0)
        checkbox_lines.InsertCellPoint(1)
        checkbox_lines.InsertCellPoint(2)
        checkbox_lines.InsertCellPoint(3)
        checkbox_lines.InsertCellPoint(0)
        
        # Add check mark if visible
        if label in self.visible_labels:
            # Check mark points
            checkbox_points.InsertNextPoint(0.2, 0.5, 0)  # Point 4
            checkbox_points.InsertNextPoint(0.4, 0.3, 0)  # Point 5
            checkbox_points.InsertNextPoint(0.8, 0.7, 0)  # Point 6
            
            # Check mark lines
            checkbox_lines.InsertNextCell(3)
            checkbox_lines.InsertCellPoint(4)
            checkbox_lines.InsertCellPoint(5)
            checkbox_lines.InsertCellPoint(6)
        
        checkbox_polydata = vtk.vtkPolyData()
        checkbox_polydata.SetPoints(checkbox_points)
        checkbox_polydata.SetLines(checkbox_lines)
        
        checkbox_mapper = vtk.vtkPolyDataMapper2D()
        checkbox_mapper.SetInputData(checkbox_polydata)
        
        checkbox_actor = vtk.vtkActor2D()
        checkbox_actor.SetMapper(checkbox_mapper)
        checkbox_actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # Brighter color
        checkbox_actor.GetProperty().SetLineWidth(3)  # Thicker lines
        
        # Position checkbox
        checkbox_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        checkbox_actor.GetPositionCoordinate().SetValue(x_pos, y_pos - size/2)
        checkbox_actor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        checkbox_actor.GetPosition2Coordinate().SetValue(size, size)
        
        return checkbox_actor

    def _create_text_actor(self, label: int, x_pos: float, y_pos: float, checkbox_size: float) -> vtk.vtkTextActor:
        """Create a text actor with colored background for the legend."""
        text_actor = vtk.vtkTextActor()
        
        # Add padding to the text
        text_actor.SetInput(f" {self.label_names[label]} ")
        text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        text_actor.SetPosition(x_pos + checkbox_size + 0.015, y_pos)
        
        # Style the text and its background
        r, g, b = self.colors[label]
        text_prop = text_actor.GetTextProperty()
        text_prop.SetBackgroundColor(r, g, b)
        text_prop.SetColor(1.0, 1.0, 1.0)  # White text
        text_prop.SetFontSize(14)  # Larger font size
        text_prop.SetFontFamilyToArial()
        text_prop.SetJustificationToLeft()
        text_prop.SetVerticalJustificationToCentered()
        text_prop.SetBold(1)  # Make text bold for better visibility
        
        if label in self.visible_labels:
            text_prop.SetBackgroundOpacity(1.0)
        else:
            text_prop.SetOpacity(0.3)
            text_prop.SetBackgroundOpacity(0.3)
        
        return text_actor

    def to_vtk(self, opacity: float = 0.5, colors: Optional[Dict[int, Tuple[float, float, float]]] = None, **kw) -> List[vtk.vtkProp]:
        """
        Convert atlas to a VTK volume actor and an interactive legend.
        
        Parameters:
        -----------
        opacity : float
            Opacity for all regions
        colors : dict, optional
            Custom color mapping for regions {label: (r, g, b)}
        
        Returns:
        --------
        List containing the volume actor and legend actors
        """
        self._reset_view_state()
        if colors is not None:
            self.colors.update(colors)
        
        # Create VTK image data
        vtk_data = vtk.vtkImageData()
        vtk_data.SetDimensions(self.array.shape)
        
        spacing = self.metadata.get('spacing', (1.0, 1.0, 1.0))
        origin = self.metadata.get('origin', (0.0, 0.0, 0.0))
        vtk_data.SetSpacing(spacing)
        vtk_data.SetOrigin(origin)
        
        flat_array = self.array.flatten(order='F').astype(np.int32)
        vtk_array = numpy_support.numpy_to_vtk(flat_array, deep=True, array_type=vtk.VTK_INT)
        vtk_data.GetPointData().SetScalars(vtk_array)
        
        # Create transfer functions for color and opacity
        color_tf = vtk.vtkColorTransferFunction()
        self.opacity_tf = vtk.vtkPiecewiseFunction()
        
        color_tf.AddRGBPoint(0, 0, 0, 0, 0.5, 0.0)
        self.opacity_tf.AddPoint(0, 0)
        
        for label in self.unique_labels:
            r, g, b = self.colors[label]
            color_tf.AddRGBPoint(label, r, g, b)
            if label in self.visible_labels:
                self.opacity_tf.AddPoint(label, opacity)
            else:
                self.opacity_tf.AddPoint(label, 0)

        # Create volume properties
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetColor(color_tf)
        volume_property.SetScalarOpacity(self.opacity_tf)
        volume_property.SetInterpolationTypeToNearest()
        volume_property.ShadeOn()
        
        # Create volume mapper and volume
        mapper = vtk.vtkSmartVolumeMapper()
        mapper.SetInputData(vtk_data)
        
        self.vtk_volume = vtk.vtkVolume()
        self.vtk_volume.SetMapper(mapper)
        self.vtk_volume.SetProperty(volume_property)

        # Create legend actors
        legend_actors = []
        
        # Starting position for the legend
        x_pos = 0.82
        y_start = 0.95
        y_spacing = 0.055
        checkbox_size = 0.05
        
        for i, label in enumerate(self.unique_labels):
            y_pos = y_start - i * y_spacing
            
            if y_pos < 0.05:  # Don't create widgets that would be off-screen
                print("Warning: Too many labels to display all in the legend.")
                break
            
            # Create checkbox actor
            checkbox_actor = self._create_checkbox_actor(label, x_pos, y_pos, checkbox_size)
            legend_actors.append(checkbox_actor)
            
            # Create text actor with colored background
            text_actor = self._create_text_actor(label, x_pos, y_pos, checkbox_size)
            legend_actors.append(text_actor)
            
            # Store actors for interaction
            self.legend_items[label] = {
                'checkbox': checkbox_actor,
                'text': text_actor,
                'bounds': (x_pos, y_pos - checkbox_size/2, 
                          x_pos + 0.3, y_pos + checkbox_size/2)
            }
        
        # Return all actors
        return [self.vtk_volume] + legend_actors

    def setup_interaction(self, iren: vtk.vtkRenderWindowInteractor,
                                ren:  vtk.vtkRenderer):
        """Attach legend + click handler to an existing scene."""
        # guard: reuse across scenes
        if getattr(self, "_click_callback_id", None) is not None:
            try: self.interactor.RemoveObserver(self._click_callback_id)
            except RuntimeError: pass

        self.interactor, self.renderer = iren, ren
        if not self.legend_items:                # build legend once
            self._create_checkbox_legend()

        self._click_callback_id = iren.AddObserver(
            "LeftButtonPressEvent", self._on_click)

    def _create_checkbox_legend(self):
        """Create checkbox-style legend with clickable elements."""
        if not self.renderer:
            return
            
        # Starting position for the legend
        x_pos = 0.82  # Normalized viewport coordinates
        y_start = 0.95
        y_spacing = 0.055  # Increased spacing to accommodate larger checkboxes
        checkbox_size = 0.05  # Made checkbox 67% larger
        
        for i, label in enumerate(self.unique_labels):
            y_pos = y_start - i * y_spacing
            
            # Don't create widgets that would be off-screen
            if y_pos < 0.05:
                print("Warning: Too many labels to display all in the legend.")
                break
            
            # Create checkbox (square outline with check mark)
            checkbox_points = vtk.vtkPoints()
            checkbox_lines = vtk.vtkCellArray()
            
            # Checkbox outline (square)
            checkbox_points.InsertNextPoint(0, 0, 0)
            checkbox_points.InsertNextPoint(1, 0, 0)
            checkbox_points.InsertNextPoint(1, 1, 0)
            checkbox_points.InsertNextPoint(0, 1, 0)
            
            # Create outline
            checkbox_lines.InsertNextCell(5)
            checkbox_lines.InsertCellPoint(0)
            checkbox_lines.InsertCellPoint(1)
            checkbox_lines.InsertCellPoint(2)
            checkbox_lines.InsertCellPoint(3)
            checkbox_lines.InsertCellPoint(0)
            
            # Add check mark if visible
            if label in self.visible_labels:
                # Check mark points
                checkbox_points.InsertNextPoint(0.2, 0.5, 0)  # Point 4
                checkbox_points.InsertNextPoint(0.4, 0.3, 0)  # Point 5
                checkbox_points.InsertNextPoint(0.8, 0.7, 0)  # Point 6
                
                # Check mark lines
                checkbox_lines.InsertNextCell(3)
                checkbox_lines.InsertCellPoint(4)
                checkbox_lines.InsertCellPoint(5)
                checkbox_lines.InsertCellPoint(6)
            
            checkbox_polydata = vtk.vtkPolyData()
            checkbox_polydata.SetPoints(checkbox_points)
            checkbox_polydata.SetLines(checkbox_lines)
            
            checkbox_mapper = vtk.vtkPolyDataMapper2D()
            checkbox_mapper.SetInputData(checkbox_polydata)
            
            checkbox_actor = vtk.vtkActor2D()
            checkbox_actor.SetMapper(checkbox_mapper)
            checkbox_actor.GetProperty().SetColor(0.9, 0.9, 0.9)  # Brighter color
            checkbox_actor.GetProperty().SetLineWidth(3)  # Thicker lines
            
            # Position checkbox
            checkbox_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            checkbox_actor.GetPositionCoordinate().SetValue(x_pos, y_pos - checkbox_size/2)
            checkbox_actor.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
            checkbox_actor.GetPosition2Coordinate().SetValue(checkbox_size, checkbox_size)
            
            self.renderer.AddActor2D(checkbox_actor)
            
            # Create text label with a colored background
            text_actor = vtk.vtkTextActor()
            
            # Add padding to the text
            text_actor.SetInput(f" {self.label_names[label]} ")
            
            text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            text_actor.SetPosition(x_pos + checkbox_size + 0.015, y_pos)
            
            # Style the text and its background
            r, g, b = self.colors[label]
            text_prop = text_actor.GetTextProperty()
            text_prop.SetBackgroundColor(r, g, b)
            text_prop.SetColor(1.0, 1.0, 1.0)  # White text
            text_prop.SetFontSize(14)  # Larger font size
            text_prop.SetFontFamilyToArial()
            text_prop.SetJustificationToLeft()
            text_prop.SetVerticalJustificationToCentered()
            text_prop.SetBold(1)  # Make text bold for better visibility
            
            if label in self.visible_labels:
                text_prop.SetBackgroundOpacity(1.0)
            else:
                text_prop.SetOpacity(0.3)
                text_prop.SetBackgroundOpacity(0.3)
            
            self.renderer.AddActor2D(text_actor)
            
            # Store all actors and bounds for this label
            self.legend_items[label] = {
                'checkbox': checkbox_actor,
                'text': text_actor,
                'bounds': (x_pos, y_pos - checkbox_size/2, 
                           x_pos + 0.3, y_pos + checkbox_size/2)  # Clickable area
            }

    def _on_click(self, obj, event):
        """Handle click events on the legend."""
        try:
            click_pos = self.interactor.GetEventPosition()
            renderer = self.interactor.FindPokedRenderer(click_pos[0], click_pos[1])
            
            if renderer is None:
                return
                
            size = renderer.GetSize()
            if size[0] == 0 or size[1] == 0:
                return
                
            norm_x = click_pos[0] / size[0]
            norm_y = click_pos[1] / size[1]
            
            # Check each label's bounds
            for label, item in self.legend_items.items():
                x1, y1, x2, y2 = item['bounds']
                if x1 <= norm_x <= x2 and y1 <= norm_y <= y2:
                    self.toggle_label_visibility(label)
                    return
                    
        except Exception as e:
            print(f"Error in click handler: {str(e)}")

    def toggle_label_visibility(self, label: int):
        """Toggle the visibility of a specific label."""
        if label in self.visible_labels:
            # Hide the region
            self.opacity_tf.AddPoint(label, 0)
            self.visible_labels.remove(label)
            visible = False
        else:
            # Show the region
            self.opacity_tf.AddPoint(label, 0.5)
            self.visible_labels.add(label)
            visible = True
        
        # Update checkbox appearance
        if label in self.legend_items:
            item = self.legend_items[label]
            
            # Update checkbox with or without check mark
            checkbox_points = vtk.vtkPoints()
            checkbox_lines = vtk.vtkCellArray()
            
            # Checkbox outline
            checkbox_points.InsertNextPoint(0, 0, 0)
            checkbox_points.InsertNextPoint(1, 0, 0)
            checkbox_points.InsertNextPoint(1, 1, 0)
            checkbox_points.InsertNextPoint(0, 1, 0)
            
            checkbox_lines.InsertNextCell(5)
            checkbox_lines.InsertCellPoint(0)
            checkbox_lines.InsertCellPoint(1)
            checkbox_lines.InsertCellPoint(2)
            checkbox_lines.InsertCellPoint(3)
            checkbox_lines.InsertCellPoint(0)
            
            if visible:
                # Add check mark
                checkbox_points.InsertNextPoint(0.2, 0.5, 0)
                checkbox_points.InsertNextPoint(0.4, 0.3, 0)
                checkbox_points.InsertNextPoint(0.8, 0.7, 0)
                
                checkbox_lines.InsertNextCell(3)
                checkbox_lines.InsertCellPoint(4)
                checkbox_lines.InsertCellPoint(5)
                checkbox_lines.InsertCellPoint(6)
            
            checkbox_polydata = vtk.vtkPolyData()
            checkbox_polydata.SetPoints(checkbox_points)
            checkbox_polydata.SetLines(checkbox_lines)
            
            item['checkbox'].GetMapper().SetInputData(checkbox_polydata)
            
            # Update opacity of text and its background
            opacity = 1.0 if visible else 0.3
            text_prop = item['text'].GetTextProperty()
            text_prop.SetOpacity(opacity)
            text_prop.SetBackgroundOpacity(opacity)
        
        if hasattr(self, 'interactor') and self.interactor is not None:
            self.interactor.GetRenderWindow().Render()

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

    def _reset_view_state(self):
        """Forget all per‑scene interaction state (legend, visibility, TF, actors)."""
        self.visible_labels = set(self.unique_labels)      # everything visible
        self.legend_items   = {}                           # no stale actors
        self.opacity_tf     = vtk.vtkPiecewiseFunction()   # fresh TF
        self.vtk_volume     = None                         # will be rebuilt

    
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