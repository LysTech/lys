import vtk
import numpy as np
from typing import Dict, Optional, List, Tuple
import colorsys
from vtk.util import numpy_support

from lys.visualization.plot3d import VTKScene

#TODO: this doesn't look good with the segmentation in demo.py,
# why don't I see a full skull?


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
        self.legend_bounds = {}
        self.vtk_volume = None
        self.opacity_tf = None
        
        # Generate default colors
        self._generate_default_colors()
        
        self._check_atlas()
        
        if show:
            self.scene = VTKScene()
            self.scene.add(self)
            
            # The interactor is not available until the scene is created,
            # so we set up the interaction after adding the atlas to the scene.
            # Wait for the scene to be shown before setting up interaction
            self.scene.show()
            if hasattr(self.scene, '_qt_window') and self.scene._qt_window is not None:
                interactor = self.scene._qt_window.vtkWidget
                self.setup_interaction(interactor)
    
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
        
        legend_actors = self._create_interactive_legend()
        
        return [self.vtk_volume] + legend_actors

    def _create_interactive_legend(self) -> List[vtk.vtkActor2D]:
        """Create a clickable legend for toggling region visibility.

        Each entry is now a single `vtkTextActor` whose background is filled with
        the colour assigned to the corresponding label.  The text itself is
        rendered in black for readability.  We keep the same high-level layout
        logic so that the previously implemented click detection continues to
        work (bounds are stored per entry in `self.legend_bounds`).
        """

        legend_actors: List[vtk.vtkActor2D] = []

        # Normalised viewport coordinates for the legend layout.
        start_x = 0.82
        start_y = 0.95
        box_height = 0.04        # Height of each text box (in normalised units)
        padding = 0.015          # Vertical padding between entries
        entry_height = box_height + padding

        for i, label in enumerate(self.unique_labels):
            y_pos = start_y - i * entry_height

            # Do not draw entries that would fall off-screen.
            if y_pos < 0.05:
                print("Warning: Too many labels to display all in the legend.")
                break

            # Text actor with coloured background.
            text_actor = vtk.vtkTextActor()
            text_actor.SetInput(self.label_names[label])

            # Position in normalised viewport coordinates.
            text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            text_actor.SetPosition(start_x, y_pos)

            # Style the text and background.
            text_prop = text_actor.GetTextProperty()
            text_prop.SetColor(0.0, 0.0, 0.0)          # Black text
            text_prop.SetFontSize(14)
            text_prop.SetFontFamilyToArial()

            r, g, b = self.colors[label]

            # Background colour matching the region colour.
            # Depending on the VTK version, background is controlled either via
            # the text property or directly on the actor.  We set both for
            # maximum compatibility.
            if hasattr(text_prop, "SetBackgroundColor"):
                text_prop.SetBackgroundColor(r, g, b)
                text_prop.SetBackgroundOpacity(1.0)
            if hasattr(text_actor, "SetBackgroundColor"):
                text_actor.SetBackgroundColor(r, g, b)
                text_actor.SetBackgroundOpacity(1.0)

            # Vertically centre the text inside its box.
            text_prop.SetVerticalJustificationToCentered()

            # Get the actual bounds of the text actor
            text_actor.GetTextProperty().SetJustificationToLeft()
            text_actor.GetTextProperty().SetVerticalJustificationToCentered()
            
            # Force the text actor to compute its bounds
            text_actor.GetTextProperty().SetFontSize(14)
            text_actor.GetTextProperty().SetFontFamilyToArial()
            text_actor.GetTextProperty().SetBold(0)
            text_actor.GetTextProperty().SetItalic(0)
            text_actor.GetTextProperty().SetShadow(0)
            text_actor.GetTextProperty().SetOpacity(1.0)
            
            # Get the actual bounds of the text actor
            bounds = text_actor.GetBounds()
            if bounds is not None:
                # Convert bounds to normalized viewport coordinates
                viewport = text_actor.GetViewport()
                if viewport is not None:
                    size = viewport.GetSize()
                    if size[0] > 0 and size[1] > 0:
                        x1 = bounds[0] / size[0]
                        y1 = bounds[2] / size[1]
                        x2 = bounds[1] / size[0]
                        y2 = bounds[3] / size[1]
                        # Add some padding to make clicking easier
                        padding = 0.01
                        self.legend_bounds[label] = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
                    else:
                        # Fallback to approximate bounds if viewport size is not available
                        self.legend_bounds[label] = (start_x, y_pos, 0.99, y_pos + box_height)
                else:
                    # Fallback to approximate bounds if viewport is not available
                    self.legend_bounds[label] = (start_x, y_pos, 0.99, y_pos + box_height)
            else:
                # Fallback to approximate bounds if bounds are not available
                self.legend_bounds[label] = (start_x, y_pos, 0.99, y_pos + box_height)

            legend_actors.append(text_actor)

        return legend_actors

    def setup_interaction(self, interactor: vtk.vtkRenderWindowInteractor):
        """Set up the interaction callback for the legend."""
        self.interactor = interactor
        self._click_callback_id = self.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_legend_click
        )

    def _on_legend_click(self, obj, event):
        """Handle clicks on the legend to toggle visibility."""
        try:
            click_pos = self.interactor.GetEventPosition()
            renderer = self.interactor.FindPokedRenderer(click_pos[0], click_pos[1])
            
            # Convert display coordinates to normalized viewport coordinates
            if renderer is None:
                print("Warning: No renderer found at click position")
                return
                
            size = renderer.GetSize()
            if size[0] == 0 or size[1] == 0:
                print("Warning: Invalid renderer size")
                return
                
            norm_x = click_pos[0] / size[0]
            norm_y = click_pos[1] / size[1]
            
            # Debug print for click position
            print(f"Click at normalized coordinates: ({norm_x:.3f}, {norm_y:.3f})")
            
            # Check each label's bounds with a small tolerance
            tolerance = 0.005
            for label, (x1, y1, x2, y2) in self.legend_bounds.items():
                if (x1 - tolerance <= norm_x <= x2 + tolerance and 
                    y1 - tolerance <= norm_y <= y2 + tolerance):
                    print(f"Click detected on label {label}: {self.label_names[label]}")
                    self.toggle_label_visibility(label)
                    return
                    
            print("No label bounds matched the click position")
            
        except Exception as e:
            print(f"Error in legend click handler: {str(e)}")

    def toggle_label_visibility(self, label: int):
        """Toggle the visibility of a specific label."""
        current_opacity = self.opacity_tf.GetValue(label)
        if current_opacity > 0:
            self.opacity_tf.AddPoint(label, 0)
            if label in self.visible_labels:
                self.visible_labels.remove(label)
        else:
            # Restore to default opacity (from to_vtk)
            self.opacity_tf.AddPoint(label, 0.5)
            self.visible_labels.add(label)
            
        self.opacity_tf.Modified()
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