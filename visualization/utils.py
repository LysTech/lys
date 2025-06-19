import vtk
import numpy as np
import matplotlib
from typing import Dict

_custom_colormaps: Dict[str, vtk.vtkLookupTable] = {}

def register_custom_colormap(name: str, lut: vtk.vtkLookupTable):
    """Register a custom colormap."""
    _custom_colormaps[name] = lut


def get_vtk_colormap(name: str, n_colors: int = 256) -> vtk.vtkLookupTable:
    """
    Get a VTK lookup table by name.
    Searches custom colormaps first, then falls back to matplotlib.
    """
    if name in _custom_colormaps:
        return _custom_colormaps[name]
    
    # Fallback to matplotlib
    try:
        colormap = matplotlib.colormaps[name]
    except ValueError:
        raise ValueError(
            f"Unknown colormap: '{name}'. It is not a registered custom "
            "colormap or a valid matplotlib colormap."
        )

    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(n_colors)
    
    # Get colors from matplotlib colormap and set them in the vtk lookup table
    for i, x in enumerate(np.linspace(0.0, 1.0, n_colors)):
        r, g, b, _a = colormap(x) # colormap returns rgba between 0 and 1
        lut.SetTableValue(i, r, g, b, 1.0)
    
    lut.Build()
    return lut


def _make_scalar_bar(lut: vtk.vtkLookupTable,
                     title: str = "",
                     n_labels: int = 5) -> vtk.vtkScalarBarActor:
    """Helper function to create a scalar bar."""
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle(title)
    bar.SetNumberOfLabels(n_labels)
    bar.UnconstrainedFontSizeOn()      # keeps text readable at any window size
    return bar

