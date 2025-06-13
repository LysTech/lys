import vtk

def create_viridis_colormap() -> vtk.vtkLookupTable:
    """Create a viridis colormap lookup table."""
    lut = vtk.vtkLookupTable()
    lut.SetHueRange(0.7, 0.0)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(0.0, 1.0)
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

