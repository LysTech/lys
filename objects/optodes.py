from typing import List, Tuple, Union
import numpy as np
import vtk

from lys.interfaces.plottable import Plottable

#TODO: implement create_flow2_optodes_from_volume


class Points(Plottable):
    def __init__(self, coordinates: List[Tuple[float, float, float]], 
                 color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
                 radius: float = 1.0):
        """
        Initialize Points with 3D coordinates, color, and radius.
        
        Args:
            coordinates: List of (x,y,z) tuples
            color: RGB tuple in range [0,1]
            radius: Size of the spheres
        """
        self.coordinates = np.array(coordinates)
        self.color = color
        self.radius = radius
        
    def to_vtk(self, **style) -> List[vtk.vtkActor]:
        """
        Convert points to VTK sphere actors.
        
        Returns:
            List of VTK actors, one for each point
        """
        actors = []
        for coord in self.coordinates:
            # Create sphere source
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(coord)
            sphere.SetRadius(self.radius)
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            
            # Create mapper
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())
            
            # Create actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # Set initial color
            actor.GetProperty().SetColor(*self.color)
            
            actors.append(actor)
            
        return actors
    
    def apply_style(self, actor: vtk.vtkActor, **updates):
        """
        Update the visual properties of the sphere actors.
        
        Args:
            actor: VTK actor to update
            updates: Dictionary of style updates including:
                - color: RGB tuple
                - radius: float
                - opacity: float in range [0,1]
        """
        if 'color' in updates:
            actor.GetProperty().SetColor(*updates['color'])
            
        if 'opacity' in updates:
            actor.GetProperty().SetOpacity(updates['opacity'])
            
        if 'radius' in updates:
            # Need to recreate the sphere source with new radius
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(actor.GetCenter())
            sphere.SetRadius(updates['radius'])
            sphere.SetPhiResolution(20)
            sphere.SetThetaResolution(20)
            
            # Update the mapper with new geometry
            mapper = actor.GetMapper()
            mapper.SetInputConnection(sphere.GetOutputPort())

class Optodes(Plottable):
    def __init__(self, source_coords: List[Tuple[float, float, float]], 
                 detector_coords: List[Tuple[float, float, float]],
                 radius: float = 1.0):
        """
        Initialize Optodes with source and detector coordinates.
        
        Args:
            source_coords: List of (x,y,z) tuples for sources
            detector_coords: List of (x,y,z) tuples for detectors
            radius: Size of the spheres
        """
        # Sources are always red (1,0,0), detectors are always blue (0,0,1)
        self.sources = Points(source_coords, color=(1.0, 0.0, 0.0), radius=radius)
        self.detectors = Points(detector_coords, color=(0.0, 0.0, 1.0), radius=radius)
        
    def to_vtk(self, **style) -> List[vtk.vtkActor]:
        """
        Convert optodes to VTK sphere actors.
        
        Returns:
            List of VTK actors for all sources and detectors
        """
        source_actors = self.sources.to_vtk(**style)
        detector_actors = self.detectors.to_vtk(**style)
        return source_actors + detector_actors
    
    def apply_style(self, actor: vtk.vtkActor, **updates):
        """
        Update the visual properties of the sphere actors.
        Only radius can be modified - colors are fixed.
        
        Args:
            actor: VTK actor to update
            updates: Dictionary of style updates:
                - radius: float
        """
        # Only allow radius updates
        if 'radius' in updates:
            self.sources.apply_style(actor, radius=updates['radius'])
            self.detectors.apply_style(actor, radius=updates['radius'])
