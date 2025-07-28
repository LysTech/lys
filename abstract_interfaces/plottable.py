from abc import ABC, abstractmethod
from typing import List, Union
import vtk


class Plottable(ABC):
    """
    Abstract base class for objects that can be plotted in a VTKScene.
    
    Any class that inherits from Plottable must implement:
    - to_vtk(): Convert the object to VTK actors/props
    - apply_style(): Apply styling to existing VTK actors
    """
    
    @abstractmethod
    def to_vtk(self, **style) -> Union[vtk.vtkProp, List[vtk.vtkProp]]:
        """
        Convert the object to VTK actor(s) or prop(s).
        
        Returns:
            A single VTK prop or a list of VTK props that can be added to a renderer.
        """
        pass
    
    @abstractmethod
    def apply_style(self, actor: vtk.vtkActor, **style) -> None:
        """
        Apply styling to an existing VTK actor.
        
        Args:
            actor: The VTK actor to style
            **style: Style parameters to apply
        """
        pass
