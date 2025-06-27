from __future__ import annotations
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from PyQt5 import QtWidgets, QtCore  # PySide6 works too – change import
from vtkmodules.qt.QVTKRenderWindowInteractor import (
    QVTKRenderWindowInteractor,
)
import vtk
import numpy as np

from lys.interfaces.plottable import Plottable

@dataclass
class _Handle:
    """Tracks a single VTK actor and its styling metadata."""
    actor: vtk.vtkProp
    source_obj: Any  # The original high‑level object (Mesh, Optodes, …)
    style: Dict[str, Any] = field(default_factory=dict)

class VTKScene:
    
    def __init__(self):
        # Core renderer and bookkeeping
        self._ren = vtk.vtkRenderer()
        self._ren.SetBackground(0.1, 0.1, 0.1)  # dark gray

        self._handles: List[_Handle] = []
        self._obj_to_handles: Dict[int, List[_Handle]] = {}
        self._objects: Dict[int, Any] = {}  # strong refs – no GC until removed
        self._timeseries_objects: Dict[int, Any] = {}
        self._current_global_timepoint = 0
        self._need_interaction: List[Any] = []

        # Qt / VTK windowing bits – created lazily in ``show``
        self._qt_app: Optional[QtWidgets.QApplication] = None
        self._qt_window: Optional[_SceneWindow] = None
    
    def _create_timeseries_slider(self, parent=None) -> QtWidgets.QSlider:
        """Create a time series slider widget."""
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, parent)
        slider.setMinimum(0)
        slider.setMaximum(self._max_timepoints() - 1)
        slider.setValue(self._current_global_timepoint)
        slider.valueChanged.connect(lambda v: self.set_timepoint(v))
        return slider

    def add(self, obj: Any, **style) -> "VTKScene":
        """Add any object that implements the Plottable interface."""
        if not isinstance(obj, Plottable):
            raise TypeError(f"Object {type(obj).__name__} must implement the Plottable interface")
        
        actors = obj.to_vtk(**style)
        if not isinstance(actors, (list, tuple)):
            actors = [actors]

        obj_handles: List[_Handle] = []
        for actor in actors:
            # Add to renderer
            if isinstance(actor, vtk.vtkScalarBarActor):
                self._ren.AddActor2D(actor)
            else:
                self._ren.AddActor(actor)

            # Bookkeeping
            h = _Handle(actor, obj, style.copy())
            self._handles.append(h)
            obj_handles.append(h)

        obj_id = id(obj)
        self._obj_to_handles.setdefault(obj_id, []).extend(obj_handles)
        self._objects.setdefault(obj_id, obj)

        if hasattr(obj, "timeseries") and hasattr(obj, "current_timepoint"):
            self._timeseries_objects[obj_id] = obj

        if self._qt_window is not None and hasattr(obj, "timeseries"):
            if not any(isinstance(w, QtWidgets.QSlider)
                       for w in self._qt_window.centralWidget().findChildren(QtWidgets.QSlider)):
                slider = self._create_timeseries_slider(self._qt_window)
                self._qt_window.centralWidget().layout().addWidget(slider, stretch=0)
        
        if hasattr(obj, "setup_interaction"):
            if self._qt_window is None:          # window not yet built
                self._need_interaction.append(obj)
            else:                                # window already open
                obj.setup_interaction(self._qt_window.vtkWidget,
                                       self._ren)

        return self

    def remove(self, obj: Any) -> "VTKScene":
        """Remove an object (and all its actors) from the scene."""
        obj_id = id(obj)
        # Delete actors from the renderer and internal lists
        for handle in self._obj_to_handles[obj_id]:
            # Use the correct removal method based on actor type
            if isinstance(handle.actor, vtk.vtkScalarBarActor):
                self._ren.RemoveActor2D(handle.actor)
            else:
                self._ren.RemoveActor(handle.actor)
            self._handles.remove(handle)

        # Clean up tracking dictionaries
        self._obj_to_handles.pop(obj_id, None)
        self._objects.pop(obj_id, None)
        self._timeseries_objects.pop(obj_id, None)

        # Ask the window to refresh if it is already visible
        self.render()
        return self
    
    def format(self, obj: Any, **updates) -> "VTKScene":
        """Update styling for any object that implements the Plottable interface."""
        if not isinstance(obj, Plottable):
            raise TypeError(f"Object {type(obj).__name__} must implement the Plottable interface")
        
        obj_id = id(obj)
        if obj_id not in self._obj_to_handles:
            raise ValueError(f"Object {obj!r} not found in scene")

        for h in self._obj_to_handles[obj_id]:
            h.style.update(updates)
            obj.apply_style(h.actor, **updates)
        
        self.render()
        return self

    def set_timepoint(self, t: int) -> "VTKScene":
        self._current_global_timepoint = t
        for obj in self._timeseries_objects.values():
            obj.set_timepoint(t)

            self._update_timeseries_visualization(obj)
        return self.render()

    def _update_timeseries_visualization(self, obj: Any):
        """Update timeseries visualization if object supports it."""
        if not hasattr(obj, 'update_timeseries_actor'):
            return
            
        obj_id = id(obj)
        if obj_id not in self._obj_to_handles:
            return
            
        for h in self._obj_to_handles[obj_id]:
            obj.update_timeseries_actor(h.actor)

    def render(self) -> "VTKScene":
        if self._qt_window is not None:
            self._qt_window.vtkWidget.GetRenderWindow().Render()
        return self

    def _ensure_qt_app(self):
        if QtWidgets.QApplication.instance() is None:
            self._qt_app = QtWidgets.QApplication(sys.argv)
        else:
            self._qt_app = QtWidgets.QApplication.instance()

    def _enable_qt_integration(self):
        """Enable Qt integration in Jupyter/IPython environments."""
        try:
            # Check if we're in an IPython environment
            from IPython import get_ipython
            ipython = get_ipython()
            
            if ipython is not None:
                # Check if Qt integration is already enabled
                if not hasattr(ipython, '_qt_integration_enabled'):
                    # Enable Qt5 integration
                    ipython.magic('gui qt5')
                    ipython._qt_integration_enabled = True
        except ImportError:
            # Not in IPython, no need for integration
            pass
        except Exception as e:
            # If enabling fails, warn but continue
            print(f"Warning: Could not enable Qt integration: {e}")
            print("You may need to run '%gui qt' manually in Jupyter/IPython")

    def show(self, size=(800, 600), title="VTK Scene", block=False):
        """Open a Qt window with the scene."""
        # 1 — make sure a Qt application exists *first*
        self._enable_qt_integration()     # Jupyter "%gui qt5", optional
        self._ensure_qt_app()             # creates / fetches QApplication

        # 2 — create (or reuse) the main window **after** QApplication
        if self._qt_window is None:
            self._qt_window = _SceneWindow(self, size, title)
        else:
            self._qt_window.resize(*size)
            self._qt_window.setWindowTitle(title)

        # 3 — hand the ready‑made interactor to objects that asked for it
        for obj in self._need_interaction:
            obj.setup_interaction(self._qt_window.vtkWidget, self._ren)
        self._need_interaction.clear()

        # 4 — first render & show
        self._ren.ResetCamera()
        self._qt_window.vtkWidget.GetRenderWindow().Render()
        self._qt_window.show()

        if block:
            self._qt_app.exec_()
        return self

    def close(self):
        """Programmatically close the Qt window and release resources."""
        if self._qt_window is not None:
            self._qt_window.close()  # triggers clean‑up hooks below
            self._qt_window = None

        # Release actors & strong refs so that Python / VTK GC can work
        self._handles.clear()
        self._obj_to_handles.clear()
        self._objects.clear()
        self._timeseries_objects.clear()

    def start_interaction(self):
        """Compatibility shim – Qt already has an event‑loop."""
        if self._qt_app is not None:
            self._qt_app.exec_()
        else:
            print("Warning: call show() first to create a Qt window")
        return self

    def _max_timepoints(self) -> int:
        """Return the largest timeseries length currently in the scene."""
        return max((obj.timeseries.shape[1]
                   for obj in self._timeseries_objects.values()), default=0)

    def screenshot(self, filename: str, size: Tuple[int, int] = (800, 600)):
        if self._qt_window is None:
            # Off‑screen fallback – create render‑window w/out Qt
            rw = vtk.vtkRenderWindow()
            rw.AddRenderer(self._ren)
            rw.SetSize(*size)
            rw.SetOffScreenRendering(1)
        else:
            rw = self._qt_window.vtkWidget.GetRenderWindow()
            old_size = rw.GetSize()
            rw.SetSize(*size)

        self._ren.ResetCamera()
        rw.Render()

        w2if = vtk.vtkWindowToImageFilter()
        w2if.SetInput(rw)
        w2if.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

        print(f"Screenshot saved to {filename}")
        if self._qt_window is not None:
            rw.SetSize(*old_size)  # restore
        return self


class _SceneWindow(QtWidgets.QMainWindow):
    """Internal QMainWindow that hosts the VTK render‑window and slider."""

    def __init__(self, scene: VTKScene, size: Tuple[int, int], title: str):
        super().__init__()
        self._scene = scene
        self.setWindowTitle(title)
        self.resize(*size)

        # Central widget – plain QWidget with vertical layout
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        # VTK widget
        self.vtkWidget = QVTKRenderWindowInteractor(central)
        self.vtkWidget.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        layout.addWidget(self.vtkWidget, stretch=1)

        # Hook renderer / render‑window back into the scene object
        rw = self.vtkWidget.GetRenderWindow()
        rw.AddRenderer(scene._ren)
        scene._qt_window = self  # back‑reference

        # Time‑series slider – only if needed
        max_tp = self._max_timepoints()
        if max_tp > 1:
            slider = scene._create_timeseries_slider()
            layout.addWidget(slider, stretch=0)

        # Finalise interactor *after* widgets are laid out
        self.vtkWidget.Initialize()
        self.vtkWidget.Enable()

    def _max_timepoints(self) -> int:
        mt = 0
        for obj in self._scene._timeseries_objects.values():
            mt = max(mt, obj.timeseries.shape[1])
        return mt

    # Ensure VTK resources are freed when the Qt window is closed
    def closeEvent(self, ev):
        rw = self.vtkWidget.GetRenderWindow()
        rw.Finalize()
        self.vtkWidget.Disable()
        ev.accept()
