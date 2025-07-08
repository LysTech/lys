import numpy as np
import pyqtgraph as pg  # type: ignore
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QGridLayout, QSizePolicy, QScrollArea, QLabel  # type: ignore
from PyQt5.QtCore import Qt  # type: ignore
from typing import Optional
import sys
from tqdm import tqdm


class ChannelsPlot:
    """
    Interactive multi-channel plotting widget that allows expanding individual plots.
    
    Creates a scrollable grid of small plots where clicking on any plot expands it to fill
    the entire window. An X button allows returning to the original grid layout.
    Shows at most 100 plots at once by default, with scrolling for additional plots.
    """
    
    def __init__(self, title: Optional[str] = None, max_plots_visible: int = 100):
        """
        Initialize the ChannelsPlot widget.
        
        Args:
            title: Optional title for the window
            max_plots_visible: Maximum number of plots to show at once (default: 100)
        """
        self.title = title
        self.max_plots_visible = max_plots_visible
        self.app: Optional[QApplication] = None
        self.window: Optional[QMainWindow] = None
        self.plot_widgets: list = []
        self.expanded_plot: Optional[pg.PlotWidget] = None
        self.close_button: Optional[QPushButton] = None
        self.original_layout: Optional[QGridLayout] = None
        self.original_window_title: str = ""
        self.data: Optional[np.ndarray] = None
        self.grid_widget: Optional[QWidget] = None
        self.scroll_area: Optional[QScrollArea] = None
        self.expanded_plot_info: dict = {}
        
    def plot(self, data: np.ndarray, block: bool = True):
        """
        Create and display the interactive multi-channel plot.
        
        Args:
            data: Array of shape (M, T) where M is number of channels and T is timepoints
            block: If True, the call will block and start the Qt event loop.
                   Set to False if integrating into an existing Qt application.
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2-dimensional with shape (channels, timepoints)")
            
        self.data = data
        num_channels, num_timepoints = data.shape
        
        # Initialize Qt application if not already done
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
            
        # Create main window
        self.window = QMainWindow()
        assert self.window is not None  # type: ignore
        
        if self.title:
            self.original_window_title = self.title
        else:
            self.original_window_title = f"Channels Plot - {num_channels} channels, {num_timepoints} timepoints"
        
        self.window.setWindowTitle(self.original_window_title)
        self.window.resize(1200, 800)
        
        # Create central widget that will hold layouts
        central_widget = QWidget()
        self.window.setCentralWidget(central_widget)
        main_layout = QGridLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        central_widget.setLayout(main_layout)

        # Create scroll area for the grid
        self.scroll_area = QScrollArea()
        assert self.scroll_area is not None
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create widget to hold the grid
        self.grid_widget = QWidget()
        assert self.grid_widget is not None
        self.grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._create_grid_layout(self.grid_widget, data)
        
        # Set the grid widget as the scroll area's widget
        self.scroll_area.setWidget(self.grid_widget)
        main_layout.addWidget(self.scroll_area, 0, 0)

        # Show window
        self.window.show()
        
        if block:
            assert self.app is not None
            # This will start the event loop and block until the window is closed.
            self.app.exec_()
    
    def _create_grid_layout(self, parent_widget: QWidget, data: np.ndarray):
        """Create the original grid layout with all plots."""
        num_channels, num_timepoints = data.shape
        
        # Calculate grid dimensions (aim for roughly square grid)
        grid_size = int(np.ceil(np.sqrt(self.max_plots_visible)))
        
        # Create grid layout
        grid_layout = QGridLayout()
        parent_widget.setLayout(grid_layout)
        
        # Create time axis
        time_axis = np.arange(num_timepoints)
        
        # Create individual plot widgets for all channels
        self.plot_widgets = []
        for i in tqdm(range(num_channels)):
            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground(None)
            plot_widget.setTitle(f"Ch. {i+1}")
            plot_widget.getAxis('left').setTextPen(None)
            plot_widget.getAxis('bottom').setTextPen(None)
            
            # Disable scroll zoom on small plots
            plot_widget.setMouseEnabled(x=False, y=False)
            
            # Plot the data
            plot_widget.plot(time_axis, data[i, :], pen=pg.mkPen('b', width=1))
            
            # Make plot clickable
            plot_widget.mousePressEvent = lambda event, idx=i: self._expand_plot(idx)
            
            # Add to grid
            row = i // grid_size
            col = i % grid_size
            grid_layout.addWidget(plot_widget, row, col)
            
            self.plot_widgets.append(plot_widget)
        
        # Add a note about scrolling if there are many channels
        if num_channels > self.max_plots_visible:
            info_text = f"Showing {num_channels} channels in a scrollable grid. Use scroll bars to navigate."
            info_label = QLabel(info_text)
            info_label.setStyleSheet("QLabel { color: gray; font-size: 10px; padding: 5px; }")
            info_label.setAlignment(Qt.AlignCenter)
            
            # Add the info label at the bottom of the grid
            grid_layout.addWidget(info_label, grid_layout.rowCount(), 0, 1, grid_size)
        
        self.original_layout = grid_layout
    
    def _expand_plot(self, channel_idx: int):
        """Expand the selected plot to fill the entire window."""
        if self.expanded_plot is not None or self.window is None or self.data is None:
            return  # Already expanded or invalid state
            
        central_widget = self.window.centralWidget()
        assert central_widget is not None
        main_layout = central_widget.layout()
        assert main_layout is not None
        assert self.scroll_area is not None

        # Hide scroll area
        self.scroll_area.hide()

        # Create a new plot for the expanded view
        expanded_plot = pg.PlotWidget()
        expanded_plot.setBackground(None)
        
        # Enable mouse interaction (zoom, pan) for expanded plot
        expanded_plot.setMouseEnabled(x=True, y=True)
        
        time_axis = np.arange(self.data.shape[1])
        expanded_plot.plot(time_axis, self.data[channel_idx, :], pen=pg.mkPen('b', width=2))
        
        self.expanded_plot = expanded_plot
        main_layout.addWidget(self.expanded_plot, 0, 0)

        # Update window title
        assert self.window is not None
        self.window.setWindowTitle(f"Channel {channel_idx + 1} - Expanded View")
        
        # Create close button
        if self.close_button is None:
            close_button = QPushButton("X")
            close_button.setFixedSize(30, 30)
            close_button.clicked.connect(self._collapse_plot)
            close_button.setStyleSheet("""
                QPushButton {
                    background-color: #ff4444;
                    color: white;
                    border: none;
                    border-radius: 15px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #ff6666;
                }
            """)
            self.close_button = close_button
        
        assert self.close_button is not None
        # Add close button to main layout, not plot
        main_layout.addWidget(self.close_button, 0, 0, Qt.AlignTop | Qt.AlignRight)
        self.close_button.show()
        self.close_button.raise_()

    def _collapse_plot(self):
        """Return to the original grid layout."""
        if self.expanded_plot is None or self.window is None or self.data is None or self.close_button is None:
            return

        central_widget = self.window.centralWidget()
        assert central_widget is not None
        main_layout = central_widget.layout()
        assert main_layout is not None
        assert self.scroll_area is not None

        # Restore window title
        assert self.window is not None
        self.window.setWindowTitle(self.original_window_title)

        # Remove and delete the expanded plot
        main_layout.removeWidget(self.expanded_plot)
        self.expanded_plot.deleteLater()
        self.expanded_plot = None

        # Remove close button from layout
        main_layout.removeWidget(self.close_button)
        self.close_button.hide()
        
        self.scroll_area.show()

        # Reset expanded state
        self.expanded_plot = None


if __name__ == "__main__":
    # Generate random test data: 100 channels, 2000 timepoints
    np.random.seed(42)  # For reproducible results
    num_channels = 100
    num_timepoints = 2000
    
    # Create some realistic-looking data with different patterns
    data = np.zeros((num_channels, num_timepoints))
    
    for i in range(num_channels):
        # Add some baseline noise
        data[i, :] = np.random.normal(0, 0.1, num_timepoints)
        
        # Add some periodic components
        t = np.arange(num_timepoints)
        data[i, :] += 0.5 * np.sin(2 * np.pi * t / 500 + i * 0.1)
        data[i, :] += 0.3 * np.sin(2 * np.pi * t / 200 + i * 0.05)
        
        # Add some random spikes
        spike_indices = np.random.choice(num_timepoints, size=5, replace=False)
        data[i, spike_indices] += np.random.normal(0, 0.5, len(spike_indices))
    
    # Create and display the plot
    ChannelsPlot(title="100 Channels Example").plot(data)
