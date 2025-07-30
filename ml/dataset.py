from dataclasses import dataclass
import numpy as np


@dataclass
class MLDataset:
    """Simple container for ML data - just X and y arrays."""
    X: np.ndarray
    y: np.ndarray
    metadata: dict = None
    
    def __post_init__(self):
        """Validate that X and y have compatible shapes."""
        if len(self.X) != len(self.y):
            raise ValueError(f"X and y must have same length. Got X: {len(self.X)}, y: {len(self.y)}")
        
        # Initialize empty metadata if None
        if self.metadata is None:
            self.metadata = {}

