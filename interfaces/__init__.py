"""
Interfaces package for lys library.

This package contains abstract base classes that define the core interfaces
for the lys library. These interfaces establish contracts that concrete
implementations must follow.
"""

from .plottable import Plottable

__all__ = ['Plottable'] 