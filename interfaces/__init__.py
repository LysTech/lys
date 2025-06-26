"""
Interfaces package for lys library.

This package contains abstract base classes that define the core interfaces
for the lys library. These interfaces establish contracts that concrete
implementations must follow.
"""

from lys.interfaces.plottable import Plottable
from lys.interfaces.processing_step import ProcessingStep
from lys.interfaces.session_adapters import ISessionAdapter

__all__ = ['Plottable', 'ProcessingStep', 'ISessionAdapter'] 