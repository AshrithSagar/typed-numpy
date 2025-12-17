"""
Typed NumPy
=======
"""
# src/typed_numpy/__init__.py

import numpy

from .ndarray import ShapedNDArray, TypedNDArray

__all__ = [
    "numpy",
    "TypedNDArray",
    "ShapedNDArray",
]
