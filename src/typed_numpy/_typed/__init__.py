"""
Typed NumPy core
=======
"""
# src/typed_numpy/_typed/__init__.py

from typed_numpy._typed.generics import DimVar, GenericDim, ShapedNDArray
from typed_numpy._typed.ndarray import TypedNDArray

__all__ = [
    "TypedNDArray",
    "GenericDim",
    "DimVar",
    "ShapedNDArray",
]
