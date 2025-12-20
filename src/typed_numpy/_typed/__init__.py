"""
Typed NumPy core
=======
"""
# src/typed_numpy/_typed/__init__.py

from typed_numpy._typed.generics import DimVar, DimVarBinder, ShapedNDArray
from typed_numpy._typed.ndarray import TypedNDArray

__all__ = [
    "TypedNDArray",
    "DimVar",
    "DimVarBinder",
    "ShapedNDArray",
]
