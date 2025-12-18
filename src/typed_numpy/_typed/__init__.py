"""
Typed NumPy core
=======
"""
# src/typed_numpy/_typed/__init__.py

from typed_numpy._typed.ndarray import ShapedNDArray, TypedNDArray

__all__ = [
    "TypedNDArray",
    "ShapedNDArray",
]
