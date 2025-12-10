"""
Helpers for TypedNDArray
=======
src/typed_numpy/helpers.py
"""

from typing import Literal, TypeAlias

import numpy as np

from .ndarray import TypedNDArray

## Helpers

def_dtype: TypeAlias = np.double
"""The default `dtype` used throughout, mostly."""

# Shape type aliases
Shape1D: TypeAlias = tuple[int]
"""A tuple representing a 1D shape, i.e., `(N,)`."""
Shape2D: TypeAlias = tuple[int, int]
"""A tuple representing a 2D shape, i.e., `(M, N)`."""
Shape3D: TypeAlias = tuple[int, int, int]
"""A tuple representing a 3D shape, i.e., shape `(L, M, N)`."""
Shape4D: TypeAlias = tuple[int, int, int, int]
"""A tuple representing a 4D shape, i.e., shape `(K, L, M, N)`."""
ShapeND: TypeAlias = tuple[int, ...]
"""A tuple representing a ND shape, i.e., shape `(N, ...)`."""

# Array type aliases
Array1D: TypeAlias = TypedNDArray[Shape1D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
Array2D: TypeAlias = TypedNDArray[Shape2D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(M, N)` with the default `dtype`."""
Array3D: TypeAlias = TypedNDArray[Shape3D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(L, M, N)` with the default `dtype`."""
Array4D: TypeAlias = TypedNDArray[Shape4D, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(K, L, M, N)` with the default `dtype`."""
ArrayND: TypeAlias = TypedNDArray[ShapeND, np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, ...)` with the default `dtype`."""

TWO: TypeAlias = Literal[2]
"""Literal type for the integer `2`."""

Array2: TypeAlias = TypedNDArray[tuple[TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2,)` with the default `dtype`."""
Array2x2: TypeAlias = TypedNDArray[tuple[TWO, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(2, 2)` with the default `dtype`."""
ArrayN: TypeAlias = TypedNDArray[tuple[int], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N,)` with the default `dtype`."""
ArrayNx2: TypeAlias = TypedNDArray[tuple[int, TWO], np.dtype[def_dtype]]
"""A `numpy.ndarray` of shape `(N, 2)` with the default `dtype`."""
