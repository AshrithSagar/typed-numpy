"""
Shapes
=======
"""
# src/typed_numpy/_typed/shapes.py

from typing import Literal, TypeAlias

# Literal type aliases for small integers
ZERO: TypeAlias = Literal[0]
"""Literal type for the integer `0`."""
ONE: TypeAlias = Literal[1]
"""Literal type for the integer `1`."""
TWO: TypeAlias = Literal[2]
"""Literal type for the integer `2`."""
THREE: TypeAlias = Literal[3]
"""Literal type for the integer `3`."""
FOUR: TypeAlias = Literal[4]
"""Literal type for the integer `4`."""

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
