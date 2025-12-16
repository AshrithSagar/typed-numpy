"""
Shapes
=======
"""
# src/typed_numpy/shapes.py

from typing import Callable, Generic, Literal, TypeAlias, cast

import numpy.typing as npt

from typed_numpy.ndarray import TypedNDArray, _ShapeT_co


class Shape(Generic[_ShapeT_co]):
    """
    Descriptor that produces a constructor:
        (ArrayLike) -> TypedNDArray[_ShapeT_co]
    """

    def __init__(self, *dims: object) -> None:
        # dims are symbolic: "D", int, None, etc.
        self.dims = dims

    def __get__(
        self, obj, owner
    ) -> Callable[[npt.ArrayLike], TypedNDArray[_ShapeT_co]]:
        if obj is None:
            return self  # type: ignore[return-value]

        dim = getattr(obj, "__dim__", None)

        def constructor(arr: npt.ArrayLike) -> TypedNDArray[_ShapeT_co]:
            runtime_shape = tuple(dim if d == "D" else d for d in self.dims)
            runtime_shape = cast(_ShapeT_co, runtime_shape)
            return TypedNDArray(arr, shape=runtime_shape)

        return constructor


# Literal type aliases for small integers
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
