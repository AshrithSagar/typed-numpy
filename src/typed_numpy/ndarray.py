"""
Typed NumPy NDArray
=======
"""
# src/typed_numpy/ndarray.py

from types import GenericAlias
from typing import Any, Literal, TypeAlias, TypeVar, cast, get_args, get_origin

import numpy as np
import numpy.typing as npt

## Typed NDArray

_AcceptedDim: TypeAlias = int | TypeVar | None
_RuntimeDim: TypeAlias = int | None
_RuntimeShape: TypeAlias = tuple[_RuntimeDim, ...]


class DimensionError(Exception): ...


class ShapeError(Exception): ...


# `numpy` privates
_Shape: TypeAlias = tuple[Any, ...]  # Weakened type reduction
_AnyShape: TypeAlias = tuple[_AcceptedDim, ...]  # Stronger type promotion

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)


def _normalise_dim(dim: _AcceptedDim) -> _RuntimeDim:
    """Normalise a dimension specifier into something that can be runtime-validated."""

    if dim is None:
        return None
    if isinstance(dim, int):
        return dim
    if isinstance(dim, TypeVar):
        return None

    origin = get_origin(dim)
    if origin is Literal:
        lit = get_args(dim)
        if len(lit) == 1 and isinstance(lit[0], int):
            return lit[0]

    return None  # Fallback


def _normalise_shape(shape: _Shape) -> _RuntimeShape:
    """Normalise each dimension in a shape specifier."""
    return tuple(_normalise_dim(dim) for dim in shape)


class TypedNDArray(np.ndarray[_ShapeT_co, _DTypeT_co]):
    """Generic `numpy.ndarray` subclass with static shape typing and runtime shape validation."""

    __shape__: _RuntimeShape | None = None
    """Runtime shape metadata."""

    __static_params__: tuple[Any, Any] | None = None
    """Static parameters: (shape, dtype)."""

    @classmethod
    def __class_getitem__(cls, item: Any, /) -> Any:  # Overrides base
        if isinstance(item, tuple):
            _shape, _dtype = item
        elif isinstance(item, GenericAlias):
            _shape, _dtype = item, Any
        else:
            _shape, _dtype = Any, Any
        return type(
            f"{cls.__name__}[{item}]", (cls,), {"__static_params__": (_shape, _dtype)}
        )

    def __new__(
        cls,
        arr: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
    ) -> "TypedNDArray[_ShapeT_co, _DTypeT_co]":
        _arr: np.ndarray[tuple[int, ...]]
        _arr = np.asarray(arr, dtype=dtype)

        _shape_static: _Shape | None = None
        if cls.__static_params__ is not None:
            _shape, _dtype = cls.__static_params__

            # Infer dtype
            if _dtype is not Any:
                dtype_args = get_args(_dtype)
                if len(dtype_args) == 1 and issubclass(dtype_args[0], np.generic):
                    _arr = _arr.astype(dtype_args[0])  # Cast

            # Infer shape
            if _shape is not Any:
                if isinstance(_shape, tuple):
                    _shape_static = _shape
                elif isinstance(_shape, GenericAlias):
                    _shape_static = get_args(_shape)

        obj = _arr.view(cls)

        # Set metadata
        if shape is not None:
            obj.__shape__ = _normalise_shape(shape)
        elif _shape_static is not None:
            obj.__shape__ = _normalise_shape(_shape_static)
        else:
            obj.__shape__ = None

        # Runtime validation
        if obj.__shape__ is not None:
            expected = obj.__shape__
            actual = _arr.shape

            if len(expected) != len(actual):
                raise DimensionError(
                    f"Dimension mismatch: expected {len(expected)}, got {len(actual)}"
                )

            for exp, act in zip(expected, actual):
                if exp is not None and exp != act:
                    raise ShapeError(
                        f"Shape mismatch: expected {expected}, got {actual}"
                    )

        # numpy.ndarray.view should suffice, but we cast explicitly for strict type checkers
        return cast(TypedNDArray[_ShapeT_co, _DTypeT_co], obj)

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        if obj is None:
            return

        # Propagate metadata
        self.__shape__ = getattr(obj, "__shape__", None)
        self.__static_params__ = getattr(obj, "__static_params__", None)

    def __repr__(self) -> str:
        return str(np.asarray(self).__repr__())
