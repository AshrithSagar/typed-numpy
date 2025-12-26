"""
NDArray
=======
"""
# src/typed_numpy/_typed/ndarray.py

from types import GenericAlias
from typing import Any, Literal, TypeAlias, TypeVar, get_args, get_origin

import numpy as np
import numpy.typing as npt

## Typed NDArray


_AcceptedDim: TypeAlias = int | TypeVar | None
_AcceptedShape: TypeAlias = tuple[_AcceptedDim, ...]
_RuntimeDim: TypeAlias = int | None
_RuntimeShape: TypeAlias = tuple[_RuntimeDim, ...]


class DimensionError(Exception): ...


class ShapeError(Exception): ...


# `numpy` privates
_Shape: TypeAlias = tuple[Any, ...]  # Weakened type reduction
_AnyShape: TypeAlias = tuple[_AcceptedDim, ...]  # Stronger type promotion

_ShapeT_co = TypeVar("_ShapeT_co", bound=_Shape, default=_AnyShape, covariant=True)
_DTypeT_co = TypeVar("_DTypeT_co", bound=np.dtype, default=np.dtype, covariant=True)


def _resolve_dim(dim: _AcceptedDim | type[int]) -> _RuntimeDim:
    """Resolve a dimension specifier into something that can be runtime-validated."""
    if dim is None or dim is int or type(dim) is TypeVar:
        return None
    elif isinstance(dim, int):
        return dim
    elif get_origin(dim) is Literal:
        if (lit := get_args(dim)) and len(lit) == 1 and isinstance(lit[0], int):
            return lit[0]
    return None  # Fallback


def _resolve_shape(shape: _Shape) -> _RuntimeShape:
    """Resolve each dimension in a shape specifier."""
    return tuple(_resolve_dim(dim) for dim in shape)


def _normalise_shape(item: _AcceptedDim | _AcceptedShape) -> _AcceptedShape:
    """Ensure shape is a tuple."""
    return item if isinstance(item, tuple) else (item,)


def _validate_shape(expected: _RuntimeShape, actual: tuple[int, ...]) -> None:
    """Validate shapes at runtime."""
    # Rank enforcement
    if len(expected) != len(actual):
        raise DimensionError(
            f"Dimension mismatch: expected {len(expected)}, got {len(actual)}"
        )

    # Shape enforcement
    for exp, act in zip(expected, actual):
        if exp is not None and exp != act:
            raise ShapeError(f"Shape mismatch: expected {expected}, got {actual}")


def _validate_shape_against_active_contexts(
    shape_spec: _Shape, actual_shape: tuple[int, ...]
) -> None:
    """Validate shape against both class-level and method-level TypeVar contexts."""
    from typed_numpy._typed.context import (
        _active_class_context,
        _method_typevar_context,
    )

    typevars_in_spec: list[tuple[int, TypeVar]] = []
    for idx, dim in enumerate(shape_spec):
        if isinstance(dim, TypeVar):
            typevars_in_spec.append((idx, dim))
    if not typevars_in_spec:
        return

    method_context = _method_typevar_context.get()
    class_context = _active_class_context.get()
    for dim_idx, typevar in typevars_in_spec:
        if dim_idx >= len(actual_shape):
            continue
        actual_dim = actual_shape[dim_idx]
        expected_dim = (
            method_context.get(typevar)
            if typevar in method_context
            else class_context.get(typevar)
        )
        if expected_dim is not None and actual_dim != expected_dim:
            raise ShapeError(
                f"TypeVar {typevar} mismatch: expected {expected_dim}, got {actual_dim}"
            )


class TypedNDArray(np.ndarray[_ShapeT_co, _DTypeT_co]):
    """Generic `numpy.ndarray` subclass with static shape typing and runtime shape validation."""

    __bound_shape__: _RuntimeShape | None = None
    """Runtime shape metadata."""

    __static_params__: tuple[Any, Any] | None = None
    """Static parameters: (shape, dtype)."""

    @classmethod
    def __class_getitem__(
        cls,
        item:
        # Stronger type promotion
        GenericAlias | tuple[GenericAlias, GenericAlias],
        /,
    ) -> Any:  # Overrides base
        # [HACK] Misuses __class_getitem__
        # See https://docs.python.org/3/reference/datamodel.html#the-purpose-of-class-getitem

        _dtype: Any
        if isinstance(item, tuple):
            if len(item) != 2:
                raise TypeError(f"{cls.__name__}[...] expects (shape, dtype) or shape")
            _shape, _dtype = item
        elif isinstance(item, GenericAlias):
            _shape, _dtype = item, Any
        else:
            _shape, _dtype = Any, Any

        # Defer evaluation for generics
        if isinstance(_shape, GenericAlias):
            args = get_args(_shape)
            if any(type(a) is TypeVar for a in args):
                return _NDShape(cls, args)

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
        _shape_for_validation: _Shape | None = None
        if shape is not None:
            obj.__bound_shape__ = _resolve_shape(shape)
            if isinstance(shape, tuple):
                _shape_for_validation = shape
        elif _shape_static is not None:
            obj.__bound_shape__ = _resolve_shape(_shape_static)
        else:
            obj.__bound_shape__ = None

        # Runtime validation
        if obj.__bound_shape__ is not None:
            _validate_shape(expected=obj.__bound_shape__, actual=_arr.shape)
        if _shape_for_validation is not None:
            _validate_shape_against_active_contexts(_shape_for_validation, _arr.shape)

        # [NOTE] numpy.ndarray.view should suffice for the return type;
        # Explicit casting would prolly have a redunant call to TypedNDArray.__class_getitem__;
        # So we just use a type: ignore comment, for strict type checkers [mypy --strict];
        return obj  # type: ignore
        # return cast(TypedNDArray[_ShapeT_co, _DTypeT_co], obj)

    def __array_finalize__(self, obj: npt.NDArray[Any] | None, /) -> None:
        if obj is None:
            return

        # Propagate metadata
        # [FIXME] May have downstream side effects
        self.__bound_shape__ = getattr(obj, "__bound_shape__", None)
        self.__static_params__ = getattr(obj, "__static_params__", None)

    def __repr__(self) -> str:
        return str(np.asarray(self).__repr__())


class _NDShape:
    """
    Deferred TypedNDArray constructor with partially-bound shape.
    Behaves like a type-level curry.
    """

    __slots__ = ("base", "shape", "explicit")

    def __init__(
        self,
        cls: type[TypedNDArray],
        shape: _AcceptedDim | _AcceptedShape,
        explicit: bool = False,  # Track if explicitly re-specified
    ):
        self.base = cls
        self.shape = _normalise_shape(shape)
        self.explicit = explicit

    def __getitem__(self, item: _AcceptedDim | _AcceptedShape) -> "_NDShape":
        """Bind dimensions to unbound TypeVars by position, using defaults for missing ones."""
        _item = _normalise_shape(item)
        unbound_typevars = [
            (i, dim) for i, dim in enumerate(self.shape) if isinstance(dim, TypeVar)
        ]
        if len(_item) > len(unbound_typevars):
            # For a runtime error; Statically should already be caught;
            raise DimensionError(
                f"Too many type arguments: expected at most {len(unbound_typevars)}, got {len(_item)}"
            )

        shape = list(self.shape)
        for (pos, _), dim in zip(unbound_typevars[: len(_item)], _item):
            shape[pos] = dim
        for pos, typevar in unbound_typevars[len(_item) :]:
            default = getattr(typevar, "__default__", None)
            if default is not None:
                shape[pos] = default
        return _NDShape(cls=self.base, shape=tuple(shape), explicit=True)

    def __call__(
        self,
        arr: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
    ) -> TypedNDArray[_AnyShape, np.dtype[Any]]:
        # [NOTE] Should mimick TypedNDArray.__new__ signature
        # [TODO] Resolve any potential side-effects through the provided shape kwarg?
        _shape = self.shape if self.explicit else None
        return self.base(arr, dtype=dtype, shape=_shape)

    def __repr__(self) -> str:
        dims = ", ".join(str(dim) for dim in self.shape)
        return f"{self.base.__name__}[tuple({dims})]"
