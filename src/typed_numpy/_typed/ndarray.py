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


def _normalise_dim(dim: _AcceptedDim | type[int]) -> _RuntimeDim:
    """Normalise a dimension specifier into something that can be runtime-validated."""

    if dim is None:
        return None
    if dim is int:
        return None
    if isinstance(dim, int):
        return dim

    origin = get_origin(dim)
    if origin is Literal:
        lit = get_args(dim)
        if len(lit) == 1 and isinstance(lit[0], int):
            return lit[0]

    if type(dim) is TypeVar:
        # Prefer TypeAlias when using Generics
        return None

    return None  # Fallback


def _normalise_shape(shape: _Shape) -> _RuntimeShape:
    """Normalise each dimension in a shape specifier."""
    return tuple(_normalise_dim(dim) for dim in shape)


def _resolve_type_params(cls: type, root: type) -> tuple[Any, ...] | None:
    """Recursively resolve a class's type parameters back to the root class."""

    if cls is root:
        return None
    orig_bases = getattr(cls, "__orig_bases__", ())
    for base in orig_bases:
        origin = get_origin(base)
        if origin is None:
            continue
        try:
            if not issubclass(origin, root):
                continue
        except TypeError:
            continue

        args = get_args(base)
        if origin is root:
            return args

        intermediate_resolution = _resolve_type_params(origin, root)
        if intermediate_resolution is None:
            continue

        intermediate_params = getattr(origin, "__parameters__", ())
        substitution = {}
        for param, arg in zip(intermediate_params, args):
            substitution[param] = arg

        final_resolution = []
        for item in intermediate_resolution:
            if isinstance(item, TypeVar) and item in substitution:
                final_resolution.append(substitution[item])
            else:
                final_resolution.append(item)
        return tuple(final_resolution)

    return None


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

        _params: tuple[Any, Any] | None = None
        resolved_params = _resolve_type_params(cls, TypedNDArray)
        if resolved_params is not None:
            _params = resolved_params
        elif cls.__static_params__ is not None:
            _params = cls.__static_params__

        _shape_static: _Shape | None = None
        if _params is not None:
            _shape, _dtype = _params

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
            obj.__bound_shape__ = _normalise_shape(shape)
        elif _shape_static is not None:
            obj.__bound_shape__ = _normalise_shape(_shape_static)
        else:
            obj.__bound_shape__ = None

        # Runtime validation
        if obj.__bound_shape__ is not None:
            expected = obj.__bound_shape__
            actual = _arr.shape

            # Rank enforcement
            if len(expected) != len(actual):
                raise DimensionError(
                    f"Dimension mismatch: expected {len(expected)}, got {len(actual)}"
                )

            # Shape enforcement
            for exp, act in zip(expected, actual):
                if exp is None:  # No dimension check
                    continue
                if exp != act:
                    raise ShapeError(
                        f"Shape mismatch: expected {expected}, got {actual}"
                    )

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

    def __init__(self, cls: type[TypedNDArray], shape: _AcceptedDim | _AcceptedShape):
        self.base = cls
        self.shape = self._normalise_shape(shape)

    def _normalise_shape(self, item: _AcceptedDim | _AcceptedShape) -> _AcceptedShape:
        if not isinstance(item, tuple):
            _item = tuple((item,))
        else:
            _item = item
        return _item

    def __getitem__(self, item: _AcceptedDim | _AcceptedShape) -> "_NDShape":
        """Bind dimensions to the leftmost unbound TypeVars."""
        _shape: _AcceptedShape
        resolved = _resolve_type_params(self.base, TypedNDArray)
        if resolved is not None:
            resolved_shape_spec = resolved[0]  # Shape part
            if isinstance(resolved_shape_spec, GenericAlias):
                _shape = get_args(resolved_shape_spec)
            elif isinstance(resolved_shape_spec, tuple):
                _shape = resolved_shape_spec
            else:
                _shape = self.shape
        else:
            _shape = self.shape

        _item = self._normalise_shape(item)
        unbound = [i for i, dim in enumerate(_shape) if isinstance(dim, TypeVar)]
        if len(_item) > len(unbound):
            # For a runtime error; Statically should already be caught;
            raise DimensionError("Too many dimensions")

        shape = list(_shape)
        for i, dim in enumerate(_item):
            pos = unbound[i]
            shape[pos] = dim
        return _NDShape(cls=self.base, shape=tuple(shape))

    def __call__(
        self,
        arr: npt.ArrayLike,
        *,
        dtype: npt.DTypeLike | None = None,
        shape: _ShapeT_co | None = None,
    ) -> TypedNDArray[_AnyShape, np.dtype[Any]]:
        # [NOTE] Should mimick TypedNDArray.__new__ signature
        # [TODO] Resolve any potential side-effects through the provided shape kwarg?
        return self.base(arr, dtype=dtype, shape=self.shape)

    def __repr__(self) -> str:
        dims = ", ".join(str(dim) for dim in self.shape)
        return f"{self.base.__name__}[tuple({dims})]"
