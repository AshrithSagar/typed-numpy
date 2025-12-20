"""
Generics for TypedNDArray
=======
"""
# src/typed_numpy/_typed/generics.py

from typing import Any, Callable, Generic, Literal, TypeVar, cast, get_args, get_origin

import numpy.typing as npt

from typed_numpy._typed.ndarray import (
    TypedNDArray,
    _AcceptedDim,
    _RuntimeDim,
    _ShapeT_co,
)

## Dimension Generics


class DimVar:
    def __init__(self) -> None:
        self._name: str | None = None

    def __set_name__(self, owner: object, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"


class DimVarBinder:
    """Base class that binds Literal dimensions at class creation."""

    __dim_bindings__: dict[DimVar, int]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.__dim_bindings__ = {}

        dimvars: list[DimVar] = []
        for base in cls.__mro__:
            for v in base.__dict__.values():
                if isinstance(v, DimVar):
                    dimvars.append(v)

        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            args = get_args(base)
            if not origin or not issubclass(origin, DimVarBinder) or not args:
                continue

            for dimvar, arg in zip(dimvars, args):
                if get_origin(arg) is Literal:
                    (value,) = get_args(arg)
                    cls.__dim_bindings__[dimvar] = value


class ShapedNDArray(Generic[_ShapeT_co]):
    """
    Descriptor that produces a constructor:
        (ArrayLike) -> TypedNDArray[_ShapeT_co]
    """

    __dims__: tuple[_AcceptedDim | DimVar, ...] = ()

    def __init__(self, *dims: _AcceptedDim | DimVar) -> None:
        self.__dims__ = dims

    def __get__(
        self, obj: object | None, owner: Any
    ) -> Callable[[npt.ArrayLike], TypedNDArray[_ShapeT_co]]:
        if obj is None:
            return self  # type: ignore[return-value]

        bindings = getattr(type(obj), "__dim_bindings__", {})

        def resolve_dim(dim: _AcceptedDim | DimVar) -> _RuntimeDim:
            if isinstance(dim, DimVar):
                return bindings.get(dim, None)
            elif isinstance(dim, TypeVar):
                # Prefer DimVar for runtime-validation
                return None
            return dim

        def constructor(arr: npt.ArrayLike) -> TypedNDArray[_ShapeT_co]:
            runtime_shape = tuple(resolve_dim(dim) for dim in self.__dims__)
            runtime_shape = cast(_ShapeT_co, runtime_shape)
            return TypedNDArray(arr, shape=runtime_shape)

        return constructor
