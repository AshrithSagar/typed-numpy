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

DimT = TypeVar("DimT", bound=int, default=int)


class GenericDim(Generic[DimT]):
    """Base class that binds Literal dimensions at class creation."""

    __dim__: DimT | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        for base in getattr(cls, "__orig_bases__", ()):
            origin = get_origin(base)
            args = get_args(base)

            if origin is GenericDim and args:
                (arg,) = args
            elif origin and issubclass(origin, GenericDim) and args:
                (arg,) = args
            else:
                continue

            if get_origin(arg) is Literal:
                (value,) = get_args(arg)
                value = cast(DimT, value)
                cls.__dim__ = value
                return


class DimVar(Generic[DimT]):
    def __set_name__(self, owner: GenericDim, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name})"


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

        _dim: _RuntimeDim | None = getattr(obj, "__dim__", None)

        def resolve_dim(dim: _AcceptedDim | DimVar) -> _RuntimeDim:
            if isinstance(dim, DimVar):
                return _dim
            elif isinstance(dim, TypeVar):
                # Prefer DimVar for runtime-validation
                return None
            return dim

        def constructor(arr: npt.ArrayLike) -> TypedNDArray[_ShapeT_co]:
            runtime_shape = tuple(resolve_dim(dim) for dim in self.__dims__)
            runtime_shape = cast(_ShapeT_co, runtime_shape)
            return TypedNDArray(arr, shape=runtime_shape)

        return constructor
