"""
Dimensions
=======
"""
# src/typed_numpy/dimensions.py

from typing import Any, Generic, Literal, TypeVar, cast, get_args, get_origin

DimT = TypeVar("DimT", bound=int, default=int)


class GenericDim(Generic[DimT]):
    """Base class that binds Literal dimensions at class creation."""

    __dim__: DimT | None = None

    def __init_subclass__(cls, **kwargs: Any):
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
