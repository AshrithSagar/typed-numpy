"""
Context binding
=======
"""
# src/typed_numpy/_typed/context.py

import inspect
from contextvars import ContextVar
from functools import wraps
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

from typed_numpy._typed.ndarray import ShapeError, _NDShape

# Separate contexts for class-level vs method-level TypeVars
_class_typevar_context = ContextVar[dict[int, dict[TypeVar, int]]](
    "_class_typevar_context", default={}
)


def _extract_shape_typevars(annotation: Any) -> list[tuple[int, TypeVar]]:
    """Extract TypeVars from a TypedNDArray annotation with their dimension index."""

    # _NDShape
    if isinstance(annotation, _NDShape):
        typevars = list[tuple[int, TypeVar]]()
        for idx, dim in enumerate(annotation.shape):
            if isinstance(dim, TypeVar):
                typevars.append((idx, dim))
        return typevars

    # GenericAlias
    origin = get_origin(annotation)
    if origin is None:
        return []

    args = get_args(annotation)
    if not args:
        return []

    shape_spec = args[0]
    if get_origin(shape_spec) is tuple:
        shape_dims = get_args(shape_spec)
        typevars = list[tuple[int, TypeVar]]()
        for idx, dim in enumerate(shape_dims):
            if isinstance(dim, TypeVar):
                typevars.append((idx, dim))
        return typevars

    return []


def _is_class_level_typevar(typevar: TypeVar, owner_cls: type) -> bool:
    """Check if a TypeVar is bound at class level vs method level."""
    cls_params = getattr(owner_cls, "__parameters__", ())
    return typevar in set(cls_params)


def _get_instance_class_context(instance: Any) -> dict[TypeVar, int]:
    """Get or create the class-level TypeVar binding context for an instance."""
    ctx = _class_typevar_context.get()
    instance_id = id(instance)
    if instance_id not in ctx:
        ctx = ctx.copy()
        ctx[instance_id] = {}
        _class_typevar_context.set(ctx)
    return ctx[instance_id]


def enforce_shapes(func):
    """
    Automatically validate TypeVar shape bindings.
    - Class-level TypeVars (from Generic[T]) are bound per-instance, persist across calls
    - Method-level TypeVars are validated per-call only (within same call)
    - Both parameter and return types are validated
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        hints = get_type_hints(func, include_extras=True)
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        owner_cls = self.__class__
        class_context = _get_instance_class_context(self)
        method_context = dict[TypeVar, int]()

        # Validate inputs
        for param_name, param_value in bound_args.arguments.items():
            if param_name == "self":
                continue
            if param_name not in hints:
                continue

            annotation = hints[param_name]
            typevars = _extract_shape_typevars(annotation)
            if not typevars or not hasattr(param_value, "shape"):
                continue

            actual_shape = param_value.shape
            for dim_idx, typevar in typevars:
                if dim_idx >= len(actual_shape):
                    continue

                actual_dim = actual_shape[dim_idx]
                is_class_level = _is_class_level_typevar(typevar, owner_cls)
                if is_class_level:
                    if typevar in class_context:
                        expected_dim = class_context[typevar]
                        if actual_dim != expected_dim:
                            raise ShapeError(
                                f"In {func.__name__}(...): parameter '{param_name}' "
                                f"dimension {dim_idx} ({typevar.__name__}): "
                                f"expected {expected_dim} (class-level binding), got {actual_dim}"
                            )
                    else:
                        class_context[typevar] = actual_dim
                else:
                    if typevar in method_context:
                        expected_dim = method_context[typevar]
                        if actual_dim != expected_dim:
                            raise ShapeError(
                                f"In {func.__name__}(): parameter '{param_name}' "
                                f"dimension {dim_idx} ({typevar.__name__}): "
                                f"expected {expected_dim} (method-level binding), got {actual_dim}"
                            )
                    else:
                        method_context[typevar] = actual_dim

        result = func(self, *args, **kwargs)

        # Validate return
        if "return" in hints and result is not None:
            return_annotation = hints["return"]
            typevars = _extract_shape_typevars(return_annotation)

            if typevars and hasattr(result, "shape"):
                actual_shape = result.shape
                for dim_idx, typevar in typevars:
                    if dim_idx >= len(actual_shape):
                        continue

                    actual_dim = actual_shape[dim_idx]
                    is_class_level = _is_class_level_typevar(typevar, owner_cls)
                    context = class_context if is_class_level else method_context
                    if typevar in context:
                        expected_dim = context[typevar]
                        if actual_dim != expected_dim:
                            level = "class" if is_class_level else "method"
                            raise ShapeError(
                                f"In {func.__name__}(): return value "
                                f"dimension {dim_idx} ({typevar.__name__}): "
                                f"expected {expected_dim} ({level}-level binding), got {actual_dim}"
                            )

        return result

    return wrapper
