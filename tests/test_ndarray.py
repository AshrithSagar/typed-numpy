"""
Tests for TypedNDArray core functionality
"""
# tests/test_ndarray.py

from typing import Literal, TypeVar

import numpy as np
import pytest

from typed_numpy._typed.ndarray import (
    DimensionError,
    RankError,
    ShapeError,
    TypedNDArray,
    _resolve_dim,
    _resolve_shape,
    _validate_shape,
)
from typed_numpy._typed.shapes import FOUR, THREE, TWO


class TestShapeResolution:
    """Tests for shape resolution functions."""

    def test_resolve_dim_none(self):
        assert _resolve_dim(None) is None

    def test_resolve_dim_int_type(self):
        assert _resolve_dim(int) is None

    def test_resolve_dim_concrete_int(self):
        assert _resolve_dim(5) == 5

    def test_resolve_dim_typevar(self):
        T = TypeVar("T")
        assert _resolve_dim(T) is None

    def test_resolve_dim_literal(self):
        assert _resolve_dim(Literal[3]) == 3  # type: ignore

    def test_resolve_shape_mixed(self):
        T = TypeVar("T")
        shape = (2, Literal[5], int, T, None)
        resolved = _resolve_shape(shape)
        assert resolved == (2, 5, None, None, None)


class TestShapeValidation:
    """Tests for shape validation."""

    def test_validate_shape_success(self):
        expected = (3, 4, 5)
        actual = (3, 4, 5)
        _validate_shape(expected, actual)  # Should not raise

    def test_validate_shape_with_none(self):
        expected = (3, None, 5)
        actual = (3, 100, 5)
        _validate_shape(expected, actual)  # Should not raise

    def test_validate_shape_rank_mismatch(self):
        expected = (3, 4)
        actual = (3, 4, 5)
        with pytest.raises(RankError, match="Rank mismatch"):
            _validate_shape(expected, actual)

    def test_validate_shape_dimension_mismatch(self):
        expected = (3, 4, 5)
        actual = (3, 4, 6)
        with pytest.raises(ShapeError, match="Shape mismatch"):
            _validate_shape(expected, actual)


class TestTypedNDArrayCreation:
    """Tests for TypedNDArray creation and initialisation."""

    def test_create_from_list(self):
        arr = TypedNDArray([1, 2, 3])
        assert isinstance(arr, TypedNDArray)
        assert arr.shape == (3,)

    def test_create_with_dtype(self):
        arr = TypedNDArray([1, 2, 3], dtype=np.float32)
        assert arr.dtype == np.float32

    def test_create_with_shape_validation(self):
        arr = TypedNDArray([[1, 2], [3, 4]], shape=(2, 2))
        assert arr.shape == (2, 2)

    def test_create_with_shape_validation_failure(self):
        with pytest.raises(ShapeError):
            TypedNDArray([[1, 2], [3, 4]], shape=(3, 2))


class TestTypedNDArrayClassGetitem:
    """Tests for TypedNDArray type specification syntax."""

    def test_simple_shape(self):
        Array3 = TypedNDArray[tuple[THREE]]
        arr = Array3([1, 2, 3])
        assert arr.shape == (3,)

    def test_shape_with_dtype(self):
        Array2x2 = TypedNDArray[tuple[TWO, TWO], np.dtype[np.float64]]
        arr = Array2x2([[1.0, 2.0], [3.0, 4.0]])
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64

    def test_shape_with_int(self):
        Array5 = TypedNDArray[tuple[5]]
        arr = Array5([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_multidimensional_shape(self):
        Array3x4 = TypedNDArray[tuple[THREE, FOUR]]
        arr = Array3x4(np.ones((3, 4)))
        assert arr.shape == (3, 4)

    def test_invalid_shape_raises(self):
        Array3 = TypedNDArray[tuple[THREE]]
        with pytest.raises(ShapeError):
            Array3([1, 2, 3, 4])


class TestTypedNDArrayWithTypeVars:
    """Tests for TypedNDArray with TypeVar dimensions."""

    def test_typevar_in_shape(self):
        N = TypeVar("N")
        ArrayN = TypedNDArray[tuple[N]]  # type: ignore
        # TypeVars don't validate at creation without context
        arr = ArrayN([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_multiple_typevars(self):
        M = TypeVar("M")
        N = TypeVar("N")
        ArrayMxN = TypedNDArray[tuple[M, N]]  # type: ignore
        arr = ArrayMxN([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_mixed_concrete_and_typevar(self):
        N = TypeVar("N")
        Array3xN = TypedNDArray[tuple[THREE, N]]  # type: ignore
        arr = Array3xN([[1, 2], [3, 4], [5, 6]])
        assert arr.shape == (3, 2)


class TestNDShapeDeferredBinding:
    """Tests for _NDShape deferred binding mechanism."""

    def test_bind_single_typevar(self):
        N = TypeVar("N")
        ArrayN = TypedNDArray[tuple[N]]  # type: ignore
        Array5 = ArrayN[5]  # type: ignore
        arr = Array5([1, 2, 3, 4, 5])
        assert arr.shape == (5,)

    def test_bind_single_typevar_wrong_size(self):
        N = TypeVar("N")
        ArrayN = TypedNDArray[tuple[N]]  # type: ignore
        Array5 = ArrayN[5]  # type: ignore
        with pytest.raises(ShapeError):
            Array5([1, 2, 3])

    def test_bind_multiple_typevars(self):
        M = TypeVar("M")
        N = TypeVar("N")
        ArrayMxN = TypedNDArray[tuple[M, N]]  # type: ignore
        Array2x3 = ArrayMxN[2, 3]  # type: ignore
        arr = Array2x3([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_bind_partial_typevars(self):
        M = TypeVar("M")
        N = TypeVar("N")
        ArrayMxN = TypedNDArray[tuple[M, N]]  # type: ignore
        Array2xN = ArrayMxN[TWO, N]  # type: ignore
        Array2x3 = Array2xN[THREE]  # type: ignore
        arr = Array2x3([[1, 2, 3], [4, 5, 6]])
        assert arr.shape == (2, 3)

    def test_bind_too_many_dimensions(self):
        N = TypeVar("N")
        ArrayN = TypedNDArray[tuple[N]]  # type: ignore
        with pytest.raises(DimensionError, match="Too many type arguments"):
            ArrayN[2, 3]  # type: ignore


class TestRepr:
    """Tests for string representation."""

    def test_repr_1d(self):
        arr = TypedNDArray([1, 2, 3])
        _arr = np.array([1, 2, 3])
        assert repr(arr) == repr(_arr)

    def test_repr_2d(self):
        arr = TypedNDArray([[1, 2], [3, 4]])
        _arr = np.array([[1, 2], [3, 4]])
        assert repr(arr) == repr(_arr)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_array(self):
        arr = TypedNDArray([])
        assert arr.shape == (0,)

    def test_scalar_array(self):
        arr = TypedNDArray(5)
        assert arr.shape == ()

    def test_complex_nested_structure(self):
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        arr = TypedNDArray(data)
        assert arr.shape == (2, 2, 2)

    def test_invalid_shape_spec_type(self):
        with pytest.raises(TypeError):
            # Should be tuple
            TypedNDArray[1, 2, 3]  # type: ignore
