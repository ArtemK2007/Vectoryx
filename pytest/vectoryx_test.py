from Vectoryx import (
    sum_vectors,
    vector_dif,
    dot_product,
    multiply_vector_to_C,
    vector_length,
    normalize_vector,
    cross_product,
    is_colinear,
    is_ortogonal,
    cosinus_similarity,
    v_angle
)

def test_sum_vectors():
    assert sum_vectors([1, 2], [3, 4]) == [4, 6]

def test_vector_dif():
    assert vector_dif([5, 6], [1, 2]) == [4, 4]

def test_dot_product():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32

def test_multiply_vector_to_C():
    assert multiply_vector_to_C([1, 2], 3) == [3, 6]

def test_vector_length():
    assert round(vector_length([3, 4]), 5) == 5.0

def test_normalize_vector():
    norm = normalize_vector([3, 4])
    assert round(norm[0], 5) == 0.6
    assert round(norm[1], 5) == 0.8

def test_cross_product():
    assert cross_product([1, 0, 0], [0, 1, 0]) == [0, 0, 1]

def test_is_colinear():
    assert is_colinear([2, 4, 6], [1, 2, 3]) is True
    assert is_colinear([1, 0, 0], [0, 1, 0]) is False

def test_is_ortogonal():
    assert is_ortogonal([1, 0], [0, 1]) is True
    assert is_ortogonal([1, 2], [2, 4]) is False

def test_cosinus_similarity():
    cos = cosinus_similarity([1, 2, 3], [2, 3, 4])
    assert round(cos, 5) == 0.99258

def test_v_angle():
    angle = v_angle([1, 0], [0, 1], in_degrees=True)
    assert round(angle) == 90
