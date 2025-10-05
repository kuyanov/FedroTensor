import numpy as np
import torch

from FedroTensor.tensor_rank import cp_rank, cp_rank_loss


def check_factorisation(T, factors, tol=1e-2):
    assert cp_rank_loss(torch.tensor(T), [torch.tensor(factor) for factor in factors]) < tol


def test_1d():
    T = np.array([1, 2], dtype=np.float64)
    r, factors = cp_rank(T)
    assert r == 1
    check_factorisation(T, factors)


def test_2d():
    T = np.array([[-1, -1], [-1, -1]], dtype=np.float64)
    r, factors = cp_rank(T)
    assert r == 1
    check_factorisation(T, factors)


def test_2d_symmetric():
    T = np.array([[1, 2], [2, 4]], dtype=np.float64)
    r, factors = cp_rank(T, symmetric=True)
    assert r == 1
    assert (factors[0] == factors[1]).all()
    check_factorisation(T, factors)


def test_2d_complex():
    T = np.array([[1, 1j], [1j, -1]], dtype=np.complex128)
    r, factors = cp_rank(T)
    assert r == 1
    check_factorisation(T, factors)


def test_3d_regularisation():
    u = np.array([1, 0], dtype=np.float64)
    v = np.array([0, 1], dtype=np.float64)
    T = (np.tensordot(v, np.outer(u, u), axes=0) +
         np.tensordot(u, np.outer(v, u), axes=0) +
         np.tensordot(u, np.outer(u, v), axes=0))
    r, factors = cp_rank(T)
    assert r == 3
    check_factorisation(T, factors)


def test_3d_complex_rational():
    T = np.array([[[1, 1], [1, -1]], [[1, -1], [-1, -1]]], dtype=np.complex128)
    r, factors = cp_rank(T, rational=True)
    assert r == 2
    check_factorisation(T, factors, tol=1e-9)
