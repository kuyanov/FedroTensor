import numpy as np
import torch

from FedroTensor.tensor_rank import cp_rank, cp_rank_loss


def check_factorisation(A, factors, tol=1e-2):
    assert cp_rank_loss(torch.tensor(A), [torch.tensor(factor) for factor in factors]) < tol


def test_1d():
    A = np.array([1, 2], dtype=np.float64)
    r, factors = cp_rank(A)
    assert r == 1
    check_factorisation(A, factors)


def test_2d():
    A = np.array([[-1, -1], [-1, -1]], dtype=np.float64)
    r, factors = cp_rank(A)
    assert r == 1
    check_factorisation(A, factors)


def test_2d_symmetric():
    A = np.array([[1, 2], [2, 4]], dtype=np.float64)
    r, factors = cp_rank(A, symmetric=True)
    assert r == 1
    assert (factors[0] == factors[1]).all()
    check_factorisation(A, factors)


def test_2d_complex():
    A = np.array([[1, 1j], [1j, -1]], dtype=np.complex128)
    r, factors = cp_rank(A)
    assert r == 1
    check_factorisation(A, factors)


def test_3d_regularisation():
    u = np.array([1, 0], dtype=np.float64)
    v = np.array([0, 1], dtype=np.float64)
    A = (np.tensordot(v, np.outer(u, u), axes=0) +
         np.tensordot(u, np.outer(v, u), axes=0) +
         np.tensordot(u, np.outer(u, v), axes=0))
    r, factors = cp_rank(A)
    assert r == 3
    check_factorisation(A, factors)


def test_3d_complex_rational():
    A = np.array([[[1, 1], [1, -1]], [[1, -1], [-1, -1]]], dtype=np.complex128)
    r, factors = cp_rank(A, rational=True)
    assert r == 2
    check_factorisation(A, factors, tol=1e-9)
