import torch

from FedroTensor.tensor_rank import cp_rank, cp_rank_loss
from FedroTensor.matmul_tensor import build_matmul_tensor


def check_factorisation(A, factors, tol=1e-2):
    assert cp_rank_loss(torch.tensor(A), [torch.tensor(factor) for factor in factors]) < tol


def test_matmul_2x2x2():
    A = build_matmul_tensor(2, 2, 2)
    r, factors = cp_rank(A)
    assert r == 7
    check_factorisation(A, factors)


def test_matmul_2x2x2_rational():
    A = build_matmul_tensor(2, 2, 2)
    r, factors = cp_rank(A, rational=True)
    assert r == 7
    check_factorisation(A, factors, tol=1e-9)
