import torch

from FedroTensor.tensor_rank import cp_rank, cp_rank_loss
from FedroTensor.matmul_tensor import build_matmul_tensor


def check_factorisation(T, factors, tol=1e-2):
    assert cp_rank_loss(torch.tensor(T), [torch.tensor(factor) for factor in factors]) < tol


def test_matmul_2x2x2():
    T = build_matmul_tensor(2, 2, 2)
    r, factors = cp_rank(T)
    assert r == 7
    check_factorisation(T, factors)


def test_matmul_2x2x2_rational():
    T = build_matmul_tensor(2, 2, 2)
    r, factors = cp_rank(T, rational=True)
    assert r == 7
    check_factorisation(T, factors, tol=1e-9)
