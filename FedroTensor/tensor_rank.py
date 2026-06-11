import numpy as np
import sys
import torch

from numpy.typing import NDArray
from typing import List, Sequence, Tuple, Union

from .descent import DescentOptimiser


def cp_rank_loss(T: torch.Tensor, factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Loss for the CP rank optimisation.

    Args:
        T: Torch tensor.
        factors: Rank decomposition of T.

    Returns:
        Norm distance between T and the rank decomposition.
    """
    letters = 'bcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    d = len(factors)
    if d > len(letters):
        raise ValueError(f'Order {d} is too large for this simple einsum builder.')

    lhs = ','.join(f'{letters[i]}a' for i in range(d))
    rhs = letters[:d]
    return torch.linalg.norm(torch.einsum(f'{lhs}->{rhs}', *factors) - T)


def cp_rank_factorise(A: NDArray,
                      r: int,
                      initial: Union[Sequence[NDArray], None] = None,
                      num_attempts: int = 10,
                      complex: bool = False,
                      symmetric: bool = False,
                      rational: bool = False,
                      layers: Sequence[Sequence[float]] = DescentOptimiser.rationals(-5, 5, (1, 2, 3, 4, 6, 8)),
                      verbose: bool = False,
                      **desc_kwargs) -> Union[List[NDArray], None]:
    """
    Checks whether A admits a CP rank decomposition of rank r.

    Args:
        A: NumPy array.
        r: Conjectured CP rank of A.
        initial: Initial solution.
        num_attempts: Number of optimisation trials.
        complex: Whether rank decomposition is complex-valued.
        symmetric: Compute symmetric rank decomposition (factors are equal).
        rational: Whether to perform sieving for computing rational factors.
        layers: Sets of points for sieving.
        verbose: Whether to print debug info.
        desc_kwargs: Config parameters for gradient descent.

    Returns:
        Rank decomposition of rank r, if found (None otherwise).
    """
    T = torch.tensor(A)
    for _ in range(num_attempts):
        if not symmetric:
            params = initial or [np.random.randn(T.shape[i], r) if not complex else 
                                 np.random.randn(T.shape[i], r) + 1j * np.random.randn(T.shape[i], r) 
                                 for i in range(T.ndim)]
            loss = lambda fac: cp_rank_loss(T, fac)
        else:
            params = initial or [np.random.randn(T.shape[0], r) if not complex else
                                 np.random.randn(T.shape[0], r) + 1j * np.random.randn(T.shape[0], r)]
            loss = lambda fac: cp_rank_loss(T, [fac[0]] * T.ndim)
        optimiser = DescentOptimiser(params, loss)
        success = optimiser.optimise(verbose=verbose, **desc_kwargs)
        if not success:
            continue
        if rational:
            success = optimiser.sieve(layers=layers, verbose=verbose, **desc_kwargs)
            if not success:
                raise ValueError('Failed to compute rational factors')
        return optimiser.get_params()
    return None


def cp_rank(A: NDArray,
            r_max: int = 100,
            num_attempts: int = 10,
            complex: bool = False,
            symmetric: bool = False,
            rational: bool = False,
            layers: Sequence[Sequence[float]] = DescentOptimiser.rationals(-5, 5, (1, 2, 3, 4, 6, 8)),
            verbose: bool = False,
            **desc_kwargs) -> Tuple[int, List[NDArray]]:
    """
    Computes the CP rank of A and the corresponding rank decomposition.

    Args:
        A: NumPy array.
        r_max: Maximum CP rank.
        num_attempts: Number of optimisation trials.
        complex: Whether rank decomposition is complex-valued.
        symmetric: Compute symmetric rank decomposition (factors are equal).
        rational: Whether to perform sieving for computing rational factors.
        layers: Sets of points for sieving.
        verbose: Whether to print debug info.
        desc_kwargs: Config parameters for gradient descent.

    Returns:
        The CP rank and the corresponding rank decomposition of A.
    """
    rank_l = 0
    rank_r = r_max
    factors = None
    while rank_r - rank_l > 1:
        rank_m = (rank_l + rank_r) // 2
        if verbose:
            print(f'interval {rank_l, rank_r}, checking rank {rank_m}', file=sys.stderr)
        cur_factors = cp_rank_factorise(A, rank_m,
                                        num_attempts=num_attempts,
                                        complex=complex,
                                        symmetric=symmetric,
                                        rational=False,
                                        verbose=verbose,
                                        **desc_kwargs)
        if cur_factors is not None:
            rank_r = rank_m
            factors = cur_factors
        else:
            rank_l = rank_m
    if rational:
        factors = cp_rank_factorise(A, rank_r,
                                    initial=factors,
                                    num_attempts=num_attempts,
                                    complex=complex,
                                    symmetric=symmetric,
                                    rational=True,
                                    layers=layers,
                                    verbose=verbose,
                                    **desc_kwargs)
    if factors is None:
        raise ValueError('Failed to find rank decomposition, try increasing r_max')
    return rank_r, factors
