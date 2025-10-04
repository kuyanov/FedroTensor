import torch

from numpy.typing import NDArray
from typing import List, Sequence, Tuple, Union

from src.descent import DescentOptimiser


def cp_rank_loss(T: torch.Tensor, factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Loss for the CP rank optimisation.

    Args:
        T: Original tensor.
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
                      num_attempts: int = 10,
                      rational: bool = False,
                      verbose: bool = False,
                      **desc_kwargs) -> Union[List[NDArray], None]:
    """
    Checks whether tensor A admits a CP rank decomposition of rank r.

    Args:
        A: Original tensor.
        r: Conjectured CP rank of A.
        num_attempts: Number of optimisation trials.
        rational: Compute rational factors.
        verbose: Whether to print debug info.
        desc_kwargs: Config parameters for gradient descent.

    Returns:
        The rank decomposition of rank r, if found (None otherwise).
    """
    T = torch.tensor(A)
    factor_shapes = [(T.shape[i], r) for i in range(A.ndim)]
    loss = lambda fac: cp_rank_loss(T, fac)
    for _ in range(num_attempts):
        optimiser = DescentOptimiser(factor_shapes, loss, dtype=T.dtype)
        success = optimiser.optimise(verbose=verbose, **desc_kwargs)
        if not success:
            continue
        if rational:
            success = optimiser.separate(verbose=verbose, **desc_kwargs)
            if not success:
                raise ValueError('Failed to compute rational factors')
        return optimiser.get_params()
    return None


def cp_rank(A: NDArray,
            r_max: int = 100,
            num_attempts: int = 10,
            rational: bool = False,
            verbose: bool = False,
            **desc_kwargs) -> Tuple[int, List[NDArray]]:
    """
    Computes the CP rank of tensor A and the corresponding rank decomposition.

    Args:
        A: Original tensor.
        r_max: Maximum possible rank of A.
        num_attempts: Number of optimisation trials.
        rational: Compute rational factors.
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
        cur_factors = cp_rank_factorise(A, rank_m,
                                        num_attempts=num_attempts,
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
                                    num_attempts=num_attempts,
                                    rational=rational,
                                    verbose=verbose,
                                    **desc_kwargs)
    return rank_r, factors
