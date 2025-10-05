import numpy as np
import sys
import torch

from copy import deepcopy
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Callable, List, Sequence, Tuple


class DescentConfig(object):
    """
    Class isolating the gradient descent config parameters.
    Default values seem to work well in the majority of cases.
    """

    def __init__(self,
                 num_iter: int = 5000,
                 lr: float = 0.1,
                 lr_decay: float = 0.95,
                 plateau_ratio: float = 1e-6,
                 plateau_delay: int = 100,
                 weight_decay: float = 1e-3,
                 tol: float = 1e-2):
        """
        Args:
            num_iter: Number of iterations throughout the descent.
            lr: Initial learning rate passed to the PyTorch optimiser.
            lr_decay: Decay factor of the learning rate.
            plateau_ratio: Threshold below which the relative improvement of the loss is considered insignificant.
            plateau_delay: Maximum number of consecutive iterations without significant improvement of the loss.
            weight_decay: Regularisation coefficient.
            tol: Desired upper bound for the loss.
        """
        self.num_iter = num_iter
        self.lr = lr
        self.lr_decay = lr_decay
        self.plateau_ratio = plateau_ratio
        self.plateau_delay = plateau_delay
        self.weight_decay = weight_decay
        self.tol = tol


class DescentOptimiser:
    """
    Class implementing gradient descent for abstract optimisation problems.
    """

    def __init__(self,
                 shapes: Sequence[Tuple],
                 loss: Callable[[Sequence[torch.Tensor]], torch.Tensor],
                 dtype: torch.dtype = torch.float):
        self.params = [torch.randn(shape, dtype=dtype, requires_grad=True) for shape in shapes]
        self.fixed = [torch.full(shape, torch.nan, dtype=dtype) for shape in shapes]
        self.loss = loss

    def __descent(self,
                  config: DescentConfig,
                  verbose: bool = False) -> bool:
        optimiser = torch.optim.Adam(self.params, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=config.lr_decay)
        if verbose:
            pbar = tqdm(total=config.num_iter, unit='batches')
        loss_prev = torch.Tensor(1)
        plateau_cnt = 0
        for _ in range(config.num_iter):
            loss = self.loss(self.params)
            mx = torch.max(torch.abs(torch.cat(list(map(torch.flatten, self.params)))))
            if verbose:
                pbar.set_postfix(loss=f'{loss.item():.6f}', mx=f'{mx:.2f}')
                pbar.update()
            if loss.isnan() or loss.isinf():
                return False
            if loss < config.tol:
                return True
            if abs(loss_prev - loss) / loss_prev < config.plateau_ratio:
                plateau_cnt += 1
                if plateau_cnt > config.plateau_delay:
                    return False
            else:
                plateau_cnt = 0
            loss_prev = loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step(loss.item())
            with torch.no_grad():
                for p, f in zip(self.params, self.fixed):
                    p[~f.isnan()] = f[~f.isnan()]
        return False

    def __try_separate(self,
                       target: Sequence[torch.Tensor],
                       used: Sequence[torch.Tensor],
                       config: DescentConfig,
                       verbose: bool = False) -> bool:
        min_delta = torch.inf
        min_ind = None
        min_param_id = None
        for param_id in range(len(self.params)):
            deltas = torch.abs(self.params[param_id] - target[param_id])
            deltas[~self.fixed[param_id].isnan() | used[param_id]] = torch.inf
            delta = torch.min(deltas).item()
            ind = int(torch.argmin(deltas).item())
            if delta < min_delta:
                min_delta = delta
                min_ind = ind
                min_param_id = param_id
        if min_ind is None or min_param_id is None:
            raise ValueError('Empty parameters')
        min_indices = np.unravel_index(min_ind, self.params[min_param_id].shape)
        used[min_param_id][min_indices] = True
        cnt_params = sum(np.prod(p.shape) for p in self.params)
        cnt_fixed = sum((~f.isnan()).sum().item() for f in self.fixed)
        if verbose:
            print(f'[{cnt_fixed + 1}/{cnt_params}] separating '
                  f'{self.params[min_param_id][min_indices]} -> {target[min_param_id][min_indices]}',
                  file=sys.stderr)
        params_old = deepcopy(self.params)
        fixed_old = deepcopy(self.fixed)
        with torch.no_grad():
            self.params[min_param_id][min_indices] = target[min_param_id][min_indices]
            self.fixed[min_param_id][min_indices] = target[min_param_id][min_indices]
        success = self.__descent(config, verbose=verbose)
        if not success:
            with torch.no_grad():
                for param_id in range(len(self.params)):
                    self.params[param_id] = params_old[param_id]
                    self.fixed[param_id] = fixed_old[param_id]
            return True
        return False

    def __round(self, x: torch.Tensor, d: int) -> torch.Tensor:
        if torch.is_complex(x):
            return self.__round(torch.real(x), d) + 1j * self.__round(torch.imag(x), d)
        return torch.round(x * d) / d if d != 0 else torch.zeros_like(x)

    def get_params(self) -> List[NDArray]:
        """
        Access the parameters after the optimisation.

        Returns:
            Sequence of NumPy arrays.
        """
        return [param.detach().numpy() for param in self.params]

    def optimise(self,
                 verbose: bool = False,
                 **desc_kwargs) -> bool:
        """
        Run the optimisation.

        Args:
            verbose: Whether to print debug info.
            desc_kwargs: Config parameters for gradient descent.

        Returns:
            Whether the optimiser reduced the loss below the tolerance.
        """
        return self.__descent(DescentConfig(**desc_kwargs), verbose=verbose)

    def separate(self,
                 denominators: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10),
                 verbose: bool = False,
                 **desc_kwargs) -> bool:
        """
        Parameter separation by greedy rounding to the nearest rational.

        Args:
            denominators: Denominators of the nearest rational.
            verbose: Whether to print debug info.
            desc_kwargs: Config parameters for gradient descent.

        Returns:
            Whether the procedure successfully separated all the parameters.
        """
        config = DescentConfig(**desc_kwargs)
        for d in denominators:
            used = [torch.zeros_like(p, dtype=torch.bool) for p in self.params]
            while any((self.fixed[param_id].isnan() & ~used[param_id]).any()
                      for param_id in range(len(self.params))):
                target = [self.__round(p, d) for p in self.params]
                self.__try_separate(target, used, config, verbose=verbose)
        if any([f.isnan().any() for f in self.fixed]):
            return False
        return True
