import numpy as np
import sys
import torch

from copy import deepcopy
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Callable, List, Sequence, Union


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
                 weight_decay: float = 1e-2,
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
                 params: Sequence[NDArray],
                 loss: Callable[[Sequence[torch.Tensor]], torch.Tensor]):
        self.params = []
        self.params_ext2in: List[List[int]] = [[] for _ in range(len(params))]
        self.params_in2ext: List[int] = []
        for idx, param in enumerate(params):
            if np.iscomplexobj(param):
                self.params.append(torch.tensor(np.real(param), requires_grad=True))
                self.params.append(torch.tensor(np.imag(param), requires_grad=True))
                self.params_ext2in[idx] = [len(self.params) - 2, len(self.params) - 1]
                self.params_in2ext.append(idx)
                self.params_in2ext.append(idx)
            else:
                self.params.append(torch.tensor(param, requires_grad=True))
                self.params_ext2in[idx] = [len(self.params) - 1]
                self.params_in2ext.append(idx)
        self.loss = lambda params_in: loss(self.__convert_params(params_in))
        self.fixed = [torch.full_like(param, torch.nan) for param in self.params]

    def __convert_params(self, params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [params[self.params_ext2in[ext_id][0]] if len(self.params_ext2in[ext_id]) == 1 else
                params[self.params_ext2in[ext_id][0]] + 1j * params[self.params_ext2in[ext_id][1]]
                for ext_id in range(len(self.params_ext2in))]

    def __descent(self,
                  config: DescentConfig,
                  verbose: bool = False) -> bool:
        optimiser = torch.optim.Adam(self.params, lr=config.lr, weight_decay=config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=config.lr_decay)
        pbar = tqdm(total=config.num_iter, unit='batches') if verbose else None
        loss_prev = torch.Tensor(1)
        plateau_cnt = 0
        for _ in range(config.num_iter):
            loss = self.loss(self.params)
            mx = torch.max(torch.abs(torch.cat(list(map(torch.flatten, self.params)))))
            if pbar is not None:
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

    def __try_sieve(self,
                    target: Sequence[torch.Tensor],
                    used: Sequence[torch.Tensor],
                    mask: Sequence[bool],
                    config: DescentConfig,
                    verbose: bool = False) -> bool:
        min_delta = torch.inf
        min_ind = None
        min_param_id = None
        for param_id in range(len(self.params)):
            if not mask[param_id]:
                continue
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
        cnt_params = sum(np.prod(p.shape) for i, p in enumerate(self.params) if mask[i])
        cnt_fixed = sum((~f.isnan()).sum().item() for i, f in enumerate(self.fixed) if mask[i])
        cnt_used = sum(u.sum() for i, u in enumerate(used) if mask[i])
        if verbose:
            print(f'done {cnt_fixed} | skipped {cnt_used - cnt_fixed - 1} | total {cnt_params} | sieving '
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

    def get_params(self) -> List[NDArray]:
        """
        Access the parameters after the optimisation.

        Returns:
            Sequence of NumPy arrays.
        """
        return [param.detach().numpy() for param in self.__convert_params(self.params)]

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

    @staticmethod
    def rationals(start: float, stop: float, denominators: Sequence[float]) -> List[List[float]]:
        return [[0.0]] + [list(map(float, np.arange(start, stop + 1 / (2 * d), 1 / d))) for d in denominators]

    def sieve(self,
              mask: Union[Sequence[bool], None] = None,
              layers: Sequence[Sequence[float]] = rationals(-5, 5, (1, 2, 3, 4, 6, 8)),
              verbose: bool = False,
              **desc_kwargs) -> bool:
        """
        Parameter sieving by greedy rounding to the nearest value.

        Args:
            mask: Which parameters to sieve (default all).
            layers: Sets of points to round the parameters to.
            verbose: Whether to print debug info.
            desc_kwargs: Config parameters for gradient descent.

        Returns:
            Whether the procedure successfully sieved all the parameters.
        """
        if mask is not None:
            mask = [mask[self.params_in2ext[i]] for i in range(len(self.params))]
        else:
            mask = [True] * len(self.params)
        config = DescentConfig(**desc_kwargs)
        for layer in map(torch.tensor, layers):
            used = [~f.isnan() for f in self.fixed]
            while any((mask[i] & ~u).any() for i, u in enumerate(used)):
                target = [layer[torch.argmin(torch.abs(p[..., None] - layer), dim=-1)] for p in self.params]
                self.__try_sieve(target, used, mask, config, verbose=verbose)
        if any([f.isnan().any() for i, f in enumerate(self.fixed) if mask[i]]):
            return False
        return True
