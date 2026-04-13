import sys
import numpy as np

sys.path.append('..')
from FedroTensor.tensor_rank import cp_rank_factorise
from FedroTensor.matmul_tensor import build_matmul_tensor

if len(sys.argv) != 5:
    print(f'usage: {sys.argv[0]} n m p r', file=sys.stderr)
    exit(1)
n, m, p, r = map(int, sys.argv[1:5])
T = build_matmul_tensor(n, m, p)
initial = list(np.load(f'{n}_{m}_{p}_{r}_real.npy', allow_pickle=True))
initial = [arr + 1e-5 * np.random.randn(*arr.shape) for arr in initial]
factors = cp_rank_factorise(T, r, initial=initial, rational=True, denominators=(0, 1, 2, 3, 4, 2 ** 1.5), verbose=True)
if factors is not None:
    print(f'FOUND RATIONAL FACTORS n={n} m={m} p={p} r={r}')
    np.save(f'{n}_{m}_{p}_{r}_rational', factors)
else:
    print(f'NOT FOUND RATIONAL FACTORS n={n} m={m} p={p} r={r}')
