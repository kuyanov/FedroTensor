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
factors = cp_rank_factorise(T, r, num_attempts=10**9, verbose=True)
if factors is not None:
    print(f'FOUND REAL FACTORS n={n} m={m} p={p} r={r}')
    np.save(f'{n}_{m}_{p}_{r}_real.npy', np.array(factors, dtype=object), allow_pickle=True)
else:
    print(f'NOT FOUND REAL FACTORS n={n} m={m} p={p} r={r}')
