from sbox import AESSBox
import numpy as np
from juan_duarte.markov_utils import transition_matrix
from juan_duarte.markov_chain import DifferentialMarkovChain



box    = AESSBox()
#vec    = np.fromiter((box.apply(i) for i in range(256)), dtype=np.uint8)

# 8×8  ⇒  n_in = n_out = 8
vec = np.array([box.apply(i) for i in range(1 << box.input_size)], dtype=np.uint8)

P = transition_matrix(vec, n_in=8, n_out=8, skip_zero=True)

# 3. la cadena
chain = DifferentialMarkovChain(P, init="uniform")

print("∥πP-π∥₁   =", np.linalg.norm(chain.stationary_distribution() @ P
                                    - chain.stationary_distribution(), 1))
print("gap       =", chain.spectral_gap())
for r in range(1, 5):
    print(f"r={r}:  max P^r = {chain.max_probability(r):.3e}, "
          f"dTV = {chain.tv_distance(r):.3e}")
print("mezcla ε=2⁻²⁰ en", chain.mixing_time(eps=2**-20), "pasos")
