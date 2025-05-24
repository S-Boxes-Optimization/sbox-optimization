# main_markov_demo.py
import numpy as np
from sbox         import AESSBox, AsconSBox
from ddt          import display_ddt, ddt
from juan_duarte.markov_utils  import transition_matrix
from juan_duarte.markov_chain  import DifferentialMarkovChain


# ───────────────────────────────────────────────────────────
# 1.  Construir la DDT y la matriz de transición
box = AsconSBox()
vector = np.array([box.apply(i) for i in range(1 << box.input_size)], dtype=np.uint8)
P = transition_matrix(vector, box.input_size, box.output_size, skip_zero=True)

print("Cada fila suma 1: ", np.allclose(P.sum(axis=1), 1.0))
print(P)

# ───────────────────────────────────────────────────────────
# 2.  Crear la cadena y ver la distribución estacionaria
chain = DifferentialMarkovChain(P, init="uniform")
pi_stat = chain.stationary_distribution()
print("Primeras 10 probabilidades estacionarias:", pi_stat[:10])

# 3.  Probabilidades tras r rondas (por ejemplo r = 5)
r = 5
P5 = chain.r_step_matrix(r)    # P^5

dx0 = 2         # estado inicial Δx = 3 (en la nomenclatura omitimos Δx = 0)
print(f"\nProbabilidades de Δy después de {r} rondas con Δx₀={dx0}:")
for dy, p in enumerate(P5[dx0]):
    if p > 0:
        print(f"  Δy = {dy+1:02d}  →  {p:.6f}")

# ───────────────────────────────────────────────────────────
# 4.  Simular una trayectoria
trajectory = chain.simulate(15, random_state=42)
print("\nTrayectoria simulada (Δx codificados 1..63):")
print(trajectory + 1)          # +1 porque omitimos el estado 0


# ───────────────────────────────────────────────────────────
# 5. Verificacion cadena de markov
div_result = chain.analyze_markov_chain()
print( f"is_doubly_stochastic: {div_result['is_doubly_stochastic']} \n is_ergodic: {div_result['is_ergodic']} \n nconvergence_rate: {div_result['convergence_rate']}")
