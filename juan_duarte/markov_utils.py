# markov_utils.py
import numpy as np
from ddt import display_ddt, ddt   

def transition_matrix(discrete_vector: np.ndarray,
                      n_in: int, n_out: int,
                      skip_zero: bool = True) -> np.ndarray:
    """
    Devuelve la matriz de transición P (float64) a partir del S-box
    en forma de vector discreto (lookup table).

    Parameters
    ----------
    discrete_vector : np.ndarray  (len = 2**n_in)
        Tabla del S-box.
    n_in, n_out : int
        Tamaños en bits de entrada y salida.
    skip_zero : bool
        Si True descarta la fila Δx = 0 porque no se usa en cripto-ataques.
    """
    counts, _ = ddt(discrete_vector, n_in, n_out)   # matriz ddt

    if skip_zero:
        counts = counts[1:, 1:]                     # El dx = 0, no da informacion

    row_sums = counts.sum(axis=1, keepdims=True)

    return counts / row_sums.astype(np.float64)     # matriz P
