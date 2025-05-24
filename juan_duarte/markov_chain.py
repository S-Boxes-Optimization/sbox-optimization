# markov_chain.py
import numpy as np
from numpy.linalg import eigvals, matrix_power


class DifferentialMarkovChain:
    """
    Cadena de Markov basada en la tabla diferencial de un S-box.
    """

    def __init__(self, P: np.ndarray, init: str | np.ndarray = "uniform"):
        """
        Parameters
        ----------
        P : np.ndarray
            Matriz de transición (float64, filas que suman 1).
        init : "uniform", "stationary" o np.ndarray
            Distribución inicial de estados.
        """
        self.P = P.copy()
        self.k = P.shape[0]

        if isinstance(init, str):
            if init == "uniform":
                self.pi0 = np.full(self.k, 1 / self.k)
            elif init == "stationary":
                self.pi0 = self.stationary_distribution()
            else:
                raise ValueError("init debe ser 'uniform', 'stationary' o un vector numpy.")
        else:
            init = np.asarray(init, dtype=np.float64)
            assert init.shape == (self.k,) and np.isclose(init.sum(), 1.0)
            self.pi0 = init

    # ------------------------------------------------------------------
    #  Propiedades teóricas
    # ------------------------------------------------------------------
    def stationary_distribution(self, tol: float = 1e-12, max_iter: int = 10_000) -> np.ndarray:
        """Devuelve el vector estacionario π tal que πP = π."""
        pi = np.full(self.k, 1 / self.k)
        for _ in range(max_iter):
            new_pi = pi @ self.P
            if np.linalg.norm(new_pi - pi, 1) < tol:
                return new_pi
            pi = new_pi
        raise RuntimeError("No converge; quizá la cadena no sea ergódica.")

    def r_step_matrix(self, r: int) -> np.ndarray:
        """Devuelve P^r."""
        return np.linalg.matrix_power(self.P, r)
    
    # -------- espectro
    def spectral_gap(self):
        """Devuelve γ = 1 − |λ₂| (λ₂ = segundo autovalor en módulo)."""
        lams = np.sort(np.abs(eigvals(self.P)))
        lam2 = lams[-2].real
        return 1.0 - lam2

    # -------- potencia r
    def P_power(self, r: int):
        return matrix_power(self.P, r)

    # -------- máxima probabilidad diferencial a r pasos
    def max_probability(self, r: int):
        return self.P_power(r).max()

    # -------- distancia total a la estación (peor estado inicial) a r pasos
    def tv_distance(self, r: int):
        pi = self.stationary_distribution()
        diff = 0.5 * np.abs(self.P_power(r) - pi).sum(axis=1)
        return diff.max()

    # -------- “tiempo de mezcla” aproximado para ε dado
    def mixing_time(self, eps: float = 2**-20):
        gap = self.spectral_gap()
        if gap == 0:
            return np.inf
        from math import log, ceil
        return ceil(log(1 / eps) / gap)

    # ------------------------------------------------------------------
    #  Simulación
    # ------------------------------------------------------------------
    def simulate(self, steps: int, random_state: int | None = None) -> np.ndarray:
        """
        Devuelve una trayectoria (array de ints) de longitud `steps`
        siguiendo las probabilidades de transición.
        """
        rng = np.random.default_rng(random_state)
        traj = np.empty(steps, dtype=np.int32)

        traj[0] = rng.choice(self.k, p=self.pi0)
        for t in range(1, steps):
            traj[t] = rng.choice(self.k, p=self.P[traj[t-1]])
        return traj
    
    
    def analyze_markov_chain(self):
        # Analiza las propiedades de una cadena de Markov
        # Verificar si es doblemente estocástica
        is_doubly_stochastic = (np.allclose(np.sum(self.P, axis=0), 1) and 
                            np.allclose(np.sum(self.P, axis=1), 1))
        
        # Verificar ergodicidad de manera simple
        is_ergodic = np.all(np.linalg.matrix_power(self.P, self.P.shape[0]) > 0)
        
        # Analizar valores propios para tasa de convergencia
        eigenvalues = np.linalg.eigvals(self.P)
        sorted_eigs = sorted(abs(eigenvalues), reverse=True)
        
        convergence_rate = sorted_eigs[1] if len(sorted_eigs) > 1 else 0
        
        return {
            "is_doubly_stochastic": is_doubly_stochastic,
            "is_ergodic": is_ergodic,
            "convergence_rate": convergence_rate,
            "eigenvalues": sorted_eigs
    }

    