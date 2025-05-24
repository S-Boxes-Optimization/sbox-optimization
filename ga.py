import random
import numpy as np
from numba import jit

from ddt import ddt, evaluate, evaluate_table


try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("Error: DEAP no está instalado. Por favor, instálalo con 'pip install deap'")
    exit()

# --- Definición de Tipos DEAP (se hace una vez, fuera de la clase) ---
# Si ya existe, DEAP puede dar error. Usamos 'try-except' para evitarlo.
try:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
except Exception:
    # Si ya existen los tipos, no hacemos nada.
    pass

class SBoxGA:
    """
    Implementa un Algoritmo Genético para buscar S-boxes
    minimizando la Uniformidad Diferencial.
    """

    def __init__(self, n, m, **kwargs):
        """
        Inicializa el Algoritmo Genético para S-boxes.

        Args:
            n (int): Número de bits de entrada.
            m (int): Número de bits de salida.
            **kwargs: Parámetros opcionales para el AG:
                population_size (int): Tamaño de la población (def: 300).
                cxpb (float): Probabilidad de crossover (def: 0.8).
                mutpb (float): Probabilidad de mutación (def: 0.2).
                ngen (int): Número de generaciones (def: 150).
                tournsize (int): Tamaño del torneo para selección (def: 3).
                bijectivity_penalty (float): Penalización si n=m y no es biyectiva (def: 1000.0).
        """
        self.n = n
        self.m = m
        self.num_inputs = 2**n
        self.num_outputs = 2**m

        # --- Parámetros del AG con valores por defecto ---
        self.population_size = kwargs.get('population_size', 300)
        self.cxpb = kwargs.get('cxpb', 0.8)
        self.mutpb = kwargs.get('mutpb', 0.2)
        self.ngen = kwargs.get('ngen', 150)
        self.tournsize = kwargs.get('tournsize', 3)
        self.bijectivity_penalty = kwargs.get('bijectivity_penalty', 1000.0)
        self.initial_sbox = kwargs.get('initial_sbox', None)

        # --- Configurar DEAP Toolbox ---
        self.toolbox = base.Toolbox()
        self._setup_deap_toolbox()

        print(f"SBoxGA inicializado para {self.n}x{self.m}. Población: {self.population_size}, Generaciones: {self.ngen}.")

    def _evaluate_sbox_general(self, individual):
        """Función de fitness: Sum of dx = 0 plus std."""
        t = ddt(individual, self.n, self.m)
        return (evaluate_table(t, self.n, self.m), )
        #return (ddt_std_value + mult * nonzero + np.max(t[1:, :]),)
    
    def _get_individual(self, n):
        if self.num_inputs == self.num_outputs:
            sboxes = []
            for _ in range(n):
                sbox = list(range(self.num_inputs))
                random.shuffle(sbox)
                sboxes.append(creator.Individual(sbox))
            return sboxes
        return tools.initRepeat(list, self.toolbox.individual, n)

    def _setup_deap_toolbox(self):
        """Configura las herramientas de DEAP."""
        # Generador de atributos (un entero aleatorio)
        self.toolbox.register("attr_int", random.randrange, self.num_outputs)
        # Generador de individuos (lista de N atributos)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                               self.toolbox.attr_int, n=self.num_inputs)
        # Generador de población
        self.toolbox.register("population", self._get_individual)

        # Operadores Genéticos
        self.toolbox.register("evaluate", self._evaluate_sbox_general)
        if self.num_inputs == self.num_outputs:
            self.toolbox.register("mate", tools.cxOrdered)
            self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        else:
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.num_outputs - 1, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

    def run(self):
        """Ejecuta el algoritmo genético y devuelve la mejor S-box encontrada."""
        pop = self.toolbox.population(n=self.population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        print(f"--- Iniciando Evolución ---")

        # Ejecuta el algoritmo
        population, log = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb, mutpb=self.mutpb, ngen=self.ngen,
                                              stats=stats, halloffame=hof, verbose=True)

        print("--- Evolución Terminada ---")

        best_individual = hof[0]
        best_fitness_value = self._evaluate_sbox_general(best_individual)[0] # Con posible penalización
        
        # Recalculamos DDT y Max DDT sin penalización para mostrar

        print("\n--- Resultados ---")
        print(f"Mejor S-box encontrada: {best_individual}")
        print(f"Fitness (Max DDT + Penalización): {best_fitness_value}")

        # Comprobamos si es biyectiva
        if self.n == self.m:
            is_bijective = len(set(best_individual)) == len(best_individual)
            print(f"¿Es biyectiva? {is_bijective}")
            if not is_bijective:
                 print(f"  Salidas únicas: {len(set(best_individual))}/{len(best_individual)}")

        return best_individual