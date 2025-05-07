import numpy as np

from ddt import ddt, get_discrete_vector_representation
from sbox import SBox


class Search:
    sbox: SBox
    representation: np.ndarray
    best: np.ndarray

    def __init__(self, sbox: SBox):
        self.sbox = sbox
        self.representation = get_discrete_vector_representation(sbox)
        self.best = np.copy(self.representation)


    def search(self, iterations: int = 100, debug = False):
        _, initial_std = ddt(self.representation, self.sbox.input_size, self.sbox.output_size)
        best_std = initial_std
        max_step_size = ((1 << self.sbox.output_size) + 1)

        ratio = (max_step_size - 2) / iterations

        for i in range(iterations):
            to_add = np.random.randint(low=0, high=max_step_size - ratio, size=self.representation.shape)
            to_eval_1 = (self.representation + to_add) % (1 << self.sbox.output_size)
            to_eval_2 = (self.representation - to_add) % (1 << self.sbox.output_size)

            _, std_1 = ddt(to_eval_1, self.sbox.input_size, self.sbox.output_size)
            _, std_2 = ddt(to_eval_2, self.sbox.input_size, self.sbox.output_size)

            if std_1 < best_std:
                best_std = std_1
                self.best = to_eval_1

            if std_2 < best_std:
                best_std = std_2
                self.best = to_eval_2

            if debug:
                print("\rIteration:", i, "Current decay:", max_step_size - ratio, "Current best:", best_std, end="")
            max_step_size -= ratio

        print()
        print(best_std)