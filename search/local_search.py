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
        t, _ = ddt(self.representation, self.sbox.input_size, self.sbox.output_size)
        ddt_std_value = np.std(t[1:, 1:])
        nonzero = np.count_nonzero(t[1:, 0])
        mult = 100 if self.sbox.input_size == self.sbox.output_size else 1
        best_score = ddt_std_value + mult * nonzero + np.max(t[1:, :])
        max_step_size = ((1 << self.sbox.output_size) + 1)

        ratio = (max_step_size - 2) / iterations

        for i in range(iterations):
            to_add = np.random.randint(low=0, high=max_step_size - ratio, size=self.representation.shape)
            to_eval_1 = (self.representation + to_add) % (1 << self.sbox.output_size)
            to_eval_2 = (self.representation - to_add) % (1 << self.sbox.output_size)

            t1, _ = ddt(to_eval_1, self.sbox.input_size, self.sbox.output_size)
            t2, _ = ddt(to_eval_2, self.sbox.input_size, self.sbox.output_size)

            ddt_std_value1 = np.std(t1[1:, 1:])
            nonzero1 = np.count_nonzero(t1[1:, 0])
            score1 = ddt_std_value1 + mult * nonzero1 + np.max(t1[1:, :])

            ddt_std_value2 = np.std(t2[1:, 1:])
            nonzero2 = np.count_nonzero(t2[1:, 0])
            score2 = ddt_std_value2 + mult * nonzero2 + np.max(t2[1:, :])

            if score1 < best_score:
                best_score = score1
                self.best = to_eval_1

            if score2 < best_score:
                best_score = score2
                self.best = to_eval_2

            if debug:
                print("\rIteration:", i, "Current decay:", max_step_size - ratio, "Current best:", best_score, end="")
            max_step_size -= ratio

        print()
        print(best_score)