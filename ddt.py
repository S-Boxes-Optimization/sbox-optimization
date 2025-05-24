import numpy as np

from sbox import SBox
from tabulate import tabulate
from numba import jit

def get_discrete_vector_representation(sbox: SBox) -> np.ndarray:
    size = 1 << sbox.input_size
    indices = np.arange(size, dtype=np.uint32)
    return np.array([sbox.apply(i) for i in indices])

@jit(nopython=True)
def ddt(discrete_vector: np.ndarray, input_size: int, output_size: int) -> tuple[np.ndarray, np.floating]:
    table = np.zeros((1 << input_size, 1 << output_size), dtype=np.uint32)

    for dx in range(1 << input_size):
        for x in range(1 << input_size):
            x_prime = x ^ dx
            assert discrete_vector[x] < 1 << output_size
            assert discrete_vector[x_prime] < 1 << output_size
            dy = discrete_vector[x] ^ discrete_vector[x_prime]
            table[dx][dy] += 1

    return table


def evaluate(sbox: SBox) -> np.floating:
    discrete_vector = get_discrete_vector_representation(sbox)
    table = ddt(discrete_vector, sbox.input_size, sbox.output_size)
    ddt_std_value = np.std(table[1:, 1:])
    nonzero = np.count_nonzero(table[1:, 0])
    mult = 100 if sbox.input_size == sbox.output_size else 1
    return ddt_std_value + mult * nonzero + np.max(table[1:, :])

@jit(nopython=True)
def evaluate_table(table, n, m):
    ddt_std_value = np.std(table[1:, 1:])
    nonzero = np.count_nonzero(table[1:, 0])
    mult = 100 if n == m else 1
    return ddt_std_value + mult * nonzero + np.max(table[1:, :])


def display_ddt(table: np.ndarray):
    print(tabulate(table, headers="keys", tablefmt="fancy_grid"))