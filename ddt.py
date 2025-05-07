import numpy as np

from sbox import SBox
from tabulate import tabulate

def get_discrete_vector_representation(sbox: SBox) -> np.ndarray:
    size = 1 << sbox.input_size
    indices = np.arange(size, dtype=np.uint32)
    return np.array([sbox.apply(i) for i in indices])


def ddt(discrete_vector: np.ndarray, input_size: int, output_size: int) -> tuple[np.ndarray, np.floating]:
    table = np.zeros((1 << input_size, 1 << output_size), dtype=np.uint32)

    for dx in range(1 << input_size):
        for x in range(1 << input_size):
            x_prime = x ^ dx
            assert discrete_vector[x] < 1 << output_size
            assert discrete_vector[x_prime] < 1 << output_size
            dy = discrete_vector[x] ^ discrete_vector[x_prime]
            table[dx][dy] += 1

    return table, np.std(table)


def display_ddt(table: np.ndarray):
    print(tabulate(table, headers="keys", tablefmt="fancy_grid"))