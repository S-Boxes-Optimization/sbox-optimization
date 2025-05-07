from abc import ABC, abstractmethod
import numpy as np
from tabulate import tabulate

class SBox(ABC):
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size

    @abstractmethod
    def apply(self, input: int) -> int:
        pass



class DESBox(SBox):
    table = [
        [0b0010, 0b1100, 0b0100, 0b0001, 0b0111, 0b1010, 0b1011, 0b0110, 0b1000, 0b0101, 0b0011, 0b1111, 0b1101, 0b0000, 0b1110, 0b1001],
        [0b1110, 0b1011, 0b0010, 0b1100, 0b0100, 0b0111, 0b1101, 0b0001, 0b0101, 0b0000, 0b1111, 0b1100, 0b0011, 0b1001, 0b1000, 0b0110],
        [0b0100, 0b0010, 0b0001, 0b1011, 0b1100, 0b1101, 0b0111, 0b1000, 0b1111, 0b1001, 0b1100, 0b0101, 0b0110, 0b0011, 0b0000, 0b1110],
        [0b1011, 0b1000, 0b1100, 0b0111, 0b0001, 0b1110, 0b0010, 0b1101, 0b0110, 0b1111, 0b0000, 0b1001, 0b1100, 0b0100, 0b0101, 0b0011]
    ]

    def __init__(self):
        super().__init__(6, 4)

    def apply(self, input: int) -> int:
        if input > 0b111111 or input < 0:
            return -1
        row = (((input >> 5) & 1) << 1) | (input & 1)
        col = (input >> 1) & 0b1111

        return self.table[row][col]

    @classmethod
    def print_from_representation(cls, discrete_representation: np.ndarray):
        table = [[0 for _ in range(17)] for _ in range(4)]

        for i in range(len(discrete_representation)):
            row = (((i >> 5) & 1) << 1) | (i & 1)
            col = (i >> 1) & 0b1111
            table[row][col + 1] = bin(0b1111 & discrete_representation[i])

        for i in range(4):
            table[i][0] = bin(i)

        print(tabulate(table, headers=[bin(i) for i in range(16)], tablefmt="fancy_grid"))



class AsconSBox(SBox):
    table = [0x4, 0xb, 0x1f, 0x14, 0x1a, 0x15, 0x9, 0x2, 0x1b, 0x5, 0x8, 0x12, 0x1d, 0x3, 0x6, 0x1c, 0x1e, 0x13, 0x7, 0xe, 0x0, 0xd, 0x11, 0x18, 0x10, 0xc, 0x1, 0x19, 0x16, 0xa, 0xf, 0x17]

    def __init__(self):
        super().__init__(5, 5)


    def apply(self, input: int) -> int:
        return self.table[input]