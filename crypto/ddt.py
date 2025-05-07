"""
Difference Distribution Table (DDT) analysis for S-boxes.
"""
from typing import Tuple, Optional
import numpy as np
from tabulate import tabulate

from .sbox import SBox, SBoxError


class DDTError(SBoxError):
    """Base exception for DDT-related errors."""
    pass


def compute_ddt(
    sbox: SBox,
    representation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Compute the Difference Distribution Table (DDT) for an S-box.

    Args:
        sbox: The S-box to analyze
        representation: Optional vector representation of the S-box.
                       If None, uses the current S-box mapping.

    Returns:
        A tuple containing:
        - The DDT as a numpy array
        - The standard deviation of the DDT

    Raises:
        DDTError: If there's an error computing the DDT
    """
    if representation is None:
        representation = sbox.get_vector_representation()

    try:
        table = np.zeros((1 << sbox.input_size, 1 << sbox.output_size), dtype=np.uint32)

        for dx in range(1 << sbox.input_size):
            for x in range(1 << sbox.input_size):
                x_prime = x ^ dx
                dy = representation[x] ^ representation[x_prime]
                table[dx][dy] += 1

        return table, float(np.std(table))
    except Exception as e:
        raise DDTError(f"Error computing DDT: {str(e)}")


def display_ddt(
    table: np.ndarray,
    title: Optional[str] = None,
    format: str = "fancy_grid"
) -> None:
    """
    Display the Difference Distribution Table in a formatted way.

    Args:
        table: The DDT to display
        title: Optional title for the table
        format: The table format to use (default: "fancy_grid")
    """
    if title:
        print(f"\n{title}")
    
    # Add row and column headers
    headers = [f"{i:02x}" for i in range(table.shape[1])]
    rows = []
    for i, row in enumerate(table):
        rows.append([f"{i:02x}"] + [str(val) for val in row])
    
    print(tabulate(rows, headers=["Δx\\Δy"] + headers, tablefmt=format))


def analyze_ddt(
    sbox: SBox,
    representation: Optional[np.ndarray] = None
) -> None:
    """
    Perform a complete DDT analysis of an S-box.

    Args:
        sbox: The S-box to analyze
        representation: Optional vector representation of the S-box.
                       If None, uses the current S-box mapping.
    """
    table, std = compute_ddt(sbox, representation)
    
    print(f"\nDDT Analysis for {sbox.__class__.__name__}")
    print(f"Input size: {sbox.input_size} bits")
    print(f"Output size: {sbox.output_size} bits")
    print(f"DDT Standard Deviation: {std:.2f}")
    
    # Find maximum value in DDT (excluding the first row and column)
    max_val = np.max(table[1:, 1:])
    print(f"Maximum differential probability: {max_val / (1 << sbox.input_size):.4f}")
    
    display_ddt(table) 