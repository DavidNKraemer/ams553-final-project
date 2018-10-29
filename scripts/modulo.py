"""
This is a meaningless file only used to demonstrate what this directory is for.
"""
import numpy as np
import matplotlib.pyplot as plt


def multiplication_table(modulus):
    """
    Generates the multiplication table for the finite field Z/nZ, where n is the
    specified modulus.
    """
    modulo_ring = np.arange(0, modulus, 1)

    rows, cols = np.meshgrid(modulo_ring, modulo_ring)
    return (rows * cols) % modulus

def generate_image(modulus):
    """
    Generates a plot of the multiplication table for the finite field Z/nZ,
    where n is the specified modulus.
    """
    table = multiplication_table(modulus)
    fig, axis = plt.subplots(1, 1)
    axis.imshow(table)
    fig.savefig(f"../plots/multiplication_modulo_{modulus}.png",
                bbox_inches='tight')

generate_image(1009)
