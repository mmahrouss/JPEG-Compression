import numpy as np
import pandas as pd
from huffman import decode as h_decode
from encoder import generate_indecies_zigzag


def huffman_decode(huffcoded, code_dict):
    """
    Decodes a string of a List of 0 and 1
     (same choice for decoder and encoder)
    Args:
        huffcoded : List or String of 0s and 1s code to be sent or stored
        code_dict (dict): dict of symbol : code in binary
    Returns:
        rlcoded (numpy ndarray): 1d array
    """
    return h_decode(huffcoded, code_dict)


def run_length_decode(rlcoded):
    """
    Returns 1D array of serialized dct values from an encoded 1D array.
    Args:
        rlcoded  (numpy ndarray): 1d array
          Encoded in decimal not binary [Kasem]
    Returns:
        serialized (numpy ndarray): 1d array
          has shape (X*box_size*box_size*n_channels,)
    """
    # Local Variables
    serialized = []
    i = 0
    # Local Variables
    while i < len(rlcoded):
        if rlcoded[i] == 0:
            serialized.extend([0]*rlcoded[i+1])
            i += 2
        else:
            serialized.append(rlcoded[i])
            i += 1
    return np.asarray(serialized)


def deserialize(serialized, n_blocks, box_size):
    """
    Removes serialization from quantized DCT values.
    Args:
        serialized (numpy ndarray): 1d array
          has shape (X*box_size*box_size*n_channels,)
        n_blocks (int)
            number of blocks 
        box_size (int)
            size of box used in serialize function
    Returns:
        quantized (numpy ndarray): array of quantized DCT values.
          - should have a shape of (X, box_size, box_size, n_channels)
           with dtype Int
    """
    rows = columns = box_size
    output = np.zeros((n_blocks, rows, columns))
    step = 0
    for matrix in output:
        for i, j in generate_indecies_zigzag(box_size, box_size):
            matrix[i, j] = serialized[step]
            step += 1

    return output


def dequantize(quantized, quantization_table):
    """
    Divides quantization table on DCT output
    Args:
        quantized (numpy ndarray): array of quantized DCT values
        - should have a shape of (X, box_size, box_size, n_channels)
         with dct applied to all of them
        quantization_table (numpy ndarray): quantization table (matrix)
        - should have a shape of (box_size, box_size)
    Returns:
        dct_values (numpy ndarray): array of DCT values.
          same shape as dct_values but element type ints
    """
    # element by ekement multiplication. Equivelant to np.multiply()
    return np.array([block * quantization_table for block in quantized])


def idct(dct_values):
    """
    Applies Inverse Discrete Cosine Transform on DCT values:
    Args:
        dct_values (numpy ndarray): should have a shape of (box_size,box_size)
    Returns:
        sub_image (numpy ndarray): image in pixels
         with same size as input
    """


def apply_idct_to_all(subdivded_dct_values):
    """
    Maps idct to all dct values (transformed images).
    Args:
        subdivided_dct_values (numpy ndarray): array of dct values.
        - should have a shape of (X, box_size, box_size, n_channels).
    Returns:
        divided_image (numpy ndarray): array of divided images
        - should have a shape of (X, box_size, box_size, n_channels)
         with dct applied to all of them
    """


def get_reconstructed_image(divided_image, n_rows, n_cols, box_size=8):
    """
    Gets an array of (box_size,box_size) pixels
    and returns the reconstructed image
    Args:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
        n_rows: number of rows or blocks
        n_cols: number of columns in image
            the number of blocks is n_rows*n_cols
        box_size (int): Size of the box sub images
    Returns:
        reconstructed_image (numpy ndarray): Image reconstructed from the array
        of divided images.

    """
