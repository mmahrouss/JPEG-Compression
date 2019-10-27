import numpy as np
import pandas as pd
from huffman import decode as h_decode


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


def deserialize(serialized):
    """
    Removes serialization from quantized DCT values.
    Args:
        serialized (numpy ndarray): 1d array
          has shape (X*box_size*box_size*n_channels,)
    Returns:
        quantized (numpy ndarray): array of quantized DCT values.
          - should have a shape of (X, box_size, box_size, n_channels)
           with dtype Int
    """


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


def get_reconstructed_image(divided_image, box_size=8):
    """
    Gets an array of (box_size,box_size) pixels
    and returns the reconstructed image
    Args:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
        box_size (int): Size of the box sub images
    Returns:
        reconstructed_image (numpy ndarray): Image reconstructed from the array
        of divided images.

    """
