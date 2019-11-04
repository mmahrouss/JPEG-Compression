import numpy as np
from PIL import Image
import encoder as e
import decoder as d

table_8_low = [[1,  1,  1,  1,  1,  2,  2,  4],
                [1,  1,  1,  1,  1,  2,  2,  4],
                [1,  1,  1,  1,  2,  2,  2,  4],
                [1,  1,  1,  1,  2,  2,  4,  8],
                [1,  1,  2,  2,  2,  2,  4,  8],
                [2,  2,  2,  2,  2,  4,  8,  8],
                [2,  2,  2,  4,  4,  8,  8,  16],
                [4,  4,  4,  4,  8,  8, 16,  16]]
table_8_high = [[1,    2,    4,    8,    16,   32,   64,   128],
                [2,    4,    4,    8,    16,   32,   64,   128],
                [4,    4,    8,    16,   32,   64,   128,  128],
                [8,    8,    16,   32,   64,   128,  128,  256],
                [16,   16,   32,   64,   128,  128,  256,  256],
                [32,   32,   64,   128,  128,  256,  256,  256],
                [64,   64,   128,  128,  256,  256,  256,  256],
                [128,  128,  128,  256,  256,  256,  256,  256]]

table_16_low = np.repeat(np.repeat(table_8_low, 2, axis=0), 2, axis=1)
table_16_high = np.repeat(np.repeat(table_8_high, 2, axis=0), 2, axis=1)


def encode(image, box_size, quantization_table):
    """
      Gets an images of arbitrary size
      and returns a string of a list of 0 and 1 representing the compressed encoded image
      Args:
           image (PIL image): original image that needs to be reshaped and grayscaled
           box_size (int): Size of the box sub images
           quantization_table (numpy array): Table used to quantize dct values
      Returns:
          huffcoded : List or String of 0s and 1s code to be sent or stored
          code_dict (dict): dict of symbol : code in binary
        n_rows: number of rows or blocks
        n_cols: number of columns in image
          the number of blocks is n_rows*n_cols
    """
    # Reshape image and divide it into blocks
    image_array = e.reshape_image(image, box_size)
    sub_images, n_rows, n_cols = e.get_sub_images(image_array, box_size)
    
    # Apply DCT
    dct_values = e.apply_dct_to_all(sub_images)
    
    # Quantize DCT values
    quantized = e.quantize(dct_values, quantization_table)
    
    # Serialize the values
    serialized = e.serialize(quantized)
    
    # Perform run length encoding
    rlcoded = e.run_length_code(serialized)
    
    # Perform huffman encoding
    huffcoded, code_dict = e.huffman_encode(rlcoded)

    return huffcoded, code_dict, n_rows, n_cols


def decode(huffcoded, code_dict, n_rows, n_cols, box_size, quantization_table):
    """
      Gets a string of a list of 0 and 1 representing the compressed encoded image
      and returns a reconstructed image.
      Args:
           huffcoded : List or String of 0s and 1s code to be sent or stored
           code_dict (dict): dict of symbol : code in binary
           n_rows: number of rows or blocks
           n_cols: number of columns in image
               the number of blocks is n_rows*n_cols
           box_size (int): Size of the box sub images
           quantization_table (numpy array): Table used to quantize dct values
      Returns:
          reconstructed_image (numpy ndarray): Image reconstructed from the array
          of divided images.
    """
    # Huffman Decoding
    rlcoded = d.huffman_decode(huffcoded, code_dict)
    
    # Runlength decoding
    serialized = d.run_length_decode(rlcoded)
    
    # Deserialize
    quantized = d.deserialize(serialized, n_rows*n_cols, box_size)
    
    # Dequantize
    subdivded_dct_values = d.dequantize(quantized, quantization_table)
    
    # IDCT
    sub_images = d.apply_idct_to_all(subdivded_dct_values)
    
    # Reconstructed image
    reconstructed_image = d.get_reconstructed_image(sub_images, n_rows, n_cols,
                                                    box_size)

    return reconstructed_image
