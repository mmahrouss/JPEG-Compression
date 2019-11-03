from encoder import generate_indecies_zigzag, __basis_generator, lfilter
import numpy as np
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
    while i < len(rlcoded):
        if rlcoded[i] == 0:
            # found some zeros
            # add n number of zeros to result
            # where n is the subsequent number
            serialized.extend([0]*rlcoded[i+1])
            # take two steps
            i += 2
        else:
            # non-zero number, add it and take one step
            serialized.append(rlcoded[i])
            i += 1
    return np.asarray(serialized)


def deserialize(serialized, n_blocks, box_size=8):
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


def idct(dct_values, basis):
    """
    Applies Inverse Discrete Cosine Transform on DCT values:
    Args:
        dct_values (numpy ndarray): should have a shape of (box_size,box_size)
    Returns:
        sub_image (numpy ndarray): image in pixels
         with same size as input
    """
    b = dct_values.shape[0]  # block size

    outblock = np.zeros((b, b))

    for x in range(b):
        for y in range(b):
            outblock = outblock + dct_values[x, y] * basis(x, y)

    return outblock


def idwt(filtered_image, quantization_Array):
    """
        Applied the Inverse Wavelet Transform to the four subbands
        Args:
            filtered_image (list of 4 np.ndarrays):
                contains the four subbands of the image.
                i.e [LL, LH, HL, LL]
            quantization_Array (list of 4 integers):
                the quantization for each corresponding band 
        Returns:
            image_array (numpy ndarray):
                the re-constructed image or a subband if it is a deep 
                    reconstrucrion 
    """
    # dequantize
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i]*quantization_Array[i]

    # define the inverse filters
    #     [  -1, 0,    1]
    LPF = [0.5, 1, 0.5]
    LPF_center = 1

    #     [-1,       0,    1,    2,      3]
    HPF = [-0.125, -0.25, 0.75, -0.25, -0.125]
    HPF_center = 1

    # upsample the columns
    LowPass1_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    LowPass1_rows[::2, :] = filtered_image[0]

    LowPass2_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    LowPass2_rows[::2, :] = filtered_image[1]

    HighPass1_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    HighPass1_rows[::2, :] = filtered_image[2]

    HighPass2_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]))
    HighPass2_rows[::2, :] = filtered_image[3]

    # apply the inverse filters to the columns
    for i in range(0, LowPass1_rows.shape[1]):
        LowPass1_rows[:, i] = lfilter(LPF, LowPass1_rows[:, i], LPF_center)
        LowPass2_rows[:, i] = lfilter(HPF, LowPass2_rows[:, i], HPF_center)
        HighPass1_rows[:, i] = lfilter(LPF, HighPass1_rows[:, i], LPF_center)
        HighPass2_rows[:, i] = lfilter(HPF, HighPass2_rows[:, i], HPF_center)

    # overlay channels and upsamle rows
    LowPass_temp = LowPass1_rows+LowPass2_rows
    LowPass_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]*2))
    LowPass_rows[:, ::2] = LowPass_temp

    HighPass_temp = HighPass1_rows+HighPass2_rows
    HighPass_rows = np.zeros(
        (filtered_image[0].shape[0]*2, filtered_image[0].shape[1]*2))
    HighPass_rows[:, ::2] = HighPass_temp

    # apply the inverse filters to the rows
    for i in range(0, LowPass_rows.shape[0]):
        HighPass_rows[i, :] = lfilter(HPF, HighPass_rows[i, :], HPF_center)
        LowPass_rows[i, :] = lfilter(LPF, LowPass_rows[i, :], LPF_center)

    return HighPass_rows + LowPass_rows


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
    # values can be slightly less than 0.0 e.g -0.5
    # or more than 255 like 255.5
    # that is why we clip.
    # next we rounf that cast to an 8bit unsigned integer
    basis = __basis_generator(subdivded_dct_values.shape[1])
    return np.array([idct(sub_image, basis).clip(min=0, max=255).round()
                     for
                     sub_image in subdivded_dct_values]).astype(np.uint8)


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
    image_reconstructed = np.zeros((n_rows*box_size, n_cols*box_size),
                                   dtype=np.uint8)
    c = 0
    # break down the image into blocks
    for i in range(n_rows):
        for j in range(n_cols):
            image_reconstructed[i*box_size: i*box_size+box_size,
                                j*box_size:j*box_size+box_size] =\
                divided_image[c]
            c += 1

    # If you want to reconvert the output of this function into images,
    #  use the following line:
    # block_image = Image.fromarray(output[idx])

    return image_reconstructed


def dwt_deserialize(serialized, length, quantization_Array):
    """
        Deserializes and applied IDWT recursively
        recursively goes deep until four deserialized image subbands are
        available 
        then returns the IDWT of the 4 subbands
    """
    quarter_len = int(len(serialized)/4)
    images = []
    for i in range(4):
        if isinstance(length[i], list):
            # if there is a deep construction. make a recursive call.
            images.append(dwt_deserialize(serialized[quarter_len*i:
                                                     quarter_len*i +
                                                     quarter_len],
                                          length[i], quantization_Array))
        else:
            # else deserialize to get the subband
            images.append(deserialize(serialized[quarter_len*i:
                                                 quarter_len*i + quarter_len],
                                      1, int(np.sqrt(quarter_len))).squeeze())
    # returns idwt of the four subbands
    return idwt(images, quantization_Array)
