import numpy as np
import pandas as pd
from huffman import encode as h_encode


def reshape_image(image, box_size=8):
    """
    Gets an image of arbitrary size
    and returns a reshaped array of (box_size, box_size) elements
    Args:
        image (PIL image): original image that needs to be reshaped and
                            grayscaled
        box_size (int): Size of the box sub images
    Returns:
        image_array (numpy ndarray, dtype = "uint8"): image reshaped to m x m
        np array.
    """
    # convert image to Greyscale to smiplify the operations
    image = image.convert('L')

    n_rows = np.int(np.floor(image.size[0]/box_size))
    n_cols = np.int(np.floor(image.size[1]/box_size))

    image = image.resize((n_rows*box_size, n_cols*box_size))

    image_array = np.asarray(image)  # convert image to numpy array
    return image_array


def get_sub_images(image_array, box_size=8):
    """
    Gets a grayscale image and returns an array of (box_size, box_size) elements
    Args:
        image_array (numpy ndarray): Image input we want to divide to box
                                     sub_images.
         Should have shape (length, width, n_channels) where length = width
          e. g. n_channels = 3 for RGB
         box_size (int): Size of the box sub images
    Returns:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
         - should have a shape of (X, box_size, box_size, n_channels).
        n_rows: number of rows or blocks
        n_cols: number of columns in image
          the number of blocks is n_rows*n_cols
    """
    n_rows = np.int(image_array.shape[0]/box_size)
    n_cols = np.int(image_array.shape[1]/box_size)

    # make the image into a square to simplify operations based
    #  on the smaller dimension
    # d = min(n_cols, n_rows)

    # Note: images are converted to uint8 datatypes since they range between
    #  0-255. different datatypes might misbehave (based on my trials)
    image_blocks = np.asarray([np.zeros((box_size, box_size), dtype='uint8')
                               for i in range(n_rows*n_cols)], dtype='uint8')

    # break down the image into blocks
    c = 0
    for i in range(n_rows):
        for j in range(n_cols):
            image_blocks[c] = image_array[i*box_size: i*box_size+box_size,
                                          j*box_size:j*box_size+box_size]
            c += 1

    # If you want to reconvert the output of this function into images,
    #  use the following line:
    # block_image = Image.fromarray(output[idx])

    return image_blocks, n_rows, n_cols


def __basis_generator(b=8):
    """
        Helper local function to generate dct basis and cache them
        if the basis is calculated before it gets re-used again
        Args:
            b (int): Size of the box sub images
        Returns: basis (function): function that takes u,v and returns the
                                basis matrix and caches it
    """
    i = j = np.arange(b)
    basis_cache = {}

    def helper(u, v):
        base = basis_cache.get((u, v), None)
        if base is None:
            base = np.dot(np.cos((2*i + 1) * u * np.pi / (2*b)).reshape(-1, 1),
                          np.cos((2*j + 1) * v * np.pi / (2*b)).reshape(1, -1))
            basis_cache[(u, v)] = base
        return base
    return lambda u, v: helper(u, v)


def dct(sub_image, basis):
    """
    Applies Discrete Cosine Transform on a square image:
    Args:
        sub_image (numpy ndarray): should have a shape of (box_size,box_size)
    Returns:
        transformed_sub_image (numpy ndarray): image in DCT domain
         with same size as input
    """
    b = sub_image.shape[0]  # block size

    def scale(idx):
        return 2 if idx == 0 else 1
    outblock = np.zeros((b, b))

    for u in range(b):
        for v in range(b):
            outblock[u, v] =\
                np.sum(basis(u, v) * sub_image) / \
                (b**2/4) / scale(u) / scale(v)

    return outblock


def apply_dct_to_all(subdivded_image):
    """
    Maps dct to all subimages
    Args:
        divided_image (numpy ndarray): array of divided images
        - should have a shape of (X, box_size, box_size, n_channels).
    Returns:
        dct_divided_image (numpy ndarray): array of divided images
        - should have a shape of (X, box_size, box_size, n_channels)
         with dct applied to all of them
    """
    basis = __basis_generator(subdivded_image.shape[1])
    dct_divided_image = np.array([dct(sub_image, basis)
                                  for sub_image in subdivded_image])
    return dct_divided_image


def quantize(dct_divided_image, quantization_table):
    """
    Multiplies quantization table on DCT output
    Args:
        dct_divided_image (numpy ndarray): array of divided images
        - should have a shape of (n_blocks, box_size, box_size, n_channels)
         with dct applied to all of them
        quantization_table (numpy ndarray): quantization table (matrix)
        - should have a shape of (box_size, box_size)
    Returns:
        quantized_dct_image (numpy ndarray): array of quantized image.
          same shape as dct_divided_image but element type ints
    """
    return np.array([(sub_image / quantization_table).round().astype(int)
                     for sub_image in dct_divided_image])


def generate_indicies_zigzag(rows=8, cols=8):
    """
    Gets the dimensions of an array, typically a square matrix,
    and returns an array of indecies for zigzag traversal

    NOTE:
    -This function imagines the matrix as a 4 wall room
    -Needed for the serialize and deserialized functions
    """
    # initial indecies
    i = j = 0
    # This is to change the style of traversing the matrix
    going_up = True

    forReturn = [[0, 0] for i in range(rows*cols)]

    for step in range(rows*cols):
        # take a step up
        i_new, j_new = (i-1, j+1) if going_up else (i+1, j-1)

        forReturn[step] = [i, j]
        if i_new >= rows:
            # you hit the ground
            j += 1
            going_up = not going_up
        elif j_new >= cols:
            # you hit the right wall
            i += 1
            going_up = not going_up
        elif i_new < 0:
            # you hit the ceiling
            j += 1
            going_up = not going_up
        elif j_new < 0:
            # you hit the right wall
            i += 1
            going_up = not going_up
        elif i_new == rows and j_new == cols:
            # you are done
            assert step == (rows*cols - 1)
        else:
            i, j = i_new, j_new

    return forReturn


def serialize(quantized_dct_image, jpeg2000=False):
    """
    Serializes the quantized image
    Args:
        quantized_dct_image (numpy ndarray): array of quantized image.
          - should have a shape of (X, box_size, box_size, n_channels)
           with dtype Int
    Returns:
        serialized (numpy ndarray): 1d array
          has shape (X*box_size*box_size*n_channels,)
    """
    # All about resizing right.

    # This approach is simple. While travelling the matrix in the usual
    #  fashion, on basis of parity of the sum of the indices of the element,
    #  add that particular element to the list either at the beginning or
    #  at the end if sum of i and j is either even or odd respectively.
    #  Print the solution list as it is.

    if not jpeg2000:
        rows, columns = quantized_dct_image[0].shape
        output = np.zeros(len(quantized_dct_image)*rows*columns, dtype='int')
        step = 0
        for matrix in quantized_dct_image:
            for i, j in generate_indicies_zigzag(rows, columns):
                output[step] = matrix[i, j]
                step += 1
    else:
        rows, columns = quantized_dct_image.shape
        output = np.zeros(rows*columns, dtype='int')
        step = 0
        for i, j in generate_indicies_zigzag(rows, columns):
            output[step] = quantized_dct_image[i, j]
            step += 1

    return output


def run_length_code(serialized):
    """
    Applied run length coding to the serialized image.
    Args:
        serialized (numpy ndarray): 1d array
          has shape (X*box_size*box_size*n_channels,)
    Returns:
        rlcoded  (numpy ndarray): 1d array
          Encoded in decimal not binary [Kasem]
    """
    # Local Variables
    max_len = 256  # we do not want numbers bigger than 255
    rlcoded = []
    zero_count = 0  # counter for zeros
    # logic
    for number in serialized:
        if number == 0:
            zero_count += 1
            if zero_count == max_len:
                # max number of zeros reached
                rlcoded.append(0)  # indicator of zeros
                rlcoded.append(zero_count-1)  # number of zeros
                zero_count = 0
        else:
            if zero_count > 0:
                rlcoded.append(0)
                rlcoded.append(zero_count-1)
                zero_count = 0
            rlcoded.append(number)
    # for handeling trailing zeros
    if zero_count > 0:
        rlcoded.append(0)
        rlcoded.append(zero_count-1)
    # logic
    return np.asarray(rlcoded)


def huffman_encode(rlcoded):
    """
    Encodes The run-length coded again with Huffman coding.
    returns a string of a List of 0 and 1
     (same choice for decoder and encoder)
    Args:
        rlcoded (numpy ndarray): 1d array
    Returns:
        huffcoded : List or String of 0s and 1s code to be sent or stored
        code_dict (dict): dict of symbol : code in binary
    """
    # get a dictionary of the frequency of each symbol
    counts_dict = dict(pd.Series(rlcoded).value_counts())
    # get the huffman encoding dictionary / map
    code_dict = h_encode(counts_dict)
    # list of strings to one joined string
    # encode each symbol to a string of zeros and ones and stitch together
    huffcoded = ''.join([code_dict[i] for i in rlcoded])
    return huffcoded, code_dict
