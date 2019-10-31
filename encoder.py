import numpy as np
import pandas as pd
from huffman import encode as h_encode


def get_sub_images(image, box_size=8):
    """
    Gets an images of arbitrary size
    and return a reshaped array of (box_size, box_size) elements
    Args:
        image (numpy ndarray): Image input we want to divide to box sub_images.
         Should have shape (length, width, n_channels)
          e. g. n_channels = 3 for RGB
         box_size (int): Size of the box sub images
    Returns:
        divided_image (numpy ndarray, dtype = "uint8"): array of divided images
         - should have a shape of (X, box_size, box_size, n_channels).

    """
    # convert image to Greyscale to smiplify the operations
    image = image.convert('L')

    nrow = np.int(np.floor(image.size[0]/box_size))
    ncol = np.int(np.floor(image.size[1]/box_size))

    # make the image into a square to simplify operations based
    #  on the smaller dimension
    d = min(ncol, nrow)
    image = image.resize((nrow*box_size, ncol*box_size))

    image_array = np.asarray(image)  # convert image to numpy array

    # Note: images are converted to uint8 datatypes since they range between
    #  0-255. different datatypes might misbehave (based on my trials)
    image_blocks = np.asarray([np.zeros((8, 8), dtype='uint8')
                               for i in range(d)], dtype='uint8')

    # break down the image into blocks
    for i in range(0, d):
        image_blocks[i] = image_array[i*box_size: i*box_size+box_size,
                                      i*box_size:i*box_size+box_size]

    # If you want to reconvert the output of this function into images,
    #  use the following line:
    #block_image = Image.fromarray(output[idx])

    return image_blocks


def dct(sub_image):
    """
    Applies Discrete Cosine Transform on a square image:
    Args:
        sub_image (numpy ndarray): should have a shape of (box_size,box_size)
    Returns:
        transformed_sub_image (numpy ndarray): image in DCT domain
         with same size as input
    """
    b = sub_image.shape[0]  # block size
    i = j = np.arange(b)
    # basis function

    def basis(u, v):
        return np.dot(np.cos((2*i + 1) * u * np.pi / (2*b)).reshape(-1, 1),
                      np.cos((2*j + 1) * v * np.pi / (2*b)).reshape(1, -1))
    # scaling function

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
    return np.array([dct(sub_image) for sub_image in subdivded_image])


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
    return np.array([sub_image // quantization_table for sub_image in
                     dct_divided_image])

def generate_indecies_zigzag(rows = 8, cols = 8):
    """
    Gets the dimensions of an array, typically a square matrix,
    and returns an array of indecies for zigzag traversal
    
    NOTE:
    -This function imagines the matrix as a 4 wall room
    -Needed for the serialize and deserialized functions
    """
    #initial indecies
    i = j = 0
    #This is to change the style of traversing the matrix
    going_up = True
    
    forReturn = [[0,0] for i in range(rows*cols)]
    
    for step in range(rows*cols):
        # take a step up
        i_new, j_new = (i-1, j+1) if going_up else (i+1, j-1)
        
        forReturn[step] = [i,j]
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
            assert step == (rows*cols -1)
        else:
            i, j = i_new, j_new
        
    return forReturn

def serialize(quantized_dct_image):
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
    rows, columns = quantized_dct_image[0].shape
    output = np.zeros(len(quantized_dct_image)*rows*columns, dtype='int')
    

    for matrix in quantized_dct_image:
        step = 0
        for i, j in generate_indecies_zigzag(rows, columns):
            output[step] = matrix[i,j] 
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
    max_len = 255  # we do not want numbers bigger than 255
    rlcoded = []
    zero_count = 0
    # Local Variables
    #
    # logic
    for number in serialized:
        if number == 0:
            zero_count += 1
            if zero_count == max_len:
                rlcoded.append(0)
                rlcoded.append(zero_count)
                zero_count = 0
        else:
            if zero_count > 0:
                rlcoded.append(0)
                rlcoded.append(zero_count)
                zero_count = 0
            rlcoded.append(number)
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
    counts_dict = dict(pd.Series(rlcoded).value_counts())
    code_dict = h_encode(counts_dict)
    # list of strings to one joined string
    huffcoded = ''.join([code_dict[i] for i in rlcoded])
    return huffcoded, code_dict
