import numpy as np


def get_sub_images(image, box_size):
    """
    Gets an images of arbitrary size
    and return a reshaped array of (box_size, box_size) elements
    Args:
        image (numpy ndarray): Image input we want to divide to box sub_images.
         Should have shape (length, width, n_channels)
          e. g. n_channels = 3 for RGB
         box_size (int): Size of the box sub images
    Returns:
        divided_image (numpy ndarray): array of divided images
         - should have a shape of (X, box_size, box_size, n_channels).

    """


def dct(sub_image):
    """
    Applies Discrete Cosine Transform on a square image:
    Args:
        sub_image (numpy ndarray): should have a shape of (box_size,box_size)
    Returns:
        transformed_sub_image (numpy ndarray): image in DCT domain
         with same size as input
    """
    #Extract the box size of the sub_image
    box_size, dummy = sub_image.size
    #initialize x and y ranges to be used for the basis functions.
    x = y = np.arange(box_size) 
    #Calculate the basis functions
    basis_functions = lambda u,v : np.dot(np.cos((2*x + 1)*u*np.pi/16).reshape(-1,1),
                                          np.cos((2*y.T + 1)*v*np.pi/16).reshape(1,-1))
    transformed_sub_image = np.zeros((box_size,box_size))
    #Perform DCT on the sub_image
    for u in range(box_size):
        for v in range(box_size):
            transformed_sub_image[u,v] = np.sum(basis_functions(u,v)*sub_image)/16
    #Scale down rows and columns by 2 except element(0,0) by 4.        
    transformed_sub_image[0,:] = transformed_sub_image[0,:]/2
    transformed_sub_image[:,0] = transformed_sub_image[:,0]/2
    return dct


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
        - should have a shape of (X, box_size, box_size, n_channels)
         with dct applied to all of them
        quantization_table (numpy ndarray): quantization table (matrix)
        - should have a shape of (box_size, box_size)
    Returns:
        quantized_dct_image (numpy ndarray): array of quantized image.
          same shape as dct_divided_image but element type ints
    """


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
