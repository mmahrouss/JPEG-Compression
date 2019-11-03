import numpy as np
import pandas as pd
from huffman import encode as h_encode


def lfilter(taps, array, filter_centre):
    """
    Applies a FIR filter with symmetric and periodic padding
    M. Rabbani, R. Joshi described the mentioned padding.
    Args:
        taps (list): taps of the FIR filter
        array (np.ndarray): array to be filtered.
        filter_centre (int):the index of the origin tap
            i.e. the index corresponding to h(0), used for padding
    Returns 
        filtered_arrat (np.ndarray): filtered array.
                                    same length as array.
    """
    arr = array.copy()
    left_pad_len = len(taps) - filter_centre - 1
    right_pad_len = filter_centre
    arr = np.concatenate(
        (array[1:1+left_pad_len][::-1], array,
         array[-right_pad_len-1:-1][::-1]))
    return np.convolve(arr, taps[::-1], 'valid')


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
    Gets a grayscale image and returns an array of (box_size, box_size)
    elements
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
    return np.array([dct(sub_image, basis) for sub_image in subdivded_image])


def check_image(image):
    """
    Check if the image has valid dimensions and if not would resize the image
    to valid dimensions
    (valid dimensions are the divisble by 8 on rows and columns since the max
     level of decompostion is 3)
    Args:
        image (PIL): image input from the user
    Returns:
        image_array()
    """
    rows, cols = image.size
    n_rows = round(rows/8) * 8
    n_cols = round(cols/8) * 8
    d = min(n_rows, n_cols)
    image = image.resize((d, d))
    image_array = np.asarray(image)
    return image_array


def dwt(image_array, quantization_Array):
    """
    Gets an image of arbitrary size
    and return an array of the same size containing 4 different versions of the
    image by filtering the rows and colums using a low pass or a high pass
    filter with the different combinations and quantized by the quantization
    array
    Args:
        image_array (numpy ndarray): Image input we want to transform.
         Should have shape (length, width)

         quantization_Array (List of ints): An array that contains four values
         for the quantization of each image
        should be 1D and have 4 elements
    Returns:
        filtered_image (numpy ndarray): array of the 4 images [LL,LH,HL,HH]
         - should have a shape of (X, box_size, box_size).

    """
    # Create the high pass and low pass filters
    # both filters are non-causal
    # symmetric
    #     [-2,       -1,    0,    1,      2]
    LPF = [-0.125, 0.25, 0.75, 0.25, -0.125]
    LPF_center = 2

    #     [  -2,-1,    0]
    HPF = [-0.5, 1, -0.5]
    HPF_center = 2

    nrow, ncol = image_array.shape

    # create an array that will contain the 4 different subbands of the image
    LL = np.zeros((nrow, ncol))
    LH = np.zeros((nrow, ncol))
    HL = np.zeros((nrow, ncol))
    HH = np.zeros((nrow, ncol))
    filtered_image = [LL, LH, HL, HH]

    # filtering the rows using a low pass and high pass filters
    LowPass_rows = np.zeros((nrow, ncol))
    HighPass_rows = np.zeros((nrow, ncol))
    for i in range(0, nrow):
        LowPass_rows[i, :] = lfilter(LPF, image_array[i, :], LPF_center)
        HighPass_rows[i, :] = lfilter(HPF, image_array[i, :], HPF_center)

    # down sample rows.
    # which means we will have half the number of columns
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i][:, ::2]

    # apply filters accross columns
    for i in range(0, ncol):
        LL[:, i] = lfilter(LPF, LowPass_rows[:, i], LPF_center)
        LH[:, i] = lfilter(HPF, LowPass_rows[:, i], HPF_center)
        HL[:, i] = lfilter(LPF, HighPass_rows[:, i], LPF_center)
        HH[:, i] = lfilter(HPF, HighPass_rows[:, i], HPF_center)

    # down sample columns and quantize
    for i in range(0, len(filtered_image)):
        filtered_image[i] = filtered_image[i][::2, :]
        filtered_image[i] = np.round(
            filtered_image[i]/quantization_Array[i]).astype(int)

    return filtered_image


def dwt_levels(filtered_image, Levels, quantization_Array):
    """
    Gets an array of 4 elements (the output of the dwt function)
    and return an array by replacing the elements of the list that are
    addressed through the Levels array by dwt versions of them (replace 1
                                             element with a List of 4 elements)

    Args:
        filtered_image (numpy ndarray): The output of the dwt function that
        would be decomposed further.
         should have 4 elements

        quantization_Array (List): An array that contains four values for the
        quantization of each image
        should have 4 elemets

        Levels (a list of lists): The parts of the image that will be
        decomposed further.
        The Levels list should look like this [[0],[0,1],[1]]
        The above list means that the LL image would be decomposed again,
        then the new LH that was created from the LL image would be decomposed
        again, then the LH of the original image would be decomposed
        Adressing should use this code below
        LL:0
        LH:1
        HL:2
        HH:3

    """
    for i in range(0, len(Levels)):

        if len(Levels[i]) > i+1:
            raise Exception(
                '''The Array is not sorted correctly.An element that does not
                exist is called. The value of the subarray
                was: {}'''.format(Levels[i]))

        if len(Levels[i]) > 3:
            raise Exception(
                '''The length of each subarray should not exceed 3.
                 The value of the subarray was: {}'''.format(Levels[i]))

        if len(Levels[i]) == 1:

            filtered_image[Levels[i][0]] = dwt(
                filtered_image[Levels[i][0]], quantization_Array)

        if len(Levels[i]) == 2:

            filtered_image[Levels[i][0]][Levels[i][1]] = dwt(
                filtered_image[Levels[i][0]][Levels[i][1]], quantization_Array)

        if len(Levels[i]) == 3:

            filtered_image[Levels[i][0]][Levels[i][1]][Levels[i][2]] = dwt(
                filtered_image[Levels[i][0]][Levels[i][1]][Levels[i][2]],
                quantization_Array)


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


def generate_indecies_zigzag(rows=8, cols=8):
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


def dwt_serialize(filtered_image, output, length):
    """
    This function takes the output of the dwt_levels and serializes the list.
    The serialization is done by order of apperance in the filtered_image
    e.g.:[[LL,LH,HL,HH],LH,HL,HH] 
    is serialized by taking the first element , if found to be a list then the
    elements within this list
    would each be serialized and appended to to the output list, if found to be
    a numpy array then it would be serialized without further steps.

    args:
    filtered_image(list): This should be a list that can contain either numpy
                            arrays or a list of numpy arrays 
    output(list): should be passed as an empty list that will contain the final
                  serialized data of the image
    length(list):should be passed as an empty list that will contain the
             serialized length of each numpy array within the filtered_image


    """
    for i in filtered_image:
        if isinstance(i, list):
            # append the output of the recursion to the main arguments (output,
            # length)
            output_temp, length_temp = dwt_serialize(i, [], [])
            output = output + output_temp
            length.append(length_temp)
        else:
            # append the data of the serialized elements to the main arguments
            # (output,length)
            new_output = (serialize(i, True).tolist())
            output = output+new_output
            length = length+[len(new_output)]
    return output, length


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
            for i, j in generate_indecies_zigzag(rows, columns):
                output[step] = matrix[i, j]
                step += 1
    else:
        rows, columns = quantized_dct_image.shape
        output = np.zeros(rows*columns, dtype='int')
        step = 0
        for i, j in generate_indecies_zigzag(rows, columns):
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
    max_len = 255  # we do not want numbers bigger than 255
    rlcoded = []
    zero_count = 0  # counter for zeros
    # logic
    for number in serialized:
        if number == 0:
            zero_count += 1
            if zero_count == max_len:
                # max number of zeros reached
                rlcoded.append(0)  # indicator of zeros
                rlcoded.append(zero_count)  # number of zeros
                zero_count = 0
        else:
            if zero_count > 0:
                rlcoded.append(0)
                rlcoded.append(zero_count)
                zero_count = 0
            rlcoded.append(number)
    # for handeling trailing zeros
    if zero_count > 0:
        rlcoded.append(0)
        rlcoded.append(zero_count)
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
