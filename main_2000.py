import numpy as np
from PIL import Image
from encoder import huffman_encode, run_length_code
from decoder import huffman_decode, run_length_decode
import encoder_2000 as e
import decoder_2000 as d

def encode(image, levels, quantization_Array):
    """
    Gets an image of arbitrary size along with the quantization array that 
    will be used to quantize the decomposed DWT levels, and the levels to be composed.
    It returns the encoded image in a string of 0s and 1s, the code_dict used for encoding, and
    the serialized length of each numpy array within the filtered_image.
    Args: 
        image (PIL): image input from the user
        Levels (a list of lists): The parts of the image that will be
        decomposed further.
        quantization_Array (List): An array that contains four values for the
        quantization of each image
        should have 4 elemets.
    Returns:
        huffcoded : List or String of 0s and 1s code to be sent or stored
        code_dict (dict): dict of symbol : code in binary
        length(list): serialized length of each numpy array within the filtered_image.
    """
    #resize image and return it as an array
    im_arr = e.check_image(image)
    
    #perform dwt
    filtered_image = e.dwt(im_arr,quantization_Array)
    e.dwt_levels(filtered_image,levels, quantization_Array)
    
    #Perform serialization recursively
    serialized, length = e.dwt_serialize(filtered_image,output =[],length = [])
    
    #Perform runlength encoding 
    rlcoded = run_length_code(serialized)
    
    #Perform huffman encoding
    huffcoded, code_dict = huffman_encode(rlcoded)

    return huffcoded, code_dict, length



def decode( huffcoded, code_dict, length, quantization_Array):
    """
    Gets a string of a list of 0 and 1 representing the compressed encoded image
      and returns a reconstructed image.
    Args:
        huffcoded : List or String of 0s and 1s code to be sent or stored
        code_dict (dict): dict of symbol : code in binary
        length(list): serialized length of each numpy array within the filtered_image.
        quantization_Array (List): An array that contains four values for the
        quantization of each image
        should have 4 elemets.
    Returns: 
        im_arr (np array): the reconstructed image
    """
    #huffman decoding 
    rlcoded = huffman_decode(huffcoded,code_dict)
    
    #run length decoding
    serialized = run_length_decode(rlcoded)
    
    #inverse DWT
    im_arr = d.dwt_deserialize(serialized,length, quantization_Array)

    return im_arr
