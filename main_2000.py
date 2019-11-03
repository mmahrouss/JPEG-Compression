import numpy as np
from PIL import Image
import encoder as e
import decoder as d

def encode(image, levels, quantization_Array):
    
    im_arr = e.check_image(image)
    
    filtered_image = e.dwt(im_arr,quantization_Array)

    e.dwt_levels(filtered_image,levels, quantization_Array)
    
    serialized, length = e.dwt_serialize(filtered_image,output =[],length = [])
    #Perform runlength encoding 
    rlcoded = e.run_length_code(serialized)
    #Perform huffman encoding
    huffcoded, code_dict = e.huffman_encode(rlcoded)

    return huffcoded, code_dict, length

def decode( huffcoded, code_dict, length, quantization_Array):

    rlcoded = d.huffman_decode(huffcoded,code_dict)

    serialized = d.run_length_decode(rlcoded)

    im_arr = d.dwt_deserialize(serialized,length, quantization_Array)

    return im_arr
