import numpy as np
from encoder_2000 import lfilter
from decoder import deserialize

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
