import numpy as np
def cdf_funct(x):
    ans=np.arange(len(x)) / float(len(x)-1)
    return ans


def inv_cdf_funct(y,x):
    # x has to be in format np([])
    x=x.reshape(x.size,)

    sorted_x_axis=np.sort(x)
    idx_pos= y*float(len(sorted_x_axis))
    # print(idx_pos)
    # -1 because when calculating cdf_funct
    # the function appends an additional value to
    # the sorted array. However, when it comes to the inverse
    # we deal with the original sorted array. That is the one
    # without the additional x
    ans=  sorted_x_axis[int(idx_pos)-1]
    return ans