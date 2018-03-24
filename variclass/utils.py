# -*- coding: utf-8 -*-
import numpy as np

def scale_dataset(X, mag_range=(0,1)):

    new_X = np.copy(X)

    """
    for i in range(new_X.shape[2]):
    
        this_dataset = new_X[:,:,i]

        data_min = np.amin(this_dataset, axis=None)
        data_max = np.amax(this_dataset, axis=None)

        data_range = data_max - data_min
        this_scale = (mag_range[1] - mag_range[0]) / data_range
        this_min = mag_range[0] - data_min * this_scale

        this_dataset *= this_scale
        this_dataset += this_min
    
    return new_X
    """

    data_min = np.amin(new_X, axis=None)
    data_max = np.amax(new_X, axis=None)

    data_range = data_max - data_min
    this_scale = (mag_range[1] - mag_range[0]) / data_range
    this_min = mag_range[0] - data_min * this_scale

    new_X *= this_scale
    new_X += this_min
    
    return new_X