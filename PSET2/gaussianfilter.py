import numpy as np

def gaussian_filter(sigma):
    # calculate the kernel size using sigma for the gaussian filter
    k_size = int(2 * np.ceil(3 * sigma) + 1)
    # create the kernel using a lambda function that iterates over all possible [x,y] coordinates in a matrix of size k_size*k_size
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-1*((x-(k_size-1)/2)**2 + (y-(k_size-1)/2)**2)/(2*sigma**2)), (k_size, k_size))
    return kernel / np.sum(kernel)