import numpy as np
from scipy.signal import correlate2d as conv2d
from scipy.signal import convolve
import math

def get_conv_kernel(N, L1, L2):
    '''Generate cellular automaton convolutional kernel

    Example
    ==========
    get_conv_kernel(8, 2, 4):
    
     0  0 -1 -1 -1 -1  0  0
     0  0 -1 -1 -1 -1  0  0
     0  0 -1 -1 -1 -1  0  0
    -1 -1  4  4  4  4 -1 -1
    -1 -1  4  4  4  4 -1 -1
     0  0 -1 -1 -1 -1  0  0
     0  0 -1 -1 -1 -1  0  0
     0  0 -1 -1 -1 -1  0  0
    '''
    A = np.zeros((N, N))
    A[(N - L1) // 2:(N + L1) // 2, :(N - L2) // 2] = -1
    A[(N - L1) // 2:(N + L1) // 2, (N + L2) // 2:] = -1
    A[:(N - L1) // 2, (N - L2) // 2:(N + L2) // 2] = -1
    A[(N + L1) // 2:,(N - L2) // 2:(N + L2) // 2] = -1
    A[(N - L1) // 2:(N + L1) // 2,(N - L2) // 2:(N + L2) // 2] = (L1 + L2) * N / L1 / L2 - 2
    return A

def numpy_update(alive_map1, alive_map2, alive_map3, conv_kernel):
    """Perform one step of a cellular automaton with 3 channels"""
    # Convolve
    conv_result1 = conv2d(alive_map1 + alive_map2, conv_kernel, mode='same')
    conv_result2 = conv2d(alive_map2 + alive_map3, conv_kernel, mode='same')
    conv_result3 = conv2d(alive_map3 + alive_map1, conv_kernel, mode='same')
    
    # Apply game rules
    condition1 = (conv_result1 > 0) * 2 - 1
    condition2 = (conv_result2 > 0) * 2 - 1
    condition3 = (conv_result3 > 0) * 2 - 1
    np.copyto(alive_map1, condition1)
    np.copyto(alive_map2, condition2)
    np.copyto(alive_map3, condition3)
    
def numpy_update_sep(alive_map1, alive_map2, alive_map3, conv_kernel, channel_mixing=None):
    """Perform one step of a cellular automaton with 3 independent channels"""
    # Convolve
    if channel_mixing is None:
        conv_result1 = conv2d(alive_map1, conv_kernel, mode='same')
        conv_result2 = conv2d(alive_map2, conv_kernel, mode='same')
        conv_result3 = conv2d(alive_map3, conv_kernel, mode='same')
    elif isinstance(channel_mixing, str) and channel_mixing == '3D':
        maps = np.stack([alive_map1, alive_map2, alive_map3], axis=-1)
        conv_results = convolve(maps, conv_kernel, mode='same')
        conv_result1, conv_result2, conv_result3 = list(map(np.squeeze, np.dsplit(conv_results, 3)))
    else:
        conv_kernel1, conv_kernel2, conv_kernel3 = list(map(np.squeeze, np.dsplit(conv_kernel, 3)))
        conv_result1 = conv2d(alive_map1, conv_kernel1, mode='same')
        conv_result2 = conv2d(alive_map2, conv_kernel2, mode='same')
        conv_result3 = conv2d(alive_map3, conv_kernel3, mode='same')
        conv_results = np.stack([conv_result1, conv_result2, conv_result3], axis=-1)
        conv_results = np.matmul(conv_results, channel_mixing)
        conv_result1, conv_result2, conv_result3 = list(map(np.squeeze, np.dsplit(conv_results, 3)))
        
    # Apply game rules
    condition1 = (conv_result1 > 0) * 2 - 1
    condition2 = (conv_result2 > 0) * 2 - 1
    condition3 = (conv_result3 > 0) * 2 - 1
    np.copyto(alive_map1, condition1)
    np.copyto(alive_map2, condition2)
    np.copyto(alive_map3, condition3)
    
    
def get_conv_kernel_bw(N, L1, L2):
    '''Generate cellular automaton convolutional kernel

    Example
    ==========
    get_conv_kernel(8, 2, 4):
    
    -2 -2 -2 -2 -2 -2 -2 -2
    -2 -2 -2 -2 -2 -2 -2 -2
    -2 -2 -2 -2 -2 -2 -2 -2
    -2 -2  7  7  7  7 -2 -2
    -2 -2  7  7  7  7 -2 -2
    -2 -2 -2 -2 -2 -2 -2 -2
    -2 -2 -2 -2 -2 -2 -2 -2
    -2 -2 -2 -2 -2 -2 -2 -2
    '''
    A = -2 * np.ones((N, N))
    A[(N - L1) // 2:(N + L1) // 2,(N - L2) // 2:(N + L2) // 2] = N ** 2 / L1 / L2 - 1
    return A


def numpy_update_bw(alive_map, conv_kernel):
    """Perform one step of a cellular automaton with 1 channel"""
    # Convolve
    conv_result = conv2d(alive_map, conv_kernel, mode='same')

    # Apply game rules
    condition = (conv_result > 0) * 2 - 1
    np.copyto(alive_map, condition)

    
def generate_pattern(
        img_size, conv_kernel=None, n_iterations=20,
        separate_channels=False, kernel_type='2D', pointwise_kernel=None,
        kernel_N=13, kernel_L1=5, kernel_L2=5, init_maps=None):
    """Generate a pattern with a 3-channel cellular automaton of given size and convolutional kernel"""
    
    # Create kernel if none was given
    if conv_kernel is None:
        conv_kernel = get_conv_kernel(kernel_N, kernel_L1, kernel_L2)
        separate_channels = False
        kernel_type = '2D'
        pointwise_kernel = None
        
    if kernel_type not in ('2D', '3D', 'pointwise'):
        raise ValueError("kernel_type must be one of ('2D', '3D', 'pointwise'), \
                          otherwise no channel mixing is available")
    
    # Initialize game field
    if init_maps is None:
        alive_map1 = np.random.choice([-1, 1], size=img_size)
        alive_map2 = np.random.choice([-1, 1], size=img_size)
        alive_map3 = np.random.choice([-1, 1], size=img_size)
    else:
        alive_map1, alive_map2, alive_map3 = init_maps

    # Play the game
    for i in range(n_iterations):
        if not separate_channels:
            numpy_update(alive_map1, alive_map2, alive_map3, conv_kernel)
        else:
            if kernel_type == '2D':
                numpy_update_sep(alive_map1, alive_map2, alive_map3, conv_kernel, channel_mixing=None) 
            elif kernel_type == '3D':
                numpy_update_sep(alive_map1, alive_map2, alive_map3, conv_kernel, channel_mixing='3D') 
            else:
                numpy_update_sep(alive_map1, alive_map2, alive_map3, conv_kernel, channel_mixing=pointwise_kernel) 
    
    # Nurislam's faulty code:
    #img = np.concatenate([alive_map1, alive_map2, alive_map3]).reshape((*img_size, 3))
    #img = ((img + 1) / 2 * 255).astype(np.uint)
    
    # My version:
    img = np.stack([alive_map1, alive_map2, alive_map3], axis=-1)
    img = (img * 10).astype(np.int)
    return img


def generate_pattern_bw(img_size, conv_kernel, n_iterations=20):
    """Generate a pattern with a 1-channel cellular automaton of given size and convolutional kernel"""
    # Initialize game field
    alive_map = np.random.choice([-1, 1], size=img_size)

    for i in range(n_iterations):
        numpy_update_bw(alive_map, conv_kernel)
    
    img = np.concatenate([alive_map] * 3).reshape((*img_size, 3))
    #img = ((img + 1) / 2 * 255).astype(np.uint)
    img = (img * 10).astype(np.int)
    return img
