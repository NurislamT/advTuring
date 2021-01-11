import numpy as np

def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1J / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W

def df(fourier, dft_matrix):
    fmax = fourier.max()
    new = np.real(dft_matrix.dot(fourier*(np.abs(fourier)>fmax-1)).dot(dft_matrix))
    print(fmax, np.abs(fourier).mean())
    new_pattern1 = np.sign(new)*10
    new = np.real(dft_matrix.dot(fourier*(np.abs(fourier)>0.8*fmax)).dot(dft_matrix))
    new_pattern2 = np.sign(new)*10
    return new_pattern1, new_pattern2


def new_dft_patterns(pattern):
    n_channels = pattern.shape[-1]
    dft_size = pattern.shape[0]
    W = DFT_matrix(dft_size)
    pattern1 = []
    pattern2 = []
    for i in range(n_channels):
        channel = pattern[..., i]
        channel_dft = W.dot(channel).dot(W)
        new_channel1, new_channel2 = df(channel_dft, W)
        pattern1.append(new_channel1)
        pattern2.append(new_channel2)
        
    pattern1 = np.stack(pattern1, axis=-1)
    pattern2 = np.stack(pattern2, axis=-1)
    return pattern1, pattern2