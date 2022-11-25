"""
Module containing some helper functions to do some math.
"""
import sys
import numpy as np_real

USE_CUPY = False
if '-c' in sys.argv:
    import cupy as np
    USE_CUPY = True
    print("Using cupy")
else:
    print("Using Numpy")
    import numpy as np


def get_real_numpy_array(x):
    """
    Returns a real numpy array (instead of cupy)
    """
    if USE_CUPY:
        return np_real.array(np.asnumpy(x))
    
    return x


def convert_to_cupy_array(x):
    """
    Returns a cupy array, if cupy exists
    """
    if USE_CUPY:
        return np.array(x)
    return x


def generate_grid_distances(shape):
    """
    generates a grid containing distances from a point, used to quickly
    calculate distances.
    """
    distances = np.zeros(shape)
    to_visit = [(0, 0)]
    visited = []
    while True:
        try:
            point_x, point_y = to_visit.pop(0)
        except IndexError:
            break

        if (point_x, point_y) in visited:
            continue

        if point_x >= shape[0]-1:
            continue

        if point_y >= shape[1]-1:
            continue

        for x in range(0, shape[0]):
            for y in range(0, shape[1]):
                if distances[x][y] == 0:
                    d = distance(np.array([0, 0]), np.array([x, y]), shape)
                    distances[x][y] = d
                    distances[y][x] = d

    return distances


def squash(x):
    """
    Maps a function between -pi and +pi
    """
    x = np.where(x <= -np.pi, x+2*np.pi, x)
    return np.where(np.pi <= x, x-2*np.pi, x)


def distance(p1, p2, dimension):
    """
    calculates the distances between two points
    """
    total = 0
    for i, (a, b) in enumerate(zip(p1, p2)):
        delta = abs(b - a)
        if delta > dimension[i] - delta:
            delta = dimension[i] - delta
        total += delta ** 2
    return total ** 0.5


def get_correlation_fft(a):
    """
    Returns the two point correlation using the FFT
    """
    a = np.array(a)
    af = np.abs(np.fft.fft2(a))**2
    corr = np.real(np.fft.ifft2(af))
    corr = get_real_numpy_array(corr)
    return corr#[:,0]


def correlation_lengths_fft(a1, a2):
    """
    Calculates the correlation length using FFT
    """
    T_SIZE, X_SIZE, Y_SIZE = a1.shape
    a1c = get_correlation_fft(a1)
    a2c = get_correlation_fft(a2)

    corr = (a1c + a2c) / 2

    correlations = np_real.zeros((T_SIZE, X_SIZE, Y_SIZE))
    for i in range(0, T_SIZE):
        correlations[i, :, :] = corr[i, :, :] / corr[i, 0, 0]

    return correlations
    
