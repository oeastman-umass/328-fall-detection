import numpy as np


def calculate_mean(data):
    """Calculate the mean of the data."""
    return np.mean(data)

def calculate_std(data):
    """Calculate the standard deviation of the data."""
    return np.std(data)

def calculate_min(data):
    """Calculate the minimum of the data."""
    return np.min(data)

def calculate_max(data):
    """Calculate the maximum of the data."""
    return np.max(data)

def calculate_median(data):
    return np.median(data)

def compute_fft_features(data):
    sample_rate = 100
    fft_signal = np.fft.rfft(data)
    fft_magnitudes = np.abs(fft_signal)
    max_magnitude_index = np.argmax(fft_magnitudes)
    most_frequent_signal = fft_signal[max_magnitude_index]
    frequency = max_magnitude_index * sample_rate / len(data)
    return most_frequent_signal, frequency

from scipy.stats import entropy
def compute_ent_features(data):
    hist, bins = np.histogram(data, bins='auto')
    entropyval = entropy(hist, base=2)
    return entropyval