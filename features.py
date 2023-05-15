import numpy as np


def calculate_mean(data):
    return np.mean(data)

def calculate_variance(data):
    return np.var(data)

def calculate_min(data):
    return np.min(data)

def calculate_max(data):
    return np.max(data)

def calculate_median(data):
    return np.median(data)

# Return dominant frequency
def compute_fft_features(data):
    sample_rate = 100
    fft_signal = np.fft.rfft(data)
    fft_magnitudes = np.abs(fft_signal)
    max_magnitude_index = np.argmax(fft_magnitudes)
    most_frequent_signal = fft_signal[max_magnitude_index]
    frequency = max_magnitude_index * sample_rate / len(data)
    return frequency

from scipy.stats import entropy
def compute_ent_features(data):
    hist, bins = np.histogram(data, bins='auto')
    entropyval = entropy(hist, base=2)
    return entropyval

from scipy.signal import find_peaks
def compute_peak_features(window):
    peaks, _ = find_peaks(window)
    return len(peaks)