"""
Put the helper functions here.
"""

from scipy.io.wavfile import write
import numpy as np
import logging

def save_as_wav(signal, file_path, fs):
    """Saves the signal as a WAV file."""
    # Normalize signal to 16-bit integer values
    signal_int = np.int16(signal / np.max(np.abs(signal)) * 32767)
    write(file_path, fs, signal_int)
    logging.info(f"Signal has been saved to {file_path}.")
    
def cut_freq_bins(f_low, f_high, fs, n_bins):
    """
    Returns the indices of the frequency bins corresponding to the given frequency range.
    If f_low and f_high are None, the function returns 0 and n_bins//2.
    """
    if f_low is None or f_high is None:
        return 1, n_bins//2
    else:
        n_low = np.ceil(f_low * n_bins / fs)
        n_high = np.floor(f_high * n_bins / fs)
        return 85, 85+648-1#we hardcode the values here, -1 because of exclude last index

def resample_signal(signal, fs, fs_new):
    """Resamples the signal to the new sampling rate."""
    logging.info(f"Resampling signal from {fs} Hz to {fs_new} Hz.")
    n_samples_new = int(len(signal) * fs_new / fs)
    return np.interp(np.linspace(0, 1, n_samples_new), np.linspace(0, 1, len(signal)), signal)

def remove_leading_zeros(binary_data):
    while binary_data.startswith("00000000"):
        binary_data = binary_data[8:]
    return binary_data

def split_by_first_two_occurrences(binary_data, delimiter="0"*16):
    step = 8
    occurrences = []

    # Scan through the string with the step size
    index = 0
    while index <= len(binary_data) - len(delimiter):
        if binary_data[index:index + len(delimiter)] == delimiter:
            occurrences.append(index)
            if len(occurrences) == 2:
                break
        index += step

    if len(occurrences) < 2:
        return [binary_data, '', '']

    # Split the string into three parts
    first_occurrence = occurrences[0]
    second_occurrence = occurrences[1]

    part1 = binary_data[:first_occurrence]
    part2 = binary_data[first_occurrence + len(delimiter):second_occurrence]
    part3 = binary_data[second_occurrence + len(delimiter):]

    return [part1, part2, part3]

def normalize_and_clip(data_in, normalisation_factor):
    # Normalize the value to the normalisation_factor
    # normalized_value = data_in / normalisation_factor 
    # normalisation factor is 2. because 1+j,-1-j is a difference of 2 in real,imag
    # normalisation factor calculated with kmeans is 1.41 which matches
    normalized_value = data_in / 1.41
    
    # Clip the value at 1
    clipped_value = min(normalized_value, 1)
    clipped_value = max(normalized_value,-1)
    
    # clipping improves performance

    return clipped_value
    # return normalized_value

def qpsk_demap_probabilities(constellations, normalisation_factor, bins_used=648, start_bin=85, debug=False):
    """Demap QPSK symbols to binary data."""
    constellation = {
        complex(1, 1): '00',
        complex(-1, 1): '01',
        complex(-1, -1): '11',
        complex(1, -1): '10'
    }

    seed=1

    n_bins=4096

    # Reverse the modulus multiplication to get the original numbers
    np.random.seed(seed)
    constellation_points = np.array([0, 90, 180, 270])
    # constellation_points=np.array([0,0,0,0])
    pseudo_random = np.random.choice(constellation_points, n_bins)

    #TODO enable. this inserts 0 at 0th position
    # Current implementation and malachy's uses replace, not insert
    # pseudo_random = np.insert(pseudo_random, 0, 0) 


    angles_radians = np.deg2rad(pseudo_random)
    complex_exponentials = np.exp(1j * angles_radians)
    # complex_exponentials = np.concatenate(([1 + 1j], complex_exponentials)) #todo uncomment
    if debug:
        print([complex(round(c.real), round(c.imag)) for c in complex_exponentials[start_bin:start_bin + bins_used]])


    constellations = constellations / complex_exponentials[start_bin:start_bin+bins_used]


    binary_probabilities = []
    for index, symbol in enumerate(constellations):
        
        bit0=0.5-0.5*normalize_and_clip(symbol.imag, normalisation_factor)
        bit1=0.5-0.5*normalize_and_clip(symbol.real, normalisation_factor)

        binary_probabilities.append(bit0)

        binary_probabilities.append(bit1)
    return binary_probabilities


if __name__ == "__main__":
    pass