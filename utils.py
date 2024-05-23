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
    """Returns the indices of the frequency bins corresponding to the given frequency range."""
    n_low = np.ceil(f_low * n_bins / fs)
    n_high = np.floor(f_high * n_bins / fs)
    return int(n_low), int(n_high)

if __name__ == "__main__":
    assert cut_freq_bins(0, 24000, 48000, 4096) == (0, 2048)
    print(cut_freq_bins(1000, 8000, 48000, 4096))