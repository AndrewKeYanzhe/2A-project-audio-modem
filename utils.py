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