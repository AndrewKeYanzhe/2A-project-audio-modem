"""
This script plot the both transmitted and received signals to achieve visual comparison.
It plots:
1. The signals in the time domain
2. The spectrograms of the signals (Frequency variation over time)
"""


import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def plot_signals(transmitted_signal_path, received_signal_path):
    # Load audio files
    transmitted_signal, sr_trans = librosa.load(transmitted_signal_path, sr=None)
    received_signal, sr_recv = librosa.load(received_signal_path, sr=None)

    # Create a figure and a set of subplots
    plt.figure(figsize=(12, 8))

    # Plot transmitted signal
    plt.subplot(2, 1, 1)
    plt.plot(transmitted_signal, label='Transmitted Signal', color='blue')
    plt.title('Transmitted Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plot received signal
    plt.subplot(2, 1, 2)
    plt.plot(received_signal, label='Received Signal', color='orange')
    plt.title('Received Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    # Display the plot with tight layout to ensure neat alignment of subplots
    plt.tight_layout()
    plt.show()

# File paths (Replace 'path_to_files' with the actual path to your audio files)
transmitted_signal_path = 'recordings/transmitted_chirp_1k_3k_2s.wav'
received_signal_path = 'recordings/chirp_1k_3k.m4a'

# Call the function to plot the signals
plot_signals(transmitted_signal_path, received_signal_path)

def plot_spectrogram(signal, fs, title):
    f, t, Sxx = spectrogram(signal, fs=fs, window='hann', nperseg=1024, noverlap=512, nfft=2048, scaling='density', mode='magnitude')
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.ylim(500, 4000)  # Limit frequency axis to half the sampling rate

def load_and_plot_spectrograms(transmitted_path, received_path):
    # Load audio files
    transmitted_signal, sr_trans = librosa.load(transmitted_path, sr=None)
    received_signal, sr_recv = librosa.load(received_path, sr=None)

    # Ensure both signals are sampled at the same rate, if not, resample
    if sr_trans != sr_recv:
        received_signal = librosa.resample(received_signal, orig_sr=sr_recv, target_sr=sr_trans)
        sr_recv = sr_trans

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot spectrogram of the transmitted signal
    plt.subplot(2, 1, 1)
    plot_spectrogram(transmitted_signal, sr_trans, 'Spectrogram of Transmitted Signal')

    # Plot spectrogram of the received signal
    plt.subplot(2, 1, 2)
    plot_spectrogram(received_signal, sr_recv, 'Spectrogram of Received Signal')

    plt.tight_layout()
    plt.show()

# Call the function to plot the spectrograms
load_and_plot_spectrograms(transmitted_signal_path, received_signal_path)
