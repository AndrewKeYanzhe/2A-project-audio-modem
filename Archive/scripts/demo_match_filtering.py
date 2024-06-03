"""
This script implements the matched filtering technique to estimate the delay between two signals.
N.B. You correlates the received signal with the transmitted signal, not the time-reversed version of the transmitted signal.
"""


import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

def load_and_process_signals(transmitted_path, received_path, f_low=1000, f_high=3000):
    # Load audio files
    transmitted_signal, sr_trans = librosa.load(transmitted_path, sr=None)
    received_signal, sr_recv = librosa.load(received_path, sr=None)
    
    # Ensure same sampling rate
    if sr_trans != sr_recv:
        # Resample received signal to match sr_trans
        received_signal = librosa.resample(received_signal, orig_sr=sr_recv, target_sr=sr_trans)
        sr_recv = sr_trans
        print("Resampled received signal to match transmitted signal's sampling rate.")

    # Bandpass filtering both signals
    transmitted_signal = bandpass_filter(transmitted_signal, f_low, f_high, sr_trans)
    received_signal = bandpass_filter(received_signal, f_low, f_high, sr_trans)
    
    # Matched filtering
    correlation = np.correlate(received_signal, transmitted_signal, mode='full')
    
    # Finding delay
    delay = np.argmax(correlation) - (len(transmitted_signal) - 1)

    # Plotting the received signal and correlation
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(received_signal, label='Filtered Received Signal', color='blue')
    plt.axvline(x=delay, color='red', linestyle='--', label='Estimated Sync Point')
    plt.title('Filtered Received Signal with Estimated Delay')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(correlation, label='Correlation Output', color='green')
    plt.axvline(x=np.argmax(correlation), color='red', linestyle='--', label='Peak at Delay')
    plt.title('Matched Filter Output (Correlation)')
    plt.xlabel('Lag Index')
    plt.ylabel('Correlation Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Estimated delay (sync point) in samples:", delay)

# File paths
transmitted_signal_path = 'recordings/transmitted_chirp_1k_3k_2s.wav'
received_signal_path = 'recordings/chirp_1k_3k.m4a'  # Using the same signal for testing

load_and_process_signals(transmitted_signal_path, received_signal_path)
