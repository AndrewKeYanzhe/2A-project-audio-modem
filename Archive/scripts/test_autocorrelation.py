import librosa
import numpy as np
import matplotlib.pyplot as plt

def autocorrelation_test(signal_path):
    # Load the audio file
    signal, sr = librosa.load(signal_path, sr=None)

    # Perform autocorrelation
    correlation = np.correlate(signal, signal, mode='full')
    lags = np.arange(-len(signal) + 1, len(signal))

    # Finding the maximum correlation index (should be at zero lag)
    max_corr_index = np.argmax(correlation)
    center_index = len(signal) - 1  # This is where zero lag is expected
    print("Signal Length:", len(signal))
    print("Max correlation index:", max_corr_index)
    print("Expected center index:", center_index)

    # Plotting the signal and correlation
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Signal')
    plt.title('Original Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(lags, correlation, label='Autocorrelation')
    plt.axvline(x=0, color='red', linestyle='--', label='Zero Lag (Expected Peak)')
    plt.title('Autocorrelation of Signal')
    plt.xlabel('Lag')
    plt.ylabel('Correlation Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

# File path to your audio file
signal_path = 'recordings/transmitted_chirp_1k_3k_2s.wav'  # Update this path to your audio file's location
autocorrelation_test(signal_path)
