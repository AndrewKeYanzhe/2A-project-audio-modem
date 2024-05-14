import librosa
import numpy as np
import matplotlib.pyplot as plt

# Load audio files
transmitted_signal, sr_trans = librosa.load('recordings/linear_chirp.wav', sr=None)
received_signal, sr_recv = librosa.load('recordings/chirp_lib.m4a', sr=None)

# Ensure both signals are zero-padded to the same length
max_length = max(len(transmitted_signal), len(received_signal))
transmitted_signal = np.pad(transmitted_signal, (0, max_length - len(transmitted_signal)), mode='constant')
received_signal = np.pad(received_signal, (0, max_length - len(received_signal)), mode='constant')

# Compute FFT
fft_transmitted = np.fft.rfft(transmitted_signal)
fft_received = np.fft.rfft(received_signal)

# Frequency bins
frequencies = np.fft.rfftfreq(max_length, 1/sr_trans)

# Filter the frequencies to keep only the range from 20 Hz to 20 kHz
valid_indices = (frequencies > 20) & (frequencies < 20000)
fft_transmitted_filtered = fft_transmitted[valid_indices]
fft_received_filtered = fft_received[valid_indices]
frequencies_filtered = frequencies[valid_indices]

# Compute frequency response of the channel
epsilon = 1e-10  # To avoid division by zero
frequency_response = fft_received_filtered / (fft_transmitted_filtered + epsilon)

# Magnitude of FFT and Frequency Response (in dB)
magnitude_fft_transmitted = 20 * np.log10(np.abs(fft_transmitted_filtered))
magnitude_fft_received = 20 * np.log10(np.abs(fft_received_filtered))
magnitude_response = 20 * np.log10(np.abs(frequency_response))

# Plot magnitude spectra of transmitted and received signals
plt.figure(figsize=(14, 10))
plt.subplot(3, 1, 1)
plt.plot(frequencies_filtered, magnitude_fft_transmitted, label='Transmitted Signal')
plt.plot(frequencies_filtered, magnitude_fft_received, label='Received Signal', alpha=0.7)
plt.title('Magnitude Spectra of Transmitted and Received Signals')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()

# Plot the transfer function (channel frequency response)
plt.subplot(3, 1, 2)
plt.plot(frequencies_filtered, magnitude_response)
plt.title('Transfer Function of the Channel')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')

plt.tight_layout()
plt.show()
