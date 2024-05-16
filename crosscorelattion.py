import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, correlate
import sounddevice as sd
file_path = "recordings/chirp_lib.m4a"
file_path2 = "recordings/linear_chirp.wav"

fs = 48000  # Sampling frequency
# Load the audio file
received_signal, sr = librosa.load(file_path)
chirp_signal, sr2 = librosa.load(file_path2)


# Perform cross-correlation
correlation = correlate(received_signal, chirp_signal, mode='full')
lags = np.arange(-len(chirp_signal) + 1, len(received_signal))

# Find the location of the peak
peak_index = np.argmax(np.abs(correlation))
impulse_response_time = lags[peak_index] / fs

# Extract the impulse response
impulse_response = correlation[peak_index:]

from scipy.fft import fft, fftfreq

# Perform FFT on the impulse response
frequency_response = 20 * np.log10(np.abs(fft(impulse_response)))

# Frequency bins
N = len(impulse_response)
freq_bins = fftfreq(N, 1/fs)

# Plotting the frequency response
import matplotlib.pyplot as plt

plt.plot(freq_bins[:N // 2], np.abs(frequency_response)[:N // 2])
plt.title('Frequency Response of the Room')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.show()