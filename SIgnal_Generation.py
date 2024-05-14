import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram
import sounddevice as sd

# Parameters
duration = 5.0  # seconds
fs = 44100  # Sampling rate, Hz
f0 = 20  # Start frequency, Hz
f1 = 20000  # End frequency, Hz
t = np.linspace(0, duration, int(fs * duration))  # Time array

# Generate chirp signal
signal = chirp(t, f0, duration, f1, method='linear')

# Normalize signal to ensure it is within the proper range for playing
signal_normalized = signal / np.max(np.abs(signal))

# Play the signal
sd.play(signal_normalized, fs)
sd.wait()  # Wait until the sound has finished playing

# Plotting the signal in time domain
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal_normalized)
plt.title('Linear Chirp: Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plotting the spectrogram
plt.subplot(2, 1, 2)
frequencies, times, Sxx = spectrogram(signal_normalized, fs)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Linear Chirp')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()

plt.show()
