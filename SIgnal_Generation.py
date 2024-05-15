import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram
import sounddevice as sd
from scipy.io.wavfile import write

# Parameters
duration = 5.0  # seconds
fs = 48000  # Sampling rate, Hz
f0 = 20  # Start frequency, Hz
f1 = 20000  # End frequency, Hz
t = np.linspace(0, duration, int(fs * duration))  # Time array

# Generate chirp signal
signal = chirp(t, f0, duration, f1, method='linear')
# Normalize signal to 16-bit integer values
signal_int = np.int16(signal / np.max(np.abs(signal)) * 32767)

# Normalize signal to ensure it is within the proper range for playing
signal_normalized = signal / np.max(np.abs(signal))

# Save the signal as a WAV file
wav_file = 'recordings/linear_chirp.wav'
write(wav_file, fs, signal_int)

# # Play the signal
# sd.play(signal_normalized, fs)
# sd.wait()  # Wait until the sound has finished playing

# Plotting the signal in time domain
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal_normalized)
plt.title('Linear Chirp: Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# # Plotting the spectrogram
# plt.subplot(2, 1, 2)
# frequencies, times, Sxx = spectrogram(signal_normalized, fs)
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.title('Spectrogram of Linear Chirp')
# plt.colorbar(label='Intensity [dB]')
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Example signal (replace this with your actual signal)
# signal_int = np.array([...])  # Your signal data here

# Assuming signal_int is your input signal as a numpy array
# signal_int = np.array([...])  # Your signal data here

# Step 1: Compute the FFT of the signal
fft_values = np.fft.fft(signal_int)

# Step 2: Compute the Power Spectrum
power_spectrum = np.abs(fft_values) ** 2

# Step 3: Generate corresponding frequencies
Fs = 48000  # Sampling frequency in Hz
n = len(signal_int)
frequencies = np.fft.fftfreq(n, d=1/Fs)

# Step 4: Plot the Power Spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:n // 2], power_spectrum[:n // 2])  # Plot only the positive frequencies

plt.xlim(-1000, 21000)
plt.ylim(1e14, 2e14)
plt.title('Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid()
plt.show()

