import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import spectrogram

# Parameters
duration = 5.0  # seconds
fs = 48000  # Sampling rate, Hz
f0 = 1000  # Start frequency, Hz
f1 = 3000  # End frequency, Hz
n_steps = 10  # Number of frequency steps

# Create time array
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate stepped chirp signal
step_duration = duration / n_steps
frequencies = np.linspace(f0, f1, n_steps)  # Frequencies for each step
signal = np.zeros_like(t)

# Generate a signal for each frequency step
for i in range(n_steps):
    start_idx = int(i * step_duration * fs)
    end_idx = int((i + 1) * step_duration * fs)
    # Each step maintains a constant frequency
    signal[start_idx:end_idx] = np.sin(2 * np.pi * frequencies[i] * t[start_idx:end_idx])

# Normalize signal to 16-bit integer values and for playing
signal_normalized = signal / np.max(np.abs(signal))
signal_int = np.int16(signal_normalized * 32767)

# Save the signal as a WAV file
wav_file = 'recordings/stepped_chirp.wav'
write(wav_file, fs, signal_int)

# Optionally play the signal
sd.play(signal_normalized, fs)
sd.wait()  # Wait until the sound has finished playing

# Plotting the signal in time domain
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, signal_normalized)
plt.title('Stepped Chirp: Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plotting the spectrogram
plt.subplot(2, 1, 2)
frequencies, times, Sxx = spectrogram(signal_normalized, fs, nperseg=1024, noverlap=512)
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.title('Spectrogram of Stepped Chirp')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.show()
