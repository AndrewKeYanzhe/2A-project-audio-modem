"""
Generate a linear chirp signal with a circular prefix and silence.
The signal is saved as a WAV file and plotted in the time domain.
Don't forget to replace the path to the saved WAV file with your own path.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd
from scipy.signal import chirp

# Parameters
fs = 48000  # Sampling rate, Hz
t_chirp = 5.0  # Duration of the chirp, seconds
t_silence = 1.0  # Duration of silence, seconds
f_low = 20  # Start frequency, Hz
f_high = 8000  # End frequency, Hz
prefix_duration = 1.0  # Duration of circular prefix, seconds

# Calculate total duration
t_total = 2 * t_silence +prefix_duration + t_chirp

# Generate linear chirp signal
t_chirp_only = np.linspace(0, t_chirp, int(fs * t_chirp), endpoint=False)
chirp_signal = chirp(t_chirp_only, f0=f_low, f1=f_high, t1=t_chirp, method='linear')

# Create circular prefix (last 1 second of chirp)
prefix = chirp_signal[-int(fs * prefix_duration):]

# Generate silence
silence = np.zeros(int(fs * t_silence))

# Combine all parts
full_signal = np.concatenate([silence, prefix, chirp_signal, silence])
# Ensure the time array matches the signal length
t = np.linspace(0, t_total, len(full_signal), endpoint=False)

# Normalize signal to 16-bit integer values
signal_int = np.int16(full_signal / np.max(np.abs(full_signal)) * 32767)

# Normalize signal to ensure it is within the proper range for playing
signal_normalized = full_signal / np.max(np.abs(full_signal))

# # Play the signal
# sd.play(signal_normalized, fs)
# sd.wait()  # Wait until the sound has finished playing

# Save the signal as a WAV file
wav_file = 'recordings/transmitted_linear_chirp_with_prefix_and_silence.wav'
write(wav_file, fs, signal_int)

# Plotting the signal in time domain
plt.figure(figsize=(12, 6))
plt.plot(t, signal_normalized)
plt.title('Linear Chirp with Circular Prefix and Silence: Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()
