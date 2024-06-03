import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

# Define Golay Complementary Sequences (example sequences)
GolayA = np.array([1, -1, 1, 1])
GolayB = np.array([1, 1, -1, 1])

# Upsample the Golay sequences to increase duration
samples_per_chip = 400  # Define the duration of each bit
upsampled_GolayA = np.repeat(GolayA, samples_per_chip)
upsampled_GolayB = np.repeat(GolayB, samples_per_chip)

# Convert to int16 for WAV file compatibility
upsampled_GolayA_int16 = np.int16(upsampled_GolayA * 32767)
upsampled_GolayB_int16 = np.int16(upsampled_GolayB * 32767)

# Sampling parameters
fs = 48000  # Sampling rate, Hz

# Save the signals as WAV files
write('recordings/GolayA.wav', fs, upsampled_GolayA_int16)
write('recordings/GolayB.wav', fs, upsampled_GolayB_int16)

# Normalize for playback to avoid clipping
normalized_GolayA = upsampled_GolayA_int16 / 32768.0
normalized_GolayB = upsampled_GolayB_int16 / 32768.0

# Play the Golay sequences
sd.play(normalized_GolayA, fs)
sd.wait()  # Wait until GolayA has finished playing
sd.play(normalized_GolayB, fs)
sd.wait()  # Wait until GolayB has finished playing

# Plot the Golay sequences
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(normalized_GolayA)
plt.title('Golay Sequence A: Time Domain')
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(normalized_GolayB)
plt.title('Golay Sequence B: Time Domain')
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
