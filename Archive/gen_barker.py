import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import sounddevice as sd

# Barker Code of length 13
barker_code = np.array([1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1])

# Upsample the Barker code to increase duration
# Each element of the Barker code is repeated more times to lengthen the signal
samples_per_chip = 400  # Increasing from 100 to 400 for longer duration
upsampled_barker_code = np.repeat(barker_code, samples_per_chip)

# Convert to int16, scaling to the maximum range of int16 for WAV file compatibility
upsampled_barker_code_int16 = np.int16(upsampled_barker_code * 32767)

# Sampling parameters
fs = 48000  # Sampling rate, Hz

# Save the signal as a WAV file
wav_file = 'recordings/barker_code_long.wav'
write(wav_file, fs, upsampled_barker_code_int16)

# Normalize for playback to avoid clipping
upsampled_barker_code_normalized = upsampled_barker_code_int16 / 32768.0

# Play the signal
sd.play(upsampled_barker_code_normalized, fs)
sd.wait()  # Wait until the sound has finished playing

# Plot the Barker code signal
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(upsampled_barker_code) / fs, len(upsampled_barker_code)), upsampled_barker_code)
plt.title('Barker Code: Time Domain')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()
