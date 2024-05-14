

"""
source '/Users/andrewke/Library/CloudStorage/OneDrive-UniversityofCambridge/2A Audio modem/2A Project - Audio modem Repo/audio_modem/bin/activate'
python3 '/Users/andrewke/Library/CloudStorage/OneDrive-UniversityofCambridge/2A Audio modem/2A Project - Audio modem
 Repo/analyse_audio.py'
"""

import librosa
import matplotlib.pyplot as plt

# Path to your .m4a file
file_path = "recordings/chirp_lib.m4a"
file_path2 = "recordings/ZOOM0009.WAV"


# Load the audio file
audio_data, sr = librosa.load(file_path)
audio_data2, sr2 = librosa.load(file_path2)


# Plot the waveform
plt.figure(figsize=(12, 4))
plt.plot(audio_data, label=file_path.split('/')[-1])
# plt.plot(audio_data2, label=file_path2.split('/')[-1])
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.title("Waveforms of {} and {}".format(file_path.split('/')[-1], file_path2.split('/')[-1]))
plt.legend()  # Show legend with file names
plt.show()
