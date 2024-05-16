import librosa
import csv
import numpy as np
file_path = "recordings/mac_recording.m4a"
audio_data, sr = librosa.load(file_path, sr=None)

print(audio_data)

csv_file_path = 'audio_data.csv'

# # Write the audio_data NumPy array to the CSV file
# with open(csv_file_path, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(audio_data.tolist())

# print(f"Data written to {csv_file_path} successfully.")

np.savetxt(csv_file_path, audio_data, delimiter=',', fmt='%1.4f')
