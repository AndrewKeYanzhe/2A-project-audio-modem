""" 
the function audio_to_csv(file_path) creates a csv file from an audio file
input is path to audio
output is <audio_filename>.csv
"""


import librosa
import csv
import numpy as np
import os
# file_path = "recordings/mac_recording.m4a"


def audio_to_csv(file_path):
    audio_data, sr = librosa.load(file_path, sr=None)

    # print(audio_data)

    # csv_file_path = 'audio_data.csv'
    
    base = os.path.splitext(file_path)[0]
    csv_file_path = base + '.csv'
    

    # # Write the audio_data NumPy array to the CSV file
    # with open(csv_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(audio_data.tolist())

    # print(f"Data written to {csv_file_path} successfully.")

    np.savetxt(csv_file_path, audio_data, delimiter=',', fmt='%1.4f')
    print("wrote to "+csv_file_path)

# audio_to_csv('delme_rec_unlimited_hzuuzg8y.wav')