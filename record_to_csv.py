import sounddevice as sd

sampling_rate=48000
record_duration=sampling_rate*60

recording = sd.rec(record_duration, sampling_rate, channels=1, dtype='int16')
# sd.wait()


def truncate_list(lst):
    # Find index of first value larger than 2000
    index = next((i for i, val in enumerate(lst) if abs(val) > 2000), None)
    if index is not None:
        # Truncate the list from that index onwards
        return lst[:index]
    else:
        # If no value larger than 2000 found, return the original list
        return lst

# recording = recording[:-int(len(recording)*0.2)]
# recording=truncate_list(recording)

for row in recording:
    print(row)



import numpy as np


csv_file_path = 'audio_data2.csv'



np.savetxt(csv_file_path, recording, delimiter=',', fmt='%1.4f')

import matplotlib.pyplot as plt

# Example data
# data = [1, 2, 3, 4, 5]
# Plotting the data
plt.plot(recording)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Simple Plot')
plt.show()