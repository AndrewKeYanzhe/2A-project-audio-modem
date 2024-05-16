import sounddevice as sd


numsamples=48000

recording = sd.rec(numsamples, 48000, channels=1, dtype='int16')
# sd.wait()

recording = recording[:-int(len(recording)*0.2)]

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