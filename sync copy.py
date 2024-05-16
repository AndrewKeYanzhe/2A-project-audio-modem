import numpy as np

def generate_zadoff_chu_sequence(u, N):
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)

u = 25  # Root index
N = 128  # Sequence length
preamble = generate_zadoff_chu_sequence(u, N)
print(preamble.shape)
# say we transmit the binary data for 'A'
data = np.array([1, 0, 1, 0, 0, 0, 0, 1])  # Binary for 'A' -> 01000001

# Append preamble to data
transmitted_signal = np.concatenate([preamble, data])
print(transmitted_signal)
# Multipath channel
multipath_channel = np.array([1, 0.5, 0.3])
received_signal = np.convolve(transmitted_signal, multipath_channel, mode='full')

# Add noise
noise = (np.random.randn(len(received_signal)) + 1j * np.random.randn(len(received_signal))) * 0.1
received_signal += noise
from scipy.signal import correlate
import matplotlib.pyplot as plt

def correlate_and_find_peaks(received_signal, preamble, T_H, T_L, search_window):
    correlation = np.abs(correlate(received_signal, preamble))
    peaks = []
    for i in range(len(correlation)):
        if correlation[i] > T_H:
            peaks.append(i)
        elif correlation[i] > T_L and peaks and i - peaks[-1] <= search_window:
            peaks.append(i)
    return peaks, correlation

# Parameters for synchronization
T_H = 0.8 * np.max(np.abs(received_signal))
T_L = 0.4 * np.max(np.abs(received_signal))
search_window = 10

# Perform synchronization
peaks, correlation = correlate_and_find_peaks(received_signal, preamble, T_H, T_L, search_window)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(np.abs(received_signal), label='Received Signal')
plt.plot(correlation, label='Correlation')
plt.scatter(peaks, correlation[peaks], color='r', label='Detected Peaks')
plt.axhline(T_H, color='g', linestyle='--', label='High Threshold')
plt.axhline(T_L, color='b', linestyle='--', label='Low Threshold')
plt.legend()
plt.title('Synchronization Using Fast Timing Search Window and Double Threshold')
plt.show()

print("Detected peaks at positions:", peaks)
