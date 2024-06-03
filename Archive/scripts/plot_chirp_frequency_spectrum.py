import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
from scipy.fft import fft, fftfreq

# Parameters
fs = 48000  # Sampling frequency
t = np.linspace(0, 5, 5 * fs)  # Time array, 5 seconds

# Generate chirp signal
f0 = 1000  # Start frequency of the chirp
f1 = 8000  # End frequency of the chirp
chirp_signal = chirp(t, f0=f0, f1=f1, t1=5, method='linear')

# Perform FFT
N = len(chirp_signal)
yf = fft(chirp_signal)
xf = fftfreq(N, 1 / fs)

# Only take the positive half of the frequencies
xf = xf[:N // 2]
yf = np.abs(yf[:N // 2])

# Plot the amplitude spectrum
plt.figure(figsize=(3, 3))
plt.plot(xf, yf)
plt.title('Amplitude Spectrum of Chirp\n1-8kHz')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim(0, 9000)
plt.tight_layout()
plt.grid()
plt.show()
