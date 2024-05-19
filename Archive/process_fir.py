import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the FIR filter
fir_filter = pd.read_csv('fir_filter.csv').values.flatten()

def circular_shift(fir_filter, shift):
    # Compute the circular shift
    shifted_filter = np.roll(fir_filter, shift)
    return shifted_filter

# Plot function for FIR filter
def plot_fir_filter(fir_filter, title):
    plt.figure()
    plt.plot(fir_filter)
    plt.title(title)
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Example of shifting the FIR filter
shift_amount = 100  # Shift to the right by 5 samples
shifted_fir_filter = circular_shift(fir_filter, shift_amount)

# Plot the original and shifted FIR filter
plot_fir_filter(fir_filter, 'Original FIR Filter')
plot_fir_filter(shifted_fir_filter, f'Shifted FIR Filter by {shift_amount} samples')
