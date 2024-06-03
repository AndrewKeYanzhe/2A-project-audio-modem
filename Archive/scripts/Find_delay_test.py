import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import librosa
import logging
import time
import scipy.signal
from utils import save_as_wav
from channel_estimator import AnalogueSignalProcessor  # Assuming this is the class you are using

# Known sequence
known_sequence = np.array([1, 0, 1, 0, 1, 0, 1, 0])

# Parameters
delay_samples = 1000  # Delay in samples
sampling_rate = 48000
noise_levels = np.linspace(0, 1, 10)  # Noise levels to test
num_tests = 100  # Number of tests per noise level

# Containers for results
errors = []

# Run tests for each noise level
for noise_level in noise_levels:
    error_sum = 0
    
    for _ in range(num_tests):
        # Generate noisy sequence
        noise = np.random.normal(0, noise_level, len(known_sequence))
        noisy_sequence = known_sequence + noise
        
        # Add delay to the noisy sequence
        delayed_sequence = np.concatenate((np.random.normal(0, noise_level, delay_samples), noisy_sequence))
        
        # Save the known and received sequences as audio files
        transmitted_signal_path = './recordings/known_sequence.wav'
        received_signal_path = './recordings/received_sequence.wav'
        save_as_wav(known_sequence, transmitted_signal_path, sampling_rate)
        save_as_wav(delayed_sequence, received_signal_path, sampling_rate)
        
        # Instantiate the AnalogueSignalProcessor object
        processor = AnalogueSignalProcessor(transmitted_signal_path, received_signal_path)
        
        # Load the audio files
        processor.load_audio_files()
        
        # Find the delay using the known sequence
        estimated_delay = processor.find_delay(plot=False)
        
        # Calculate the error
        error = np.abs(estimated_delay - delay_samples)
        error_sum += error
    
    # Average error for the current noise level
    average_error = error_sum / num_tests
    errors.append(average_error)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(noise_levels, errors, marker='o', linestyle='-', color='b')
plt.xlabel('Noise Level (Standard Deviation)')
plt.ylabel('Average Delay Estimation Error (Samples)')
plt.title('Effect of Noise Level on Delay Estimation Error')
plt.grid(True)
plt.show()
