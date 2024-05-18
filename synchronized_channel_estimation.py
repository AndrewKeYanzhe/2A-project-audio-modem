import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import librosa

# Define the bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

# Define the sync function to align signals with zero padding
def sync(transmitted_signal, received_signal, f_low, f_high, fs):
    # Bandpass filtering both signals
    
    # Matched filtering to find correlation
    correlation = np.correlate(received_signal, transmitted_signal, mode='full')
    
    # Finding delay
    delay = np.argmax(correlation) - (len(transmitted_signal) - 1)
    
    # Calculate padding
    if delay >= 0:
        padded_transmitted_signal = np.concatenate([np.zeros(delay), transmitted_signal])
        padded_received_signal = np.concatenate([received_signal, np.zeros(delay)])
    else:
        padded_transmitted_signal = np.concatenate([transmitted_signal, np.zeros(-delay)])
        padded_received_signal = np.concatenate([np.zeros(-delay), received_signal])

    # Ensure both signals are of the same length
    max_length = max(len(padded_transmitted_signal), len(padded_received_signal))
    padded_transmitted_signal = np.pad(padded_transmitted_signal, (0, max_length - len(padded_transmitted_signal)), 'constant')
    padded_received_signal = np.pad(padded_received_signal, (0, max_length - len(padded_received_signal)), 'constant')
    
    return padded_transmitted_signal, padded_received_signal, delay,len(transmitted_signal)

# Main function to load, process, and plot signals
def plot_time_domain_signals(transmitted, realligned_received, delay,transmitted_signal_length, f_low, f_high):
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot padded transmitted signal
    plt.subplot(2, 1, 1)
    plt.plot(transmitted, label='Padded Transmitted Signal', color='blue')
    plt.axvline(x=delay, color='red', linestyle='--', label='Start of Original Transmitted Signal')
    plt.axvline(x=delay + transmitted_signal_length, color='red', linestyle='--', label='End of Original Transmitted Signal')
    plt.title('Padded Transmitted Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    # Plot padded realigned received signal
    plt.subplot(2, 1, 2)
    plt.plot(realligned_received, label='Padded Realigned Received Signal', color='green')
    plt.axvline(x=delay, color='red', linestyle='--', label='Start of Original Transmitted Signal')
    plt.axvline(x=delay + transmitted_signal_length, color='red', linestyle='--', label='End of Original Transmitted Signal')
    plt.title('Padded Realigned Received Signal')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def calculate_and_plot_frequency_responses(transmitted, received, fs, f_low, f_high):
    # Compute the length for FFT based on the input signal length
    max_length = len(transmitted)  # Assuming transmitted and received are of the same length

    # Compute FFT
    fft_transmitted = np.fft.rfft(transmitted, n=max_length)
    fft_received = np.fft.rfft(received, n=max_length)

    # Frequency bins
    frequencies = np.fft.rfftfreq(max_length, 1/fs)

    # Filter the frequencies to keep only the range from f_low to f_high
    valid_indices = (frequencies > f_low) & (frequencies < f_high)
    fft_transmitted_filtered = fft_transmitted[valid_indices]
    fft_received_filtered = fft_received[valid_indices]
    frequencies_filtered = frequencies[valid_indices]

    # Compute frequency response of the channel
    epsilon = 1e-10  # To avoid division by zero
    frequency_response = fft_received_filtered / (fft_transmitted_filtered + epsilon)

    # Magnitude of FFT and Frequency Response (in dB)
    magnitude_fft_transmitted = 20 * np.log10(np.abs(fft_transmitted_filtered))
    magnitude_fft_received = 20 * np.log10(np.abs(fft_received_filtered))
    magnitude_response = 20 * np.log10(np.abs(frequency_response))

    # Phase of the frequency response (in degrees)
    phase_response = np.angle(frequency_response, deg=True)
    plt.figure(figsize=(14, 5))
    plt.plot(frequencies_filtered, phase_response)
    plt.title('Phase Response of the Channel')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (Degrees)')
    plt.show()

    # Plot magnitude spectra of transmitted and received signals
    plt.figure(figsize=(14, 5))
    plt.plot(frequencies_filtered, magnitude_fft_transmitted, label='Transmitted Signal')
    plt.plot(frequencies_filtered, magnitude_fft_received, label='Received Signal', alpha=0.7)
    plt.title('Magnitude Spectra of Transmitted and Received Signals')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.show()

    # Plot the transfer function (channel frequency response)
    plt.figure(figsize=(14, 5))
    plt.plot(frequencies_filtered, magnitude_response)
    plt.title('Transfer Function of the Channel')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.show()

    # Optionally compute the IFFT of the frequency response to get the FIR filter (impulse response)
    fir_filter = np.fft.irfft(frequency_response, n=max_length)

    # Plot the FIR filter (impulse response of the channel)
    plt.figure(figsize=(14, 5))
    plt.plot(fir_filter)
    plt.title('Impulse Response of the Channel (FIR Filter)')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.show()


if __name__ == '__main__':
    
    # Parameters
    f_low = 20
    f_high = 5000
    fs=48000
    
    # File paths
    transmitted_signal_path = 'recordings/transmitted_chirp_1k_3k_2s.wav'
    received_signal_path = 'recordings/chirp_1k_3k.m4a'

    # Load audio files
    transmitted_signal, sr_trans = librosa.load(transmitted_signal_path, sr=None)
    received_signal, sr_recv = librosa.load(received_signal_path, sr=None)

    # Ensure same sampling rate
    if sr_trans != sr_recv:
        received_signal = librosa.resample(received_signal, orig_sr=sr_recv, target_sr=sr_trans)

    # Synchronize signals
    transmitted, realligned_received, delay,transmitted_signal_length = sync(transmitted_signal, received_signal, f_low, f_high, sr_trans)

    # Plot time domain signals
    plot_time_domain_signals(transmitted, realligned_received,delay, transmitted_signal_length, f_low, f_high)

    #Plot frequency responses
    # calculate_and_plot_frequency_responses(transmitted, realligned_received,fs, f_low, f_high)
