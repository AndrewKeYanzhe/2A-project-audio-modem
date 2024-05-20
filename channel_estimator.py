"""
Class AnalogueSignalProcessor:
This class is capable of:
1. Loading audio files.
2. Finding the delay between two signals using matched filtering. (And plotting the signals with the delay line)
3. Computing the frequency response of the channel. (And optionally plotting it)
4. Computing the FIR filter from the frequency response. (And optionally plotting it)

Attributes:
1. transmitted_signal_path (str): Path to the transmitted signal file.
2. received_signal_path (str): Path to the received signal file.
3. f_low (float): Lower frequency bound for the bandpass filter.
4. f_high (float): Upper frequency bound for the bandpass filter.
5. fs (float): Sampling frequency of the audio signals.
6. transmitted_signal (np.ndarray): Array holding the transmitted signal.
7. received_signal (np.ndarray): Array holding the received signal.
8. delay (int): Computed delay between the transmitted and received signals in samples.
9. frequency_response (np.ndarray): Computed frequency response of the channel.
10. frequencies (np.ndarray): Frequency bins associated with the frequency response.
11. fir_filter (np.ndarray): FIR filter derived from the frequency response.

Methods:
1. load_audio_files: Loads the audio files and sets the fs attribute.
2. find_delay: Calculates the delay between the transmitted and received signals.
3. plot_time_domain_signals: Plots both signals with time on the x-axis.
4. get_frequency_response: Calculates and optionally plots the frequency response of the channel.
5. plot_frequency_response: Helper method to plot magnitude and phase responses.
6. get_FIR: Computes and optionally plots and saves the FIR filter.

Usage:
1. Initialize the processor with file paths and frequency bounds.
2. Load audio files.
3. Compute and visualize the delay and frequency response.
4. Extract and utilize the FIR filter for further signal processing tasks.
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import librosa
import pandas as pd
import logging

# Define the bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y

class AnalogueSignalProcessor:
    
    def __init__(self, transmitted_signal_path, received_signal_path, f_low=None, f_high=None):
        self.trans_path = transmitted_signal_path
        self.recv_path = received_signal_path
        self.f_low = f_low # f_low of the chirp signal
        self.f_high = f_high # f_high of the chirp signal
        self.fs = None # Sampling rate - will be set after loading the signals
        self.trans = None # Transmitted signal array
        self.recv = None # Received signal array
        self.delay = None # Delay between the signals
        self.frequency_response = None # Frequency response of the channel
        self.frequencies = None # Frequency bins
        self.fir_filter = None # FIR filter
    def load_audio_files(self):
        """
        Load the transmitted and received audio files, warning if the sampling rates are different.
        """
        self.trans, fs_trans = librosa.load(self.trans_path, sr=None)
        self.recv, fs_recv = librosa.load(self.recv_path, sr=None)
        if fs_trans != fs_recv:
            logging.warning('Sampling rates of the transmitted and received signals are different.' )
        else:
            self.fs = fs_recv
            logging.info('fs = {}'.format(self.fs))
        logging.info('Signals loaded successfully.')
        return
    def find_delay(self, start_time=None, end_time=None, plot=False):
        """
        Find the delay between the transmitted and received signals.
        """
        # Check if the signals are loaded
        if self.trans is None or self.recv is None:
            logging.error('Signals are not loaded. Load the signals first.')
            return
        
        # Truncate the received signal if start_time and end_time are provided
        if start_time and end_time:
            truncated_recv = self.recv[int(start_time*self.fs):int(end_time*self.fs)]
        else:
            truncated_recv = self.recv
        
        # Bandpass filtering both signals
        if self.f_low and self.f_high: # if f_low and f_high are provided
            logging.info('Bandpass filtering the signals (for matched filtering only), f_low = {}, f_high = {}'.format(self.f_low, self.f_high))
            filtered_trans = bandpass_filter(self.trans, self.f_low, self.f_high, self.fs)
            filtered_recv = bandpass_filter(truncated_recv, self.f_low, self.f_high, self.fs)
        else:
            filtered_trans = self.trans
            filtered_recv = truncated_recv
        
        # Matched filtering to find correlation
        logging.info('Finding delay... (If this takes too long, you can specify the range to search for delay by providing start_time and end_time to truncate the received signal)')
        correlation = np.correlate(filtered_recv, filtered_trans, mode='full')
        # Finding delay
        relative_delay = np.argmax(correlation) - (len(self.trans) - 1)
        self.delay = relative_delay + start_time*self.fs if start_time else relative_delay
        logging.info('Delay = {}'.format(self.delay))
        
        # Optionally plot the transmitted signal, realligned received signal, and the delay line
        if plot:
            self.plot_time_domain_signals(
                transmitted=self.trans,
                received=truncated_recv,
                delay=relative_delay # Note: The delay relative to the truncated received signal
            )
        return self.delay
    def plot_time_domain_signals(self, transmitted, received, delay=None):
        """
        Plot the time domain signals of the transmitted and received signals with time as the x-axis.
        """
        # Calculate time vectors for transmitted and received signals
        time_vector_trans = np.arange(len(transmitted)) / self.fs
        time_vector_recv = np.arange(len(received)) / self.fs

        plt.figure(figsize=(8, 6))

        # Plotting the transmitted signal
        plt.subplot(2, 1, 1)
        plt.plot(time_vector_trans, transmitted, label='Transmitted Signal', color='blue')
        plt.title('Transmitted Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()

        # Plotting the received signal
        plt.subplot(2, 1, 2)
        plt.plot(time_vector_recv, received, label='Received Signal', color='green')
        if delay is not None:
            delay_time = delay / self.fs  # Convert delay in samples to time in seconds
            plt.axvline(x=delay_time, color='red', linestyle='--', label='Estimated Start of Received Signal')
        plt.title('Received Signal')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()
    def get_frequency_response(self, chirp_start_time, chirp_end_time, plot=False):
        """
        Compute the frequency response of the channel.
        The chirp start and end times are used to truncate the transmitted and received signals.
        Returns the frequency bins and the frequency response (zero padded outside the chirp frequency range).
        """
        # Check if the signals are loaded
        if self.trans is None or self.recv is None:
            logging.error('Signals are not loaded. Load the signals first.')
            return
        # Check if the delay is found
        if self.delay is None:
            logging.error('Delay is not found. Find the delay first.')
            return
        
        if self.f_low and self.f_high:
            f_low = self.f_low
            f_high = self.f_high
        else:
            logging.warning('f_low and f_high are not provided. Using the default values.')
            f_low = 1000
            f_high = 10000
        
        # truncate both signals
        start_index = int(chirp_start_time*self.fs)
        end_index = int(chirp_end_time*self.fs)
        truncated_trans = self.trans[start_index:end_index]
        truncated_recv = self.recv[self.delay+start_index:self.delay+end_index]
        
        # Temporary code to calculate the direct FIR filter
        # self.direct_FIR(truncated_trans, truncated_recv, plot=True, file_path='FIR_filters/direct_5.56pm.csv', truncate=True)
        
        # Compute the length for FFT based on the input signal length
        max_length = len(truncated_trans)  # Assuming transmitted and received are of the same length

        logging.info('Computing frequency response...')
        # Compute FFT
        fft_transmitted = np.fft.rfft(truncated_trans, n=max_length)
        fft_received = np.fft.rfft(truncated_recv, n=max_length)
        
        # Frequency bins
        frequencies = np.fft.rfftfreq(max_length, 1/self.fs)
        self.frequencies = frequencies
        
        # Filter the frequencies to keep only the range from f_low to f_high, and zero out the rest
        valid_indices = (frequencies > f_low) & (frequencies < f_high)
        fft_received[~valid_indices] = 0
        fft_transmitted[~valid_indices] = 0

        # Compute frequency response of the channel
        epsilon = 1e-10  # To avoid division by zero
        self.frequency_response = fft_received / (fft_transmitted + epsilon)
        if plot: 
            magnitude_fft_transmitted = 20 * np.log10(np.abs(fft_transmitted)+epsilon)
            magnitude_fft_received = 20 * np.log10(np.abs(fft_received)+epsilon)
            magnitude_response = 20 * np.log10(np.abs(self.frequency_response)+epsilon) 
            phase_response = np.angle(self.frequency_response, deg=True)
            self.plot_frequency_response(magnitude_fft_transmitted, magnitude_fft_received, magnitude_response, phase_response)
        
        return self.frequencies, self.frequency_response
    def plot_frequency_response(self, mag_trans, mag_recv, mag, phase):
        """
        Plot the magnitude and phase response of the channel.
        """
        # Magnitude of FFT and Frequency Response (in dB)
            
        frequencies = self.frequencies
        # Plot magnitude spectra of transmitted and received signals
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies, mag_trans, label='Transmitted Signal')
        plt.plot(frequencies, mag_recv, label='Received Signal', alpha=0.7)
        plt.title('Magnitude Spectra of Transmitted and Received Signals')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.show()

        # Plot the transfer function (channel frequency response)
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies, mag)
        plt.title('Transfer Function of the Channel')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.show()
        
        # Plot the phase response
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies, phase)
        plt.title('Phase Response of the Channel')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (Degrees)')
        plt.show()
    def get_FIR(self, plot=False, file_path=None, truncate=False):
        """
        Compute the inverse FFT of the frequency response to get the FIR filter.
        """
        
        if self.frequency_response is None or self.frequencies is None:
            logging.error('Frequency response is not computed. Compute the frequency response first.')
            return
        logging.info('Computing FIR filter...')
        # Compute the inverse FFT
        self.fir_filter = np.fft.irfft(self.frequency_response)
        
        if truncate:
            logging.info('Truncating the FIR filter to contain 90% of the energy.')
            # Truncate the FIR filter to contain 90% of the energy
            energy = np.sum(self.fir_filter**2)
            cumulative_energy = 0
            for i in range(len(self.fir_filter)):
                cumulative_energy += self.fir_filter[i]**2
                if cumulative_energy / energy >= 0.9:
                    break
            self.fir_filter = self.fir_filter[:i]
            logging.info('Truncated FIR filter length = {}'.format(i))
        
        if plot:
            t = np.arange(len(self.fir_filter)) / self.fs
            plt.figure(figsize=(8, 6))
            plt.plot(t, self.fir_filter)
            plt.title('Impulse Response of the Channel (FIR Filter) - Sampled at {} Hz'.format(self.fs))
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.show()
        if file_path:
            logging.info('Saving FIR filter to {}'.format(file_path))
            pd.DataFrame(self.fir_filter).to_csv(file_path, index=False)
            
        # debug messages
        # write the following prints into logging
        logging.info('FIR filter length:', len(self.fir_filter))
        if self.fir_filter.shape[0] > self.fs*1.0:
            logging.warning('Invalid FIR filter length > 1 second')
        return self.fir_filter
    def direct_FIR(self, chirp_trans, chirp_recv, plot=False, file_path=None, truncate=False):
        """
        Compute the FIR by cross-correlating the chirp section of the transmitted and received signals.
        """
        logging.info('Computing direct FIR filter...')
        # Calculate the correlation function between the chirp sections of the transmitted and received signals
        correlation = np.correlate(chirp_recv, chirp_trans, mode='full')
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(correlation)
            plt.title('Direct FIR Filter Calculation')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.show()
            
        if truncate:
            logging.info('Truncating the direc FIR filter to contain 90% of the energy.')
            # Truncate the FIR filter to contain 90% of the energy
            energy = np.sum(correlation**2)
            cumulative_energy = 0
            for i in range(len(correlation)):
                cumulative_energy += correlation[i]**2
                if cumulative_energy / energy >= 0.9:
                    break
            correlation = correlation[:i]
            logging.info('Truncated direct FIR filter length = {}'.format(i))
        
        if file_path:
            logging.info('Saving FIR filter to {}'.format(file_path))
            pd.DataFrame(correlation).to_csv(file_path, index=False)
        
        return correlation
    
    
if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parameters
    f_low = 20
    f_high = 8000
    fs=48000
    chirp_start_time = 2.0
    chirp_end_time = 7.0
    # File paths
    transmitted_signal_path = 'recordings/transmitted_5.56pm.wav'
    received_signal_path = 'recordings/received_5.56pm.m4a'
    
    # # Test Parameters
    # f_low = 1000
    # f_high = 3000
    # fs=48000
    # chirp_start_time = 0.0
    # chirp_end_time = 2.0
    # transmitted_signal_path = 'recordings/transmitted_chirp_1k_3k_2s.wav'
    # received_signal_path = 'recordings/chirp_1k_3k.m4a'
    
    signal_processor = AnalogueSignalProcessor(transmitted_signal_path, received_signal_path, f_low, f_high)
    signal_processor.load_audio_files()
    # For delay calcualtion: you will need to add the chirp start and end times
    # when your received signal comes with the information bits - make it faster!
    delay = signal_processor.find_delay(plot=True) 
    frequency_bins, frequency_response = signal_processor.get_frequency_response(chirp_start_time, chirp_end_time, plot=True)
    FIR = signal_processor.get_FIR(plot=True, truncate=True, file_path='FIR_filters/5.26pm.csv')