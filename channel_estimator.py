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
import time

import scipy.signal


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
        self.trans_chirp = None # Chirp section of the transmitted signal
        self.recv_chirp = None # Chirp section of the received signal
        self.delay1 = None # The delay for the first chirp
        self.delay2 = None # The delay for the second chirp
        self.frequency_response = None # Frequency response of the channel
        self.frequencies = None # Frequency bins
        self.fir_filter = None # FIR filter
        self.direct_FIR = None # Direct FIR filter
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
        Note the start_time and end_time is what we use to truncate the received signal.
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
        correlation_start_time = time.time()
        correlation = scipy.signal.correlate(filtered_recv, filtered_trans, mode='full') #much faster than numpy.correlate which does not use fft
        correlation_end_time = time.time()
        execution_time = correlation_end_time - correlation_start_time
        logging.info(f"Execution time: {execution_time} seconds")
        
        # Finding delay
        relative_delay = np.argmax(correlation) - (len(self.trans) - 1)
        self.delay1 = relative_delay + start_time*self.fs if start_time else relative_delay
        logging.info('Delay = {}'.format(self.delay1))
        
        # Optionally plot the transmitted signal, realligned received signal, and the delay line
        if plot:
            self.plot_time_domain_signals(
                transmitted=self.trans,
                received=truncated_recv,
                delay=relative_delay # Note: The delay relative to the truncated received signal
            )
        return self.delay1
    
    def find_two_delays(self, start1=0, end1=5, start2=-5, plot=False):
        """
        
        Find the start positions of the two chirp signals in the received signal.
        Note the start1 and end1 (+ve) are used to truncate the first received signal,
        while start2 and end2 (-ve) are used to truncate the second received signal.
        """
        
        if start1 != None and end1 != None:
            truncated_recv1 = self.recv[int(start1*self.fs):int(end1*self.fs)]
            offset1 = int(start1*self.fs)
        else:
            truncated_recv1 = self.recv
            offset1 = 0
        
        if start2:
            truncated_recv2 = self.recv[int(start2*self.fs):]
            offset2 = len(self.recv) + int(start2*self.fs)
        else:
            truncated_recv2 = self.recv
            offset2 = 0
            
        # Bandpass filtering both signals
        if self.f_low and self.f_high: # if f_low and f_high are provided
            logging.info('Bandpass filtering the signals (for matched filtering only), f_low = {}, f_high = {}'.format(self.f_low, self.f_high))
            filtered_trans = bandpass_filter(self.trans, self.f_low, self.f_high, self.fs)
            filtered_recv1 = bandpass_filter(truncated_recv1, self.f_low, self.f_high, self.fs)
            filtered_recv2 = bandpass_filter(truncated_recv2, self.f_low, self.f_high, self.fs)
        else:
            filtered_trans = self.trans
            filtered_recv1 = truncated_recv1
            filtered_recv2 = truncated_recv2
                
        # Matched filtering to find correlation
        logging.info('Finding delay... (If this takes too long, you can specify the range to search for delay by providing start_time and end_time to truncate the received signal)')
        correlation1 = scipy.signal.correlate(filtered_recv1, filtered_trans, mode='full')
        correlation2 = scipy.signal.correlate(filtered_recv2, filtered_trans, mode='full')
        relative_delay1 = np.argmax(correlation1) - (len(self.trans) - 1)
        relative_delay2 = np.argmax(correlation2) - (len(self.trans) - 1)
        self.delay1 = relative_delay1 + offset1
        self.delay2 = relative_delay2 + offset2
        
        if plot:
            self.plot_time_domain_signals(
                transmitted=self.trans,
                received=truncated_recv1,
                delay=relative_delay1 # Note: The delay relative to the truncated received signal
            )
            self.plot_time_domain_signals(
                transmitted=self.trans,
                received=truncated_recv2,
                delay=relative_delay2 # Note: The delay relative to the truncated received signal
            )
        
        return self.delay1, self.delay2
        
    
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
    def get_frequency_response(self, chirp_start_index, chirp_end_index, plot=False):
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
        if self.delay1 is None:
            logging.error('Delay is not found. Find the delay first.')
            return
        
        if self.f_low and self.f_high:
            f_low = self.f_low
            f_high = self.f_high
        else:
            logging.warning('f_low and f_high are not provided. Using the default values.')
            f_low = 1000
            f_high = 10000
        
        # truncate both signals to the chirp section
        start_index = chirp_start_index
        end_index = chirp_end_index
        self.trans_chirp = self.trans[start_index:end_index]
        self.recv_chirp = self.recv[self.delay1+start_index:self.delay1+end_index]
        
        # Compute the length for FFT based on the input signal length
        max_length = len(self.trans_chirp)  # Assuming transmitted and received are of the same length

        logging.info('Computing frequency response...')
        # Compute FFT
        fft_transmitted = np.fft.rfft(self.trans_chirp, n=max_length)
        fft_received = np.fft.rfft(self.recv_chirp, n=max_length)
        
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
        plt.figure(figsize=(6, 4))
        plt.plot(frequencies, mag_trans, label='Transmitted Signal')
        plt.plot(frequencies, mag_recv, label='Received Signal', alpha=0.7)
        plt.title('Magnitude Spectra of Transmitted and Received Signals')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0, 10000)
        plt.ylabel('Magnitude (dB)')
        plt.legend()
        plt.show()

        # Plot the transfer function (channel frequency response)
        plt.figure(figsize=(6, 4))
        plt.plot(frequencies, mag)
        plt.title('Magnitude response of the channel (Chirp)')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0, 10000)
        plt.ylabel('Magnitude (dB)')
        plt.show()
        
        # Plot the phase response
        plt.figure(figsize=(6, 4))
        plt.plot(frequencies, phase)
        plt.title('Phase Response of the channel (Chirp)')
        plt.xlabel('Frequency (Hz)')
        plt.xlim(0, 10000)
        plt.ylabel('Phase (Degrees)')
        plt.show()
        
    def truncate_fir_filter(self, fir_filter, energy_percent=0.9, last_100_threshold=0.01):
        """
        Truncate an FIR filter to retain a specified percentage of its total energy
        and to stop if the last 100 samples' energy is below a certain percentage of the total energy.

        Parameters:
            fir_filter (np.array): The FIR filter coefficients.
            energy_percent (float): The desired percentage of total energy to retain (default is 0.9).
            last_100_threshold (float): The threshold percentage for the last 100 samples' energy (default is 0.01).

        Returns:
            np.array: The truncated FIR filter.
        """
        logging.info('Starting to truncate the FIR filter.')
        
        # Calculate total energy of the FIR filter
        total_energy = np.sum(fir_filter**2)
        cumulative_energy = 0
        threshold_energy = last_100_threshold * total_energy  # Calculate threshold energy based on percentage

        # Initialize variables for efficient last 100 sample energy calculation
        last_100_energy = 0
        if len(fir_filter) >= 100:
            last_100_energy = np.sum(fir_filter[:100]**2)

        i = 0
        while i < len(fir_filter):
            cumulative_energy += fir_filter[i]**2

            # Update the last 100 samples energy if at least 100 samples have been processed
            if i >= 100:
                # Update last 100 samples' energy
                last_100_energy += fir_filter[i]**2 - fir_filter[i-100]**2

                # Check if the energy of the last 100 samples is below the threshold
                if last_100_energy < threshold_energy:
                    logging.info('Last 100 samples energy below threshold. Stopping truncation.')
                    break

            # Check if cumulative energy has reached the desired percentage
            if cumulative_energy / total_energy >= energy_percent:
                logging.info('Desired energy percentage reached. Stopping truncation.')
                break

            i += 1

        # Truncate the filter to the determined point
        truncated_filter = fir_filter[:i+1]
        logging.info(f'Truncated FIR filter length = {i+1}')

        return truncated_filter

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
            self.fir_filter = self.truncate_fir_filter(self.fir_filter, 0.9, 0.01)
        
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
        logging.info(f'FIR filter length:{len(self.fir_filter)}')
        if self.fir_filter.shape[0] > self.fs*1.0:
            logging.warning('Invalid FIR filter length > 1 second')
        return self.fir_filter
    def get_direct_FIR(self, plot=False, file_path=None, truncate=False):
        """
        Compute the FIR by cross-correlating the chirp section of the transmitted and received signals.
        """
        
        # Check if the the chirp sections of the transmitted and received signals are available
        if self.trans_chirp is None or self.recv_chirp is None:
            logging.error('Chirp sections of the transmitted and received signals are not available. Find the delay first.')
            return
        logging.info('Computing direct FIR filter...')
        # Calculate the correlation function between the chirp sections of the transmitted and received signals
        correlation = scipy.signal.correlate(self.recv_chirp, self.trans_chirp, mode='full')
        
        # direct_FIR shoud be the shifted by len(self.trans.chirp) - 1
        self.direct_FIR = np.roll(correlation, len(self.trans_chirp)-1)
            
        if truncate:
            self.direct_FIR = self.truncate_fir_filter(self.direct_FIR, 0.9, 0.01)
        
        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(self.direct_FIR)
            plt.title('Direct FIR Filter Calculation - to find the FIR length')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.show()
        
        if file_path:
            logging.info('Saving FIR filter to {}'.format(file_path))
            pd.DataFrame(self.direct_FIR).to_csv(file_path, index=False)
        
        return self.direct_FIR
    
    
if __name__ == '__main__':
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Parameters
    f_low = 20
    f_high = 8000
    fs=48000
    chirp_start_index = 1024
    chirp_end_index = 1024 + 16*4096
    # File paths
    transmitted_signal_path = 'chirps/1k_8k_0523.wav'
    received_signal_path = 'recordings/transmitted_article_2_iceland_pilot1_ldpc1.wav'
    
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
    delay1, delay2 = signal_processor.find_two_delays(start1=0, end1=5, start2=-5, plot=True)
    print(delay1, delay2)
    frequency_bins, frequency_response = signal_processor.get_frequency_response(chirp_start_index, chirp_end_index, plot=True)
    FIR = signal_processor.get_FIR(plot=True, truncate=True, file_path='FIR_filters/5.56pm.csv')
    direct_FIR = signal_processor.get_direct_FIR(plot=True, truncate=True, file_path='FIR_filters/direct_5.56pm.csv')