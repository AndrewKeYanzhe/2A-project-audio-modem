import numpy as np
import pandas as pd
import scipy
from channel_estimator import AnalogueSignalProcessor
from utils import save_as_wav, cut_freq_bins
import matplotlib.pyplot as plt
import logging
from scipy.io.wavfile import write

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

import random

import math
import cmath

import os

from ldpc_function import *

"""
    The Receiver class processes an OFDM signal to recover binary data, convert it to bytes,
    and save it to a file. It handles loading data, removing cyclic prefixes, applying FFT,
    compensating for channel effects, demapping QPSK symbols, and parsing binary data.

    Attributes:
        channel_file (str): The path to the CSV file containing the channel impulse response.
        channel_impulse_response 
        received_file (str): The path to the CSV file containing the received OFDM signal.
        prefix_length (int): The length of the cyclic prefix in the OFDM signal.
        block_size (int): The size of each OFDM block (excluding the cyclic prefix).
        channel_impulse_response (np.ndarray): The channel impulse response data.
        received_signal (np.ndarray): The received OFDM signal data.
        g_n (np.ndarray): The estimated channel frequency response.
    
    Methods:
        load_data(file_path): Load data from a CSV file into a numpy array.
        remove_cyclic_prefix(signal): Remove the cyclic prefix of an OFDM signal.
        apply_fft(signal, n): Apply FFT to the signal.
        channel_compensation(r_n, g_n): Compensate for the channel effects in the frequency domain.
        qpsk_demapper(compensated_symbols): Demap QPSK symbols to binary data.
        process_signal(): Process the received signal to recover binary data.
        binary_to_bytes(binary_data): Convert binary data to bytes.
        parse_bytes_data(bytes_data): Parse the bytes data to extract filename, size, and content.
        save_file(file_path, content): Save the content to a file.
    """
class Receiver:
    def __init__(self, channel_file, received_file,
                 fs,frequencies, channel_impulse_response,
                 prefix_length, block_size,
                 f_low = None, f_high = None):
        self.channel_file = channel_file
        self.fs= fs
        self.frequencies = frequencies
        self.channel_impulse_response = channel_impulse_response
        self.received_file = received_file
        self.prefix_length = prefix_length
        self.block_size = block_size
        self.received_signal = None
        self.received_constellations = []
        self.compensated_constellations = []
        self.compensated_constellations_subsampled = []
        self.g_n = None
        self.f_low = f_low
        self.f_high = f_high

    def load_data(self, file_path):
        """Load data from a CSV file into a numpy array."""
        return pd.read_csv(file_path, header=None).values.flatten()

    def remove_cyclic_prefix(self, signal):
        """Remove the cyclic prefix of an OFDM signal."""
        num_blocks = len(signal) // (self.block_size + self.prefix_length)
        blocks = []
        for i in range(num_blocks):
            start_index = i * (self.block_size + self.prefix_length) + self.prefix_length
            end_index = start_index + self.block_size
            blocks.append(signal[start_index:end_index])
        return blocks

    def apply_fft(self, signal, n):
        """Apply FFT to the signal."""
        return np.fft.fft(signal, n=n)

    def channel_compensation(self, r_n, g_n):
        """Compensate for the channel effects in the frequency domain."""
        epsilon = 1e-10  # To avoid division by zero or very small values
        return r_n / (g_n + epsilon)

    def qpsk_demapper(self, compensated_symbols, n_bins=4096, seed=1, offset=85):
        """Decode compensated QPSK symbols to original binary data."""
        # Define the QPSK Gray coding constellation mapping
        constellation = {
            complex(1, 1): 0,    # '00'
            complex(-1, 1): 1,   # '01'
            complex(-1, -1): 2,  # '11'
            complex(1, -1): 3    # '10'
        }

        # Demap QPSK symbols to numbers {0, 1, 2, 3}
        demapped_numbers = []
        for symbol in compensated_symbols:
            min_dist = float('inf')
            number = None
            for point, mapping in constellation.items():
                dist = np.abs(symbol - point)
                if dist < min_dist:
                    min_dist = dist
                    number = mapping
            if number is not None:
                demapped_numbers.append(number)
            else:
                logging.warning(f"No matching constellation point found for symbol {symbol}")

        # Reverse the modulus multiplication to get the original numbers
        np.random.seed(seed)
        constellation_points = np.array([0, 1, 2, 3])
        number_extended = np.random.choice(constellation_points, n_bins)[85:85+648]

        original_numbers = []

        for i in range(len(demapped_numbers)):
            corresponding_index = i % 648
            for x in range(4):
                if (x + number_extended[corresponding_index]) % 4 == demapped_numbers[i]:
                    original_numbers.append(x)
                    break

        # Map numbers back to binary data using QPSK with Gray coding
        reverse_constellation = {
            0: '00',
            1: '01',
            2: '11',
            3: '10'
        }

        binary_data = ''
        for number in original_numbers:
            binary_data += reverse_constellation[number]

        return binary_data

    def interpolate_frequency_response(self, subcarrier_frequencies):
        """Interpolate the frequency response to match the OFDM subcarrier frequencies."""
        interpolation_function = scipy.interpolate.interp1d(self.frequencies, self.channel_impulse_response, kind='linear', fill_value="extrapolate")
        interpolated_response = interpolation_function(subcarrier_frequencies)
    
        # Add epsilon to avoid division by zero in channel compensation
        epsilon = 1e-10
        interpolated_response = np.where(np.abs(interpolated_response) < epsilon, epsilon, interpolated_response)
    
        return interpolated_response

    def process_signal(self):
        """Process the received signal to recover binary data."""
        
        self.received_signal = self.load_data(self.received_file)

        # Remove cyclic prefix and get blocks
        blocks = self.remove_cyclic_prefix(self.received_signal)
        # Define subcarrier frequencies for OFDM
        subcarrier_frequencies = np.fft.fftfreq(self.block_size, d=1/self.fs)

        # Interpolate the frequency response to match the subcarrier frequencies
        interpolated_response = self.interpolate_frequency_response(subcarrier_frequencies)

        # Estimate channel frequency response
        # Process each block
        complete_binary_data = ''
        # Get the frequency bins corresponding to the given frequency range
        bin_low,bin_high = cut_freq_bins(self.f_low, self.f_high, self.fs, self.block_size) 

        print("number of blocks:",len(blocks))

        for index, block in enumerate(blocks):
            
            n_bins = 4096
            if index == 0 and use_pilot_tone:
                print("using pilot tone")
                np.random.seed(1)
                constellation_points = np.array([1+1j, -1+1j, -1-1j, 1-1j])
                symbols_extended = np.random.choice(constellation_points, n_bins)

                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))
                pilot_n = symbols_extended
                r_n = self.apply_fft(block, self.block_size)
                print("pilot_n length",len(pilot_n))
                pilot_response = r_n/pilot_n

                self.g_n = pilot_response

                frequencies=subcarrier_frequencies
                phase_response = np.angle(self.g_n, deg=True)

                # Plot the phase response
                plt.figure(figsize=(6, 4))
                plt.plot(frequencies, phase_response)
                plt.title('Phase Response of the Channel (Pilot symbol)')
                plt.xlabel('Frequency (Hz)')
                plt.xlim(0, 10000)
                plt.ylabel('Phase (Degrees)')
                plt.show()

            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            if use_pilot_tone == False:
                self.g_n = interpolated_response
            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)

            # Save the constellation points for plotting
            self.received_constellations.extend(r_n[bin_low:bin_high+1])
            self.compensated_constellations.extend(x_n[bin_low:bin_high+1]) 

        self.plot_constellation(self.received_constellations, title="Constellation\nBefore Compensation")

        self.plot_constellation(self.compensated_constellations, title="Constellation\nAfter Compensation")
        # self.plot_constellation(compensated_constellations_subsampled, title="Constellation After Compensation,\nsubsampled 1:10")


        data = np.array([[z.real, z.imag] for z in self.compensated_constellations])
        # data = np.array([[z.real, z.imag] for z in subsample])
        n_clusters = 5

        # Apply k-means clustering
        kmeans = KMeans(n_clusters, init='k-means++', n_init=10, random_state=42).fit(data)

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_

        top_4 = sorted(centroids, key=lambda c: c[0]**2 + c[1]**2, reverse=True)[:4]
        phases = [(c, math.atan2(c[1], c[0])) for c in top_4]

        phases_sorted = sorted(phases, key=lambda x: x[1])

        sum_angles = 0
        for c, angle in phases_sorted:
            # Convert angle from radians to degrees
            angle_degrees = math.degrees(angle)
            
            print(f"Coordinate: {c}, Magnitude: {math.sqrt(c[0]**2 + c[1]**2)}, Phase: {angle_degrees} degrees")
            # print(angle_degrees)
            if angle_degrees < 0:
                angle_degrees = angle_degrees + 360
            
            sum_angles = sum_angles + angle_degrees
 
        phase_shift_needed = (720-sum_angles)/4
        print("phase shift needed", phase_shift_needed)

        # Convert centroids back to complex numbers
        centroid_complex_numbers = [complex(c[0], c[1]) for c in centroids]

        # centroid_complex_numbers

        self.plot_constellation(centroid_complex_numbers, title="k-means clusters="+str(n_clusters), dot_size=100)


        for index, block in enumerate(blocks):
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            if use_pilot_tone == False:
                self.g_n = interpolated_response
            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)

            constellations = np.copy(x_n[bin_low:bin_high+1])
            
            shifted_constellations = [z * cmath.exp(1j * math.radians(phase_shift_needed)) for z in constellations]
  
            
            if shift_constellation_phase:
                binary_data = self.qpsk_demapper(shifted_constellations) # change: now we only demap the frequency bins of interest
            else:
                binary_data = self.qpsk_demapper(constellations) # change: now we only demap the frequency bins of interest

            # print("binary_data length",len(binary_data))

            

            if use_ldpc:
                # if index == 0 and use_pilot_tone:
                #     continue


                block_length = len(binary_data)
                ldpc_encoded_length = (block_length//24)*24

                ldpc_signal = binary_data[0:ldpc_encoded_length]

                # print(list(ldpc_signal))

                #convert string to list
                ldpc_signal_list = np.array([int(element) for element in list(ldpc_signal)])

                # print(ldpc_signal_list)

                ldpc_decoded, ldpc_decoded_with_redundancies = decode_ldpc(ldpc_signal_list)

                
                #convert list to string
                ldpc_decoded = ''.join(str(x) for x in ldpc_decoded)

                complete_binary_data += ldpc_decoded

            elif use_ldpc == False:
                if index != 0:
                    complete_binary_data += binary_data

        logging.info(f"Recovered Binary Data Length: {len(complete_binary_data)}")
        return complete_binary_data

    def plot_constellation(self, symbols, title="QPSK Constellation", dot_size=20):
        font_size = 16
        plt.figure(figsize=(4, 4))
        plt.scatter(np.real(symbols), np.imag(symbols), marker='.', s=dot_size)
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('Real', fontsize=font_size)
        plt.ylabel('Imaginary', fontsize=font_size)
        plt.title(title, fontsize=font_size)
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=font_size)
        plt.tight_layout()
        plt.show()

    def binary_to_bytes(self, binary_data):
        """Convert binary data to bytes."""
        # Pad the binary string to make its length a multiple of 8
        padded_binary = binary_data + '0' * ((8 - len(binary_data) % 8) % 8)
        byte_array = bytearray()
        for i in range(0, len(padded_binary), 8):
            byte_part = padded_binary[i:i + 8]
            byte_array.append(int(byte_part, 2))
        return bytes(byte_array)

    def parse_bytes_data(self, bytes_data):
        """Parse the bytes data to extract filename, size, and content."""
        # Splitting the data at null bytes
        parts = bytes_data.split(b'\0')
        filename = parts[0].decode('utf-8')
        start_of_image_data = bytes_data.find(b'\0', bytes_data.find(b'\0') + 1) + 1
        file_size = int(bytes_data[bytes_data.find(b'\0') + 1: start_of_image_data - 1])

        file_content = bytes_data[start_of_image_data:start_of_image_data + file_size]

        return filename, file_size, file_content
    def binary_to_bin_file(self, binary_data, file_path):
        """Convert binary string to a .bin file."""
        padded_binary = binary_data + '0' * ((8 - len(binary_data) % 8) % 8)
        byte_array = bytearray()
        for i in range(0, len(padded_binary), 8):
            byte_part = padded_binary[i:i + 8]
            byte_array.append(int(byte_part, 2))
        with open(file_path, 'wb') as bin_file:
            bin_file.write(byte_array)
        print(f"Binary data has been saved to {file_path}.")

    def save_file(self, file_path, content):
        """Save the content to a file."""
        with open(file_path, 'wb') as file:
            file.write(content)
        print(f"File has been saved to {file_path}. Please check the file to see if the image is correctly reconstructed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

    # Parameters
    fs =  48000
    # recording_name = '0525_1749'
    OFDM_prefix_length = 1024
    OFDM_suffix_length = 0
    OFDM_block_size = 4096
    chirp_start_time = 0.0  # Example start time of chirp
    chirp_end_time = 15.0    # Example end time of chirp
    chirp_f_low = 761.72
    chirp_f_high = 8824.22
    chirp_transmitted_path = 'chirps/1k_8k_0523.wav'
    received_signal_path = 'recordings/transmitted_article_2_iceland_pilot1_ldpc1.wav'
    # received_signal_path = 'recordings/0602_1120_iceland_ldpc_noSuffix.m4a'

    # kmeans flag
    shift_constellation_phase = False

    use_pilot_tone = True
    use_ldpc = True
    # pilot1, ldpc0/1 works
    # pilot0, ldpc0/1 doesnt work

    recording_name = os.path.splitext(os.path.basename(received_signal_path))[0]

    
    # Initialize AnalogueSignalProcessor with the chirp signals
    asp = AnalogueSignalProcessor(chirp_transmitted_path, received_signal_path,chirp_f_low,chirp_f_high)

    # Load the chirp signals
    asp.load_audio_files()

    # Find the delay
    delay = asp.find_delay(0,10,plot=True)

    # Trim the received signal
    start_index = int(delay) # delay is an integer though
    received_signal_trimmed = asp.recv[start_index+1024*1+int(1.365*fs):] #can directly use int()??

    # # Save the trimmed signal to a new file (or directly process it)
    trimmed_signal_path = './files/trimmed_received_signal_' + recording_name + '.csv'
    logging.info(f"Saving trimmed received signal to:{trimmed_signal_path}")
    pd.DataFrame(received_signal_trimmed).to_csv(trimmed_signal_path, index=False, header=False)
    
    # # Also save the trimmed signal to a WAV file
    # trimmed_signal_path_wav = './recordings/trimmed_received_signal_' + recording_name + '.wav'
    # save_as_wav(signal=received_signal_trimmed, file_path=trimmed_signal_path_wav, fs=fs)
    # logging.info(f"Saving trimmed received signal to:{trimmed_signal_path_wav}")

    # Compute the frequency response
    frequencies, frequency_response = asp.get_frequency_response(chirp_start_time, chirp_end_time, plot=False)


    # Compute the FIR filter (impulse response) from the frequency response
    impulse_response = asp.get_FIR(plot=False, truncate=False)
    direct_impulse_response = asp.get_direct_FIR(plot=False, truncate=False)

    # # Initialize Receiver with the trimmed signal
    print("start demodulating ")
    receiver = Receiver(channel_file =trimmed_signal_path,
                        received_file=trimmed_signal_path,
                        fs=fs,
                        frequencies=frequencies,
                        channel_impulse_response=frequency_response,
                        prefix_length=OFDM_prefix_length, block_size=OFDM_block_size,
                        f_low=chirp_f_low, f_high=chirp_f_high)

    binary_data = receiver.process_signal()
    deomudulated_binary_path='./binaries/received_'+recording_name+'.bin'
    binfile = receiver.binary_to_bin_file(binary_data, deomudulated_binary_path)
    # binfile = receiver.binary_to_bin_file(binary_data+"00000000000000000000000000000", deomudulated_binary_path)

    # bytes_data = receiver.binary_to_bytes(binary_data)
    # filename, file_size, content = receiver.parse_bytes_data(bytes_data)
    # print("Filename:", filename)
    # print("File Size:", file_size)
    # print(content[0:10])

    # Save the byte array to a file
    # output_file_path = './files/test_image_received.tiff'
    # receiver.save_file(output_file_path, content)