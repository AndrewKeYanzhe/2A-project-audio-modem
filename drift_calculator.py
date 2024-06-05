import numpy as np
import pandas as pd
import scipy
from channel_estimator import AnalogueSignalProcessor
from utils import *
import matplotlib.pyplot as plt
import logging
from scipy.io.wavfile import write
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import math
import cmath
import os
from ldpc_function import *
from tqdm import tqdm
from calculate_gradient_phaseWrap import calculate_gradients

class DriftCalculator:
    def __init__(self, channel_file, received_file,
                 fs, prefix_length, block_size,
                 f_low = None, f_high = None):
        self.channel_file = channel_file
        self.fs= fs
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
        num_blocks = round(len(signal) / (self.block_size + self.prefix_length))
        
        blocks = []
        for i in range(num_blocks):
            start_index = i * (self.block_size + self.prefix_length) + self.prefix_length
            end_index = start_index + self.block_size
            if end_index <= len(signal):
                blocks.append(signal[start_index:end_index])
            else:
                logging.info(f"Error in block {i} out of {num_blocks}: start={start_index}, end={end_index}, len(signal)={len(signal)}")
                logging.info("Padd the OFDM with zeros")
                ith_block = np.concatenate((signal[start_index:], np.zeros(self.block_size - len(signal[start_index:]))))
                blocks.append(ith_block)
        return blocks

    def apply_fft(self, signal, n):
        """Apply FFT to the signal."""
        return np.fft.fft(signal, n=n)

    def channel_compensation(self, r_n, g_n):
        """Compensate for the channel effects in the frequency domain."""
        epsilon = 1e-10  # To avoid division by zero or very small values
        return r_n / (g_n + epsilon)

    def map_bits_to_numbers(self, binary_data):
        """Map 2N information bits to N numbers using QPSK with Gray coding."""
        constellation = {
            '00': 0,
            '01': 1,
            '11': 2,
            '10': 3
        }
        String_bin = ''
        for bit in binary_data:
            String_bin += str(bit)
        numbers = []
        for i in range(0, len(String_bin), 2):
            bits = String_bin[i:i+2]
            numbers.append(constellation[bits])
        return numbers
    def map_numbers_to_symbols(self, numbers):
        """Map N information numbers to N constellation symbols using QPSK with Gray coding."""
        # Define the QPSK Gray coding constellation symbols
        constellation = {
            0: complex(1, 1),    # '00'
            1: complex(-1, 1),   # '01'
            2: complex(-1, -1),  # '11'
            3: complex(1, -1)    # '10'
        }    
        # List to store the constellation symbols
        symbols = []
        # Iterate over the list of numbers
        for number in numbers:
            # Append the corresponding symbol to the list
            symbols.append(constellation[number])
        return symbols
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
                logging.debug(f"No matching constellation point found for symbol {symbol}")

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

    def find_drift_per_block(self):
        """Process the received signal to recover binary data."""
        self.received_signal = self.load_data(self.received_file)

        # Remove cyclic prefix and get blocks
        blocks = self.remove_cyclic_prefix(self.received_signal)
        # Estimate channel frequency response
        # Process each block
        complete_binary_data = ''
        # Get the frequency bins corresponding to the given frequency range
        bin_low,bin_high = cut_freq_bins(self.f_low, self.f_high, self.fs, self.block_size) 
        logging.debug(f"Number of OFDM blocks: {len(blocks)}")

        ############### First Iteration ################
        for index, block in enumerate(blocks):
            
            n_bins = 4096
            if index == 0:
                logging.debug("using pilot tone")
                np.random.seed(1)
                constellation_points = np.array([1+1j, -1+1j, -1-1j, 1-1j])
                symbols_extended = np.random.choice(constellation_points, n_bins)

                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))
                pilot_n = symbols_extended
                r_n = self.apply_fft(block, self.block_size)
                for i in range(len(pilot_n)):
                    if pilot_n[i] == 0:
                        pilot_n[i] = 0.00000001
                pilot_response = r_n/pilot_n

                self.g_n = pilot_response
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)
            # Save the constellation points for plotting
            self.received_constellations.extend(r_n[bin_low:bin_high+1])
            self.compensated_constellations.extend(x_n[bin_low:bin_high+1]) 
        
        ############### K MEANS CLUSTERING ################
        data = np.array([[z.real, z.imag] for z in self.compensated_constellations])
        # data = np.array([[z.real, z.imag] for z in subsample])
        n_clusters = 5
        # Apply k-means clustering
        kmeans = KMeans(n_clusters, init='k-means++', n_init=10, random_state=42).fit(data)
        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        # top_4 = sorted(centroids, key=lambda c: c[0]**2 + c[1]**2, reverse=True)[:4]
        top_5 = sorted(centroids, key=lambda c: c[0]**2 + c[1]**2, reverse=True)[:5]
        # Step 1: Calculate magnitudes
        magnitudes = [np.linalg.norm(coord) for coord in top_5]
        # Step 2: Filter out magnitudes larger than 4
        filtered_coords_magnitudes = [(coord, mag) for coord, mag in zip(top_5, magnitudes) if mag <= 4]
        # Step 3: Sort the remaining magnitudes in descending order based on magnitudes
        sorted_filtered_coords_magnitudes = sorted(filtered_coords_magnitudes, key=lambda x: x[1], reverse=True)
        # Step 4: Extract the top 4 coordinates
        top_4 = [coord for coord, mag in sorted_filtered_coords_magnitudes[:4]]
        phases = [(c, math.atan2(c[1], c[0])) for c in top_4]
        phases_sorted = sorted(phases, key=lambda x: x[1])
        # phases_sorted = [phase + 360 if phase < 0 else phase for phase in phases_sorted]
        kmeans_cluster_magnitudes = []
        sum_angles = 0
        for c, angle in phases_sorted:
            # Convert angle from radians to degrees
            angle_degrees = math.degrees(angle)
            logging.debug(f"Coordinate: {c}, Magnitude: {math.sqrt(c[0]**2 + c[1]**2)}, Phase: {angle_degrees} degrees")
            if angle_degrees < 0:
                angle_degrees = angle_degrees + 360
            sum_angles = sum_angles + angle_degrees
            kmeans_cluster_magnitudes.append(math.sqrt(c[0]**2 + c[1]**2))
        avg_kmeans_magnitude = sum(kmeans_cluster_magnitudes) / len(kmeans_cluster_magnitudes) 
        logging.info(f"avg_kmeans_magnitude {avg_kmeans_magnitude}")
        ################################################


        # gn_list store the past channel frequency responses
        gn_list = []
        self.received_constellations = []
        self.compensated_constellations = []
        
        ###### Iterative Channel Estimation ########
        for index, block in enumerate(tqdm(blocks)):
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            x_n = self.channel_compensation(r_n, self.g_n)
            constellations = np.copy(x_n[bin_low:bin_high+1])
            
            block_length = 1296 #TODO change hardcode
            ldpc_encoded_length = (block_length//24)*24
            ldpc_signal_list=np.array(qpsk_demap_probabilities(constellations, avg_kmeans_magnitude))
            ldpc_signal_list = ldpc_signal_list[0:ldpc_encoded_length]
            ldpc_decoded, ldpc_decoded_with_redundancies = decode_ldpc(ldpc_signal_list)
            #convert list to string
            ldpc_decoded = ''.join(str(x) for x in ldpc_decoded)
            complete_binary_data += ldpc_decoded
            
            ############ Update g[n] for next block ############
            numbers = self.map_bits_to_numbers(ldpc_decoded_with_redundancies)
            
            # Generate the pseudo-random sequence used in the transmitter
            np.random.seed(1)
            random_constellation_points = np.array([0, 1, 2, 3])
            number_extended = np.random.choice(random_constellation_points, 4096)
            modulus_multiplication_result = list(number_extended)
            # Perform modulus 4 multiplication for the relevant indices
            for i in range(len(numbers)):
                corresponding_index = i + 85
                result = (numbers[i] + number_extended[corresponding_index]) % 4
                modulus_multiplication_result[corresponding_index] = result

            ldpc_xn = self.map_numbers_to_symbols(modulus_multiplication_result)
            ldpc_xn[0] = 0
            ldpc_xn[n_bins // 2] = 0
            ldpc_xn[n_bins//2+1:] = np.conj(np.flip(ldpc_xn[1:n_bins//2]))
            for i in range(len(ldpc_xn)):
                if ldpc_xn[i] == 0:
                    ldpc_xn[i] = 0.00000001

            gn_list.append(r_n/(ldpc_xn))
            ######### Update the g_n using AR model ##########
            self.g_n =0.6*self.g_n + 0.4*(r_n/(ldpc_xn))
            ##################################################
            # Save the constellation points for plotting
            self.received_constellations.extend(r_n[bin_low:bin_high+1])
            self.compensated_constellations.extend(x_n[bin_low:bin_high+1]) 

        phase_for_86=[]
        for i in range(len(gn_list)):
            phase_for_86.append(math.degrees(np.angle(gn_list[i][286])))
        mean_gradient = calculate_gradients(phase_for_86, plot=True)
        logging.debug(f"Mean Gradient: {mean_gradient}")
        
        return mean_gradient
    

    def apply_kmeans(self, compensated_constellations, n_clusters=5, random_state=42):

        # Convert complex numbers to a 2D array of their real and imaginary parts
        data = np.array([[z.real, z.imag] for z in compensated_constellations])
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=random_state).fit(data)
        
        # Get the cluster centroids
        centroids = kmeans.cluster_centers_
        
        # Sort the top 4 centroids by magnitude
        top_4 = sorted(centroids, key=lambda c: c[0]**2 + c[1]**2, reverse=True)[:4]
        
        # Calculate phases and sort them
        phases = [(c, math.atan2(c[1], c[0])) for c in top_4]
        phases_sorted = sorted(phases, key=lambda x: x[1])
        
        # Calculate the sum of angles in degrees
        sum_angles = 0
        for c, angle in phases_sorted:
            angle_degrees = math.degrees(angle)
            if angle_degrees < 0:
                angle_degrees += 360
            sum_angles += angle_degrees
        
        # Calculate the phase shift needed
        phase_shift_needed = (720 - sum_angles) / 4
        
        # Apply the phase shift to the original constellations
        shifted_constellations = [z * cmath.exp(1j * math.radians(phase_shift_needed)) for z in compensated_constellations]
        
        return shifted_constellations
    
    
if __name__ == "__main__":
    # Set the logging level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parameters
    fs =  48000
    # recording_name = '0525_1749'
    OFDM_prefix_length = 1024
    OFDM_suffix_length = 1024
    OFDM_block_size = 4096
    chirp_start_index = 1024
    chirp_end_index = 1024 + 4096*16
    chirp_f_low = 761.72
    chirp_f_high = 8824.22
    chirp_transmitted_path = 'chirps/1k_8k_0523_suffix.wav'
    received_signal_path = 'recordings/0605_demo_test_3.m4a'
    recording_name = os.path.splitext(os.path.basename(received_signal_path))[0]
    
    # Initialize AnalogueSignalProcessor with the chirp signals
    asp = AnalogueSignalProcessor(chirp_transmitted_path, received_signal_path,chirp_f_low,chirp_f_high)
    # Load the chirp signals
    asp.load_audio_files()
    delay1, delay2 = asp.find_two_delays(0,5,-5, plot=False, plot_corr=False)
    logging.info(f"delay1 = {delay1}")
    logging.info(f"delay2 = {delay2}")

    ########## Trim the received signal ##########
    start_index = int(delay1) # delay is an integer though
    info_start_index = start_index+1024*2+4096*16
    info_end_index = int(delay2)
    # sync_drift_per_OFDM_symbol = -0.23997/(2*math.pi)
    received_signal_trimmed = asp.recv[info_start_index:info_end_index]
    # # Save the trimmed signal to a new file (or directly process it)
    trimmed_signal_path = './files/trimmed_received_signal_' + recording_name + '.csv'
    logging.info(f"Saving trimmed received signal to:{trimmed_signal_path}")
    pd.DataFrame(received_signal_trimmed).to_csv(trimmed_signal_path, index=False, header=False)
    
    # # Initialize Receiver with the trimmed signal
    logging.info("Start Demodulating ")
    drift_calc = DriftCalculator(channel_file =trimmed_signal_path,
                        received_file=trimmed_signal_path,
                        fs=fs,
                        prefix_length=OFDM_prefix_length, block_size=OFDM_block_size,
                        f_low=chirp_f_low, f_high=chirp_f_high)

    # Find the drift per OFDM block
    gradient_86th_bin = drift_calc.find_drift_per_block()
    logging.info(f"Gradient of 86th bin: {gradient_86th_bin}")
    