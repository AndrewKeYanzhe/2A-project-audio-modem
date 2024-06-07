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
from drift_calculator import DriftCalculator

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
                 f_low = None, f_high = None,
                 sync_drift_per_OFDM_symbol = 0):
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
        self.sync_drift_per_OFDM_symbol = sync_drift_per_OFDM_symbol

    def load_data(self, file_path):
        """Load data from a CSV file into a numpy array."""
        return pd.read_csv(file_path, header=None).values.flatten()

    def remove_cyclic_prefix(self, signal):
        """Remove the cyclic prefix of an OFDM signal."""
        num_blocks = round(len(signal) / (self.block_size + self.prefix_length))
        
        blocks = []
        for i in range(num_blocks):
            sync_drift = round(i*self.sync_drift_per_OFDM_symbol)
            start_index = i * (self.block_size + self.prefix_length) + self.prefix_length + sync_drift
            end_index = start_index + self.block_size
            if end_index <= len(signal):
                blocks.append(signal[start_index:end_index])
            else:
                logging.warning(f"Error in block {i} out of {num_blocks}: start={start_index}, end={end_index}, len(signal)={len(signal)}")
                logging.warning("Padd the OFDM with zeros")
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

        logging.info(f"Number of OFDM blocks: {len(blocks)}")

        for index, block in enumerate(blocks):
            
            n_bins = 4096
            if index == 0 and use_pilot_tone:
                logging.info("using pilot tone")
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

                frequencies=subcarrier_frequencies
                phase_response = np.angle(self.g_n, deg=True)

                # # Plot the phase response
                # plt.figure(figsize=(6, 4))
                # plt.plot(frequencies, phase_response)
                # plt.title('Phase Response of the Channel (Pilot symbol)')
                # plt.xlabel('Frequency (Hz)')
                # plt.xlim(0, 10000)
                # plt.ylabel('Phase (Degrees)')
                # plt.show()

            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            if use_pilot_tone == False:
                self.g_n = interpolated_response
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
            logging.info(f"Coordinate: {c}, Magnitude: {math.sqrt(c[0]**2 + c[1]**2)}, Phase: {angle_degrees} degrees")
            if angle_degrees < 0:
                angle_degrees = angle_degrees + 360
            sum_angles = sum_angles + angle_degrees
            kmeans_cluster_magnitudes.append(math.sqrt(c[0]**2 + c[1]**2))
        avg_kmeans_magnitude = sum(kmeans_cluster_magnitudes) / len(kmeans_cluster_magnitudes) 
        logging.info(f"avg_kmeans_magnitude {avg_kmeans_magnitude}")
        phase_shift_needed = (720-sum_angles)/4
        logging.info(f"Phase shift needed: {phase_shift_needed} degrees")
        # Convert centroids back to complex numbers
        centroid_complex_numbers = [complex(c[0], c[1]) for c in centroids]
        # centroid_complex_numbers
        # self.plot_constellation(centroid_complex_numbers, title="k-means clusters="+str(n_clusters), dot_size=100)
        ################################################


        # gn_list store the past channel frequency responses
        gn_list = []
        gn_list.append(self.g_n)
        self.received_constellations = []
        self.compensated_constellations = []
        for index, block in enumerate(tqdm(blocks)):
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)

            x_n = self.channel_compensation(r_n, self.g_n)

            constellations = np.copy(x_n[bin_low:bin_high+1])
            
            shifted_constellations = [z * cmath.exp(1j * math.radians(phase_shift_needed)) for z in constellations]


            if use_ldpc:

                block_length = 1296 #TODO change hardcode
                ldpc_encoded_length = (block_length//24)*24

                if shift_constellation_phase:
                    ldpc_signal_list=np.array(qpsk_demap_probabilities(shifted_constellations, avg_kmeans_magnitude))
                elif shift_constellation_phase == 0:
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
            ######################################################
            # Save the constellation points for plotting
            self.received_constellations.extend(r_n[bin_low:bin_high+1])
            self.compensated_constellations.extend(x_n[bin_low:bin_high+1]) 
            
        phase_for_85=[]
        for i in range(len(gn_list)):
            phase_for_85.append(math.degrees(np.angle(gn_list[i][86])))
        plt.scatter(range(len(phase_for_85)),phase_for_85)
        plt.title("Phase of g[n] at bin 85")
        plt.show()
        
        # self.plot_constellation(self.received_constellations, title="Constellation\nBefore Compensation")
        # self.plot_constellation(self.compensated_constellations, title="Constellation\nAfter Compensation")
        logging.info(f"Recovered Binary Data Length: {len(complete_binary_data)}")
        return complete_binary_data
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
        
        # Convert centroids back to complex numbers
        centroid_complex_numbers = [complex(c[0], c[1]) for c in centroids]
        
        # Apply the phase shift to the original constellations
        shifted_constellations = [z * cmath.exp(1j * math.radians(phase_shift_needed)) for z in compensated_constellations]
        
        return shifted_constellations
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
        logging.info(f"Binary data has been saved to {file_path}.")
    def save_file(self, file_path, content):
        """Save the content to a file."""
        with open(file_path, 'wb') as file:
            file.write(content)
        logging.info(f"File has been saved to {file_path}. Please check the file to see if the image is correctly reconstructed.")


if __name__ == "__main__":
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
    manual_shift = 100
    chirp_transmitted_path = 'chirps/1k_8k_0523_suffix.wav'
    received_signal_path = 'recordings/cat_LR11.wav'
    received_signal_path = 'recordings/transmitted_article_2_iceland_pilot1_ldpc1.wav'
    received_signal_path = 'recordings/0605_demo_test_4.m4a'
    received_signal_path = 'recordings/g168try1.wav'
    # kmeans flag
    shift_constellation_phase = False
    use_pilot_tone = True
    use_ldpc = True
    two_chirps = False
    remove_header_frontNulls=True
    trim_end = True
    remove_drift = True

    recording_name = os.path.splitext(os.path.basename(received_signal_path))[0]
    
    # Initialize AnalogueSignalProcessor with the chirp signals
    asp = AnalogueSignalProcessor(chirp_transmitted_path, received_signal_path,chirp_f_low,chirp_f_high)
    # Load the chirp signals
    asp.load_audio_files()

    # Find the delay
    delay1, delay2 = asp.find_two_delays(0,5,-5, plot=False, plot_corr=False)
    logging.info(f"delay1 = {delay1}")
    logging.info(f"delay2 = {delay2}")

    ######## Use the drift calculator to estimate the drift ########
    if remove_drift:
        start_index = int(delay1) # delay is an integer though
        info_start_index = start_index+1024*2+4096*16
        info_end_index = int(delay2)
        received_signal_unshifted = asp.recv[info_start_index:info_end_index]
        unshifted_signal_path = './files/unshifted_received_signal_' + recording_name + '.csv'
        pd.DataFrame(received_signal_unshifted).to_csv(unshifted_signal_path, index=False, header=False)
        drift_calculator = DriftCalculator(channel_file =unshifted_signal_path,
                                        received_file=unshifted_signal_path,
                                        fs=fs,
                                        prefix_length=OFDM_prefix_length, block_size=OFDM_block_size,
                                        f_low=chirp_f_low, f_high=chirp_f_high)
        # The gradient is degree per OFDM symbol
        gradient_86th_bin = drift_calculator.find_drift_per_block()
        logging.info(f"Gradient at 86th bin: {gradient_86th_bin}")
        sync_drift_per_OFDM_symbol = -gradient_86th_bin * 4096 / (2*86*180)
        logging.info(f"Sync drift per OFDM symbol = {sync_drift_per_OFDM_symbol}")
        
        ### shift to left to avoid information loss ###
        delay1 = delay1 - manual_shift
        delay2 = delay2 - manual_shift
    ################################################################
    else:      
        sync_drift_per_OFDM_symbol = 0  
        delay1 = delay1 - manual_shift
        delay2 = delay2 - manual_shift

    if two_chirps:
        # Trim the received signal
        start_index = int(delay1) # delay is an integer though
        info_start_index = start_index+1024*2+4096*16
        info_end_index = int(delay2)
        logging.info(f"Sync drift per OFDM symbol = {sync_drift_per_OFDM_symbol}")
        
        received_signal_trimmed = asp.recv[info_start_index:info_end_index]
    else:
        start_index = int(delay1) # delay is an integer though
        received_signal_trimmed = asp.recv[start_index+1024*2+4096*16:] #can directly use int()??

    
    # # Save the trimmed signal to a new file (or directly process it)
    trimmed_signal_path = './files/trimmed_received_signal_' + recording_name + '.csv'
    logging.info(f"Saving trimmed received signal to:{trimmed_signal_path}")
    pd.DataFrame(received_signal_trimmed).to_csv(trimmed_signal_path, index=False, header=False)
    
    # Compute the frequency response
    frequencies, frequency_response = asp.get_frequency_response(chirp_start_index, chirp_end_index, plot=False)

    # # Initialize Receiver with the trimmed signal
    logging.info("Start Demodulating ")
    receiver = Receiver(channel_file =trimmed_signal_path,
                        received_file=trimmed_signal_path,
                        fs=fs,
                        frequencies=frequencies,
                        channel_impulse_response=frequency_response,
                        prefix_length=OFDM_prefix_length, block_size=OFDM_block_size,
                        f_low=chirp_f_low, f_high=chirp_f_high,
                        sync_drift_per_OFDM_symbol=sync_drift_per_OFDM_symbol)

    binary_data = receiver.process_signal()
    
    if remove_header_frontNulls:
        # # Example usage
        # binary_data = "000000000000000011010101"
        binary_data = remove_leading_zeros(binary_data)
        filename, number_of_bits,binary_data = split_by_first_two_occurrences(binary_data)
        filename=receiver.binary_to_bytes(filename).decode('utf-8')
        number_of_bits=receiver.binary_to_bytes(number_of_bits).decode('utf-8')
        logging.info(f"Filename: {filename}")
        logging.info(f"Number of bits: {number_of_bits}")
        try:
            if trim_end:
                binary_data = binary_data[:int(number_of_bits)]
        except ValueError:
            logging.warning("A ValueError occurred, because number_of_bits is not an integer. Hence trim_end is not used.")

    if two_chirps:
        deomudulated_binary_path = './binaries/received_'+recording_name+'_resampled.bin'
    else:
        deomudulated_binary_path='./binaries/received_'+recording_name+'.bin'
    binfile = receiver.binary_to_bin_file(binary_data, deomudulated_binary_path)