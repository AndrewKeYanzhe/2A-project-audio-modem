import numpy as np
import pandas as pd
import scipy
from channel_estimator import AnalogueSignalProcessor
from utils import save_as_wav
import matplotlib.pyplot as plt
import logging
from scipy.io.wavfile import write
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
    def __init__(self, channel_file, received_file,fs,frequencies, channel_impulse_response, prefix_length, block_size):
        self.channel_file = channel_file
        self.fs= fs
        self.frequencies = frequencies
        self.channel_impulse_response = channel_impulse_response
        self.received_file = received_file
        self.prefix_length = prefix_length
        self.block_size = block_size
        self.received_signal = None
        self.received_constellations = None
        self.g_n = None

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

    def qpsk_demapper(self, compensated_symbols):
        """Demap QPSK symbols to binary data."""
        constellation = {
            complex(1, 1): '00',
            complex(-1, 1): '01',
            complex(-1, -1): '11',
            complex(1, -1): '10'
        }
        binary_data = ''
        for symbol in compensated_symbols:
            min_dist = float('inf')
            bits = None
            for point, mapping in constellation.items():
                dist = np.abs(symbol - point)
                if dist < min_dist:
                    min_dist = dist
                    bits = mapping
            if bits is not None:
                binary_data += bits
            else:
                logging.warning(f"No matching constellation point found for symbol {symbol}")
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
        # Load the channel impulse response and the received signal
        #self.channel_impulse_response = self.load_data(self.channel_file)#(这是原来的代码，channel。csv情况下,这样做的话channel_impulse_response指FIR，所以下面要fft)

        self.received_signal = self.load_data(self.received_file)

        # Remove cyclic prefix and get blocks
        blocks = self.remove_cyclic_prefix(self.received_signal)
        # Define subcarrier frequencies for OFDM
        subcarrier_frequencies = np.fft.fftfreq(self.block_size, d=1/self.fs)

        # Interpolate the frequency response to match the subcarrier frequencies
        interpolated_response = self.interpolate_frequency_response(subcarrier_frequencies)


        # Estimate channel frequency response
        #self.g_n = self.apply_fft(self.channel_impulse_response, self.block_size)（这是原来的代码，channel。csv情况下,这样做的话channel_impulse_response指FIR）
        # Process each block
        complete_binary_data = ''
        all_constellations = []
        for block in blocks:
        # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)
            self.g_n = interpolated_response
        # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)


        # Save the constellation points for plotting
            all_constellations.extend(x_n[1000:10000])      
        
        # Demap QPSK symbols to binary data
            binary_data = self.qpsk_demapper(x_n[1:(self.block_size // 2)])  # Assuming data is only in these bins
            complete_binary_data += binary_data
        #plotting the constellation
        self.plot_constellation(r_n[1:(self.block_size // 2)], title="Constellation Before Compensation")
        self.plot_constellation(all_constellations, title="Constellation After Compensation")
        print("Recovered Binary Data Length:", len(complete_binary_data))
        return complete_binary_data

    def plot_constellation(self, symbols, title="QPSK Constellation"):
        plt.figure(figsize=(10, 6))
        plt.scatter(np.real(symbols), np.imag(symbols), marker='.')
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title(title)
        plt.grid()
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
    
    # Example usage with simulated data
    """
    receiver = Receiver(channel_file='./files/channel.csv', received_file='./files/received_with_channel.csv', prefix_length=32, block_size=1024)
    binary_data = receiver.process_signal()
    bytes_data = receiver.binary_to_bytes(binary_data)
    filename, file_size, content = receiver.parse_bytes_data(bytes_data)
    print("Filename:", filename)
    print("File Size:", file_size)
    print(content[0:10])

    # Save the byte array to a file
    output_file_path = './files/test_image_received.tiff'
    receiver.save_file(output_file_path, content)
    """

    # Parameters
    fs =  48000
    recording_name = '0523_1237'
    OFDM_prefix_length = 512
    OFDM_block_size = 4096
    chirp_start_time = 2.0  # Example start time of chirp
    chirp_end_time = 7.0    # Example end time of chirp
    chirp_f_low = 1000
    chirp_f_high = 8000
    chirp_transmitted_path = './recordings/transmitted_linear_chirp_with_prefix_and_silence.wav'
    #received_signal_path = './recordings/'+recording_name+'.m4a'
    received_signal_path = 'recordings/0522_1309.m4a'
    
    # Initialize AnalogueSignalProcessor with the chirp signals
    asp = AnalogueSignalProcessor(chirp_transmitted_path, received_signal_path,chirp_f_low,chirp_f_high)

    # Load the chirp signals
    asp.load_audio_files()

    # Find the delay
    delay = asp.find_delay(0,10,plot=True)

    # Trim the received signal
    start_index = int(delay) # delay is an integer though
    received_signal_trimmed = asp.recv[start_index+8*fs:]

    # # Save the trimmed signal to a new file (or directly process it)
    trimmed_signal_path = './files/trimmed_received_signal_' + recording_name + '.csv'
    logging.info(f"Saving trimmed received signal to:{trimmed_signal_path}")
    pd.DataFrame(received_signal_trimmed).to_csv(trimmed_signal_path, index=False, header=False)
    
    # # Also save the trimmed signal to a WAV file
    # trimmed_signal_path_wav = './recordings/trimmed_received_signal_' + recording_name + '.wav'
    # save_as_wav(signal=received_signal_trimmed, file_path=trimmed_signal_path_wav, fs=fs)
    # logging.info(f"Saving trimmed received signal to:{trimmed_signal_path_wav}")

    # Compute the frequency response
    frequencies, frequency_response = asp.get_frequency_response(chirp_start_time, chirp_end_time, plot=True)


    # Compute the FIR filter (impulse response) from the frequency response
    # impulse_response = asp.get_FIR(plot=True, truncate=False)
    # direct_impulse_response = asp.get_direct_FIR(plot=True, truncate=False)

    # # Initialize Receiver with the trimmed signal
    print("start demodulating ")
    receiver = Receiver(channel_file =trimmed_signal_path,
                        received_file=trimmed_signal_path,
                        fs=fs,
                        frequencies=frequencies,
                        channel_impulse_response=frequency_response,
                        prefix_length=OFDM_prefix_length, block_size=OFDM_block_size)

    binary_data = receiver.process_signal()
    deomudulated_binary_path='./binaries/received_binary_'+recording_name+'.bin'
    binfile = receiver.binary_to_bin_file(binary_data, deomudulated_binary_path)

    # bytes_data = receiver.binary_to_bytes(binary_data)
    # filename, file_size, content = receiver.parse_bytes_data(bytes_data)
    # print("Filename:", filename)
    # print("File Size:", file_size)
    # print(content[0:10])

    # Save the byte array to a file
    # output_file_path = './files/test_image_received.tiff'
    # receiver.save_file(output_file_path, content)