import numpy as np
import pandas as pd
from channel_estimator import AnalogueSignalProcessor
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
    def __init__(self, channel_file, received_file, channel_impulse_response, prefix_length, block_size):
        self.channel_file = channel_file
        self.channel_impulse_response = channel_impulse_response
        self.received_file = received_file
        self.prefix_length = prefix_length
        self.block_size = block_size
        #self.channel_impulse_response = None
        self.received_signal = None
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
        return r_n / g_n

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
            binary_data += bits
        return binary_data

    def process_signal(self):
        """Process the received signal to recover binary data."""
        # Load the channel impulse response and the received signal
        #self.channel_impulse_response = self.load_data(self.channel_file)(这是原来的代码，channel。csv情况下)

        self.received_signal = self.load_data(self.received_file)

        # Remove cyclic prefix and get blocks
        blocks = self.remove_cyclic_prefix(self.received_signal)

        # Estimate channel frequency response
        self.g_n = self.apply_fft(self.channel_impulse_response, self.block_size)

        # Process each block
        complete_binary_data = ''
        for block in blocks:
            # Apply FFT to the block
            r_n = self.apply_fft(block, self.block_size)

            # Compensate for the channel effects
            x_n = self.channel_compensation(r_n, self.g_n)

            # Demap QPSK symbols to binary data
            binary_data = self.qpsk_demapper(x_n[1:(self.block_size // 2)])  # Assuming data is only in these bins
            complete_binary_data += binary_data

        print("Recovered Binary Data Length:", len(complete_binary_data))
        return complete_binary_data

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

    def save_file(self, file_path, content):
        """Save the content to a file."""
        with open(file_path, 'wb') as file:
            file.write(content)
        print(f"File has been saved to {file_path}. Please check the file to see if the image is correctly reconstructed.")

# Example usage
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

# Example usage with AnalogueSignalProcessor

# Initialize AnalogueSignalProcessor with the chirp signals
chirp_transmitted_path = './files/transmitted_chirp.wav'
received_signal_path = './files/received_signal_with_chirp.wav'
asp = AnalogueSignalProcessor(chirp_transmitted_path, received_signal_path)

# Load the chirp signals
asp.load_audio_files()

# Find the delay
delay = asp.find_delay(plot=True)

# Trim the received signal
start_index = int(delay)
received_signal_trimmed = asp.recv[start_index:]

# Save the trimmed signal to a new file (or directly process it)
trimmed_signal_path = './files/trimmed_signal.csv'
pd.DataFrame(received_signal_trimmed).to_csv(trimmed_signal_path, index=False, header=False)

# Compute the frequency response
chirp_start_time = 2.0  # Example start time of chirp
chirp_end_time = 7.0    # Example end time of chirp
frequencies, frequency_response = asp.get_frequency_response(chirp_start_time, chirp_end_time, plot=True)

# Compute the FIR filter (impulse response) from the frequency response
impulse_response = asp.get_FIR(plot=True, truncate=True)


# Initialize Receiver with the trimmed signal
receiver = Receiver(channel_impulse_response=impulse_response, received_file=trimmed_signal_path, prefix_length=32, block_size=1024)
binary_data = receiver.process_signal()
bytes_data = receiver.binary_to_bytes(binary_data)
filename, file_size, content = receiver.parse_bytes_data(bytes_data)
print("Filename:", filename)
print("File Size:", file_size)
print(content[0:10])

# Save the byte array to a file
output_file_path = './files/test_image_received.tiff'
receiver.save_file(output_file_path, content)