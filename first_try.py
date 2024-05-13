import numpy as np
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file into a numpy array."""
    return pd.read_csv(file_path, header=None).values.flatten()

def remove_cyclic_prefix(signal, prefix_length, block_size):
    """Remove the cyclic prefix of an OFDM signal."""
    num_blocks = len(signal) // (block_size + prefix_length)
    return np.concatenate([signal[i * (block_size + prefix_length) + prefix_length:(i + 1) * (block_size + prefix_length)] for i in range(num_blocks)])

def apply_fft(signal, n):
    """Apply FFT to the signal."""
    return np.fft.fft(signal, n=n)

def channel_compensation(r_n, g_n):
    """Compensate for the channel effects in the frequency domain."""
    return r_n / g_n

def qpsk_demapper(compensated_symbols):
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

# Load the channel impulse response and the received signal
channel_impulse_response = load_data('channel.csv')
received_signal = load_data('file1.csv')

# Parameters
prefix_length = 32
block_size = 1024

# Remove cyclic prefix
received_signal_without_prefix = remove_cyclic_prefix(received_signal, prefix_length, block_size)

# Apply FFT to the received signal
r_n = apply_fft(received_signal_without_prefix, block_size)

# Estimate channel frequency response
g_n = apply_fft(channel_impulse_response, block_size)

# Compensate for the channel effects
x_n = channel_compensation(r_n, g_n)

# Demap QPSK symbols to binary data
binary_data = qpsk_demapper(x_n[1:(block_size//2)])  # Assuming data is only in these bins

print("Recovered Binary Data:", binary_data)

def binary_to_bytes(binary_data):
    # Pad the binary string to make its length a multiple of 8
    padded_binary = binary_data + '0' * ((8 - len(binary_data) % 8) % 8)
    byte_array = bytearray()
    for i in range(0, len(padded_binary), 8):
        byte_part = padded_binary[i:i+8]
        byte_array.append(int(byte_part, 2))
    return bytes(byte_array)



def parse_binary_data(binary_data):
    # Splitting the data at null bytes
    parts = binary_data.split(b'\0')
    filename = parts[0].decode('utf-8')
    file_size = int(parts[1].decode('utf-8'))
    file_content = ''
    for part in parts[2:]:
        print(part)
        file_content += part.decode('utf-8')

    return filename, file_size, file_content

# Convert demapped binary data to bytes
bytes_data = binary_to_bytes(binary_data)

# Parse the bytes data to extract filename, size, and content
filename, file_size, content = parse_binary_data(bytes_data)
print("Filename:", filename)
print("File Size:", file_size)
print("Content:", content.decode('utf-8'))  # Assuming content is text and UTF-8 decodable

with open('files/3829010287.tiff', 'wb') as file:
    file.write(content)

