import numpy as np
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file into a numpy array."""
    return pd.read_csv(file_path, header=None).values.flatten()

def remove_cyclic_prefix(signal, prefix_length, block_size):
    """Remove the cyclic prefix of an OFDM signal."""
    num_blocks = len(signal) // (block_size + prefix_length)
    blocks = []
    for i in range(num_blocks):
        start_index = i * (block_size + prefix_length) + prefix_length
        end_index = start_index + block_size
        blocks.append(signal[start_index:end_index])
    return blocks

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
received_signal = load_data('./files/received.csv')

# Parameters
prefix_length = 32
block_size = 1024

# Remove cyclic prefix and get blocks
blocks = remove_cyclic_prefix(received_signal, prefix_length, block_size)

# Estimate channel frequency response
g_n = apply_fft(channel_impulse_response, block_size)

# Process each block
complete_binary_data = ''
for block in blocks:
    # Apply FFT to the block
    r_n = apply_fft(block, block_size)

    # Compensate for the channel effects
    x_n = channel_compensation(r_n, g_n)

    # Demap QPSK symbols to binary data
    binary_data = qpsk_demapper(x_n[1:(block_size//2)])  # Assuming data is only in these bins
    complete_binary_data += binary_data

print("Recovered Binary Data Length:", len(complete_binary_data))

def binary_to_bytes(binary_data):
    # Pad the binary string to make its length a multiple of 8
    padded_binary = binary_data + '0' * ((8 - len(binary_data) % 8) % 8)
    byte_array = bytearray()
    for i in range(0, len(padded_binary), 8):
        byte_part = padded_binary[i:i+8]
        byte_array.append(int(byte_part, 2))
    return bytes(byte_array)



def parse_bytes_data(bytes_data):
    # Splitting the data at null bytes
    parts = bytes_data.split(b'\0')
    filename = parts[0].decode('utf-8')
    start_of_image_data = bytes_data.find(b'\0', bytes_data.find(b'\0') + 1) + 1
    file_size = int(bytes_data[bytes_data.find(b'\0') + 1 : start_of_image_data - 1])
    
    file_content = bytes_data[start_of_image_data:start_of_image_data + file_size]
    
    return filename, file_size, file_content

# Convert demapped binary data to bytes
bytes_data = binary_to_bytes(complete_binary_data)


# Parse the bytes data to extract filename, size, and content
filename, file_size, content = parse_bytes_data(bytes_data)
print("Filename:", filename)
print("File Size:", file_size)  
print(content[0:10])


# Save the byte array to a TIFF file
file_path = './files/test_linear_chirp.wav'
with open(file_path, 'wb') as file:
    file.write(content)

print(f"File has been saved to {file_path}. Please check the file to see if the image is correctly reconstructed.")

