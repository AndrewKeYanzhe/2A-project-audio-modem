#work in progress

# import first_try

"""
this code does 
[Map 2N information bits to N constellation symbols]
[Take the inverse DFT of the block of N constellation symbols]
[Copy the last k symbol in the block and append them to the beginning of the block]


"""

import numpy as np

def map_bits_to_symbols(binary_data):
    """Map 2N information bits to N constellation symbols."""
    constellation = {
        '00': complex(1, 1),
        '01': complex(-1, 1),
        '11': complex(-1, -1),
        '10': complex(1, -1)
    }
    symbols = []
    for i in range(0, len(binary_data), 2):
        bits = binary_data[i:i+2]
        symbols.append(constellation[bits])
    return symbols

def inverse_dft(symbols):
    """Take the inverse DFT of the block of N constellation symbols."""
    return np.fft.ifft(symbols)

def add_cyclic_prefix(signal, prefix_length):
    """Copy the last k symbol in the block and append them to the beginning of the block."""
    return np.concatenate((signal[-prefix_length:], signal))

def split_data_into_blocks(binary_data, block_size, prefix_length):
    """Split binary data into blocks, append cyclic prefix, and combine new blocks."""
    num_blocks = len(binary_data) // block_size
    blocks_with_prefix = []
    for i in range(num_blocks):
        start_index = i * block_size
        end_index = start_index + block_size
        block_data = binary_data[start_index:end_index]
        symbols = map_bits_to_symbols(block_data)
        # Append time-reversed, conjugate symbols to the end
        symbols_extended = np.append(symbols, np.conj(np.flip(symbols)))
        # print(symbols_extended)
        # Perform inverse DFT on the extended symbols
        time_domain_signal = inverse_dft(symbols_extended)
        # time_domain_signal = inverse_dft(symbols)
        transmitted_signal = add_cyclic_prefix(time_domain_signal, prefix_length)
        blocks_with_prefix.append(transmitted_signal)
    
    # Check if there's any remaining data to pad
    remaining_data_length = len(binary_data) % block_size
    if remaining_data_length > 0:
        remaining_data = binary_data[num_blocks * block_size:]
        remaining_data += '0' * (block_size - remaining_data_length)  # Pad with zeros
        symbols = map_bits_to_symbols(remaining_data)
        # Append time-reversed, conjugate symbols to the end
        symbols_extended = np.append(symbols, np.conj(np.flip(symbols)))
        print(len(symbols_extended))
        print(symbols_extended)
        # Perform inverse DFT on the extended symbols
        time_domain_signal = inverse_dft(symbols_extended)
        print(time_domain_signal)
        # time_domain_signal = inverse_dft(symbols)
        transmitted_signal = add_cyclic_prefix(time_domain_signal, prefix_length)
        blocks_with_prefix.append(transmitted_signal)
        
    return np.concatenate(blocks_with_prefix)

# Example binary data (replace this with your actual binary data)
binary_data = '0011101101100011'  # Example data, less than block size

# Parameters
block_size = 1024 #qpsk maps 2 bits to 1 complex
prefix_length = 32

# Split data into blocks, append cyclic prefix, and combine blocks
transmitted_signal = split_data_into_blocks(binary_data, block_size, prefix_length)

# The transmitted signal is ready to be sent over the channel
# print(transmitted_signal)
print(len(transmitted_signal))


import csv

file_name = "transmitted.csv"

np.savetxt(file_name, transmitted_signal.real, delimiter=",")

print(f"Data has been written to {file_name}.")