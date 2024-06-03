import numpy as np
import librosa
import pandas as pd
import sounddevice as sd
def map_bits_to_symbols(binary_data):
    """Map 2N information bits to N constellation symbols using QPSK with Gray coding."""
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
    total_bits_needed = (block_size * 2) * ((len(binary_data) + block_size * 2 - 1) // (block_size * 2))
    
    # Zero-pad the binary data to ensure it fits into complete blocks
    binary_data_padded = binary_data.ljust(total_bits_needed, '0')
    
    num_blocks = len(binary_data_padded) // (block_size * 2)  # 2 bits per symbol, we run IDFT to 1024 symbols
    blocks_with_prefix = []

    for i in range(num_blocks):
        start_index = i * block_size * 2
        end_index = start_index + block_size * 2
        block_data = binary_data_padded[start_index:end_index]
        symbols = map_bits_to_symbols(block_data)
        
        # Prepare symbols array for IDFT
        symbols_extended = np.zeros(1024, dtype=complex)
        symbols_extended[1:512] = symbols[:511]
        symbols_extended[513:] = np.conj(np.flip(symbols[:511]))
        
        # Perform inverse DFT on the extended symbols
        time_domain_signal = inverse_dft(symbols_extended)
        transmitted_signal = add_cyclic_prefix(time_domain_signal, prefix_length)
        blocks_with_prefix.append(transmitted_signal)

    
        
    return np.concatenate(blocks_with_prefix)

def audio_to_binary(audio_data):
    """Convert audio data to a binary string."""
    binary_data = ''.join(format(byte, '08b') for byte in audio_data.astype(np.uint8))
    return binary_data


def file_data_to_binary_with_header(binary_data, filename):
    """Convert file binary data to a binary string with a header."""
    # Create the header
    file_size = len(binary_data)
    header = f"{filename}\0{file_size}\0"
    header_binary = ''.join(format(ord(char), '08b') for char in header)
    
    # Convert binary data to binary string
    binary_data_string = ''.join(format(byte, '08b') for byte in binary_data)
    
    # Combine the header and binary data
    complete_binary_data = header_binary + binary_data_string
    return complete_binary_data

def load_data(file_path):
    """Load data from a CSV file into a numpy array."""
    return pd.read_csv(file_path, header=None).values.flatten()

def load_binary_data(file_path):
    """Load binary data from a file."""
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return binary_data


file_path1 = './recordings/transmitted_chirp_1k_3k_2s.wav'
data = load_binary_data(file_path1)




    
# Load the audio file
#audio_data, sr = librosa.load(file_path1, sr=None)
    
# Convert audio data to binary string
filename = "linear_chirp.wav"
#binary_data = audio_to_binary_with_header(audio_data, filename)
binary_data = file_data_to_binary_with_header(data, filename)
# Parameters
block_size = 511  # Each block should contain 511 symbols
prefix_length = 32
    
    # Split data into blocks, append cyclic prefix, and combine blocks
transmitted_signal = split_data_into_blocks(binary_data, block_size, prefix_length)
transmitted_signal = transmitted_signal.real
transmitted_signal_type = type(transmitted_signal) 
print(f"Type of transmitted_signal: {transmitted_signal_type}")

#amplification
#transmitted_signal = transmitted_signal * 10
    
# The transmitted signal is ready to be sent over the channel
print(f"Length of transmitted signal: {len(transmitted_signal)}")
    
    # Save the transmitted signal to a CSV file
output_csv_path = './files/transmitted_data.csv'
file_name = "transmitted.csv"
np.savetxt(output_csv_path, transmitted_signal, delimiter=',', fmt='%1.4f')
print(f"Data has been written to {file_name}.")




# assume we know the channel impulse response


channel_impulse_response = load_data('./files/channel.csv')
received_signal = np.convolve(transmitted_signal, channel_impulse_response, mode='full')

receive_csv_path = './files/received.csv'

file_name2 = "received.csv"
np.savetxt(receive_csv_path, received_signal, delimiter=',', fmt='%1.4f')
print(f"Data has been written to {file_name2}.")




# Play the transmitted signal
filepath3='./recordings/linear_chirp_with_prefix_and_silence.wav'
chirp_data, chirp_sr = librosa.load(filepath3, sr=None)  # Load chirp with its original sample rate

# Ensure both signals have the same sample rate
fs = 48000  # Target sample rate


audio_list=list(chirp_data)
# Concatenate the transmitted signal with the chirp sequence
combined_signal = np.concatenate((chirp_data,transmitted_signal))
audio_list=list(combined_signal)
sd.play(audio_list, samplerate=fs)
sd.wait()

