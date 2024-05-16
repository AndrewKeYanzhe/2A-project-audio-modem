import numpy as np

def remove_cyclic_prefix(received_signal, prefix_length):
    """Remove the cyclic prefix from the received signal."""
    return received_signal[prefix_length:]

def perform_fft(received_signal):
    """Perform FFT on the received signal."""
    return np.fft.fft(received_signal)

def extract_symbols(frequency_domain_signal, num_original_symbols):
    """Extract the original symbols from the frequency domain signal."""
    return frequency_domain_signal[:num_original_symbols]

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

def receive_signal_from_csv(file_name, block_size, prefix_length):
    """Receive the transmitted signal from a CSV file and recover the original binary data."""
    # Read received signal from CSV
    received_signal = np.loadtxt(file_name, delimiter=",")
    # Remove cyclic prefix
    signal_without_prefix = remove_cyclic_prefix(received_signal, prefix_length)
    # Perform FFT
    frequency_domain_signal = perform_fft(signal_without_prefix)
    # Determine the number of original symbols per block
    num_original_symbols = block_size // 2
    # Extract original symbols
    original_symbols = extract_symbols(frequency_domain_signal, num_original_symbols)
    # Demap symbols to bits
    binary_data = qpsk_demapper(original_symbols)
    return binary_data

# Example parameters (should match the parameters used at the transmitter)
block_size = 1024
prefix_length = 32
received_file_name = "transmitted.csv"  # Name of the CSV file containing the received signal

# Receive the signal from the CSV and recover the original binary data
recovered_binary_data = receive_signal_from_csv(received_file_name, block_size, prefix_length)
print("Recovered Binary Data:", recovered_binary_data)
