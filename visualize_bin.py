from matplotlib import pyplot as plt

def load_bin_file(file_path):
    """Load a .bin file and return its contents as a byte array."""
    with open(file_path, 'rb') as bin_file:
        byte_data = bin_file.read()
    return byte_data

def bytes_to_binary_string(byte_data):
    """Convert byte data to a binary string."""
    binary_string = ''.join(format(byte, '08b') for byte in byte_data)
    return binary_string


def extract_OFDM_bins(binary_string, num_bins=1024, fs=48000, f_low=20, f_high=8000):
    
    # segment the binary string into chunks of 1022 bits
    chunk_size = 1022
    num_chunks = len(binary_string) // chunk_size
    print("num_chunks:", num_chunks)
    chunks = [binary_string[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    start_index = f_low * num_bins // fs
    end_index = f_high * num_bins // fs
    print("start_index:", start_index)
    print("end_index:", end_index)
    
    # keep only the bins corresponding to the specified frequency range
    for i, chunk in enumerate(chunks):
        chunks[i] = chunk[start_index:end_index]

    return chunks
    
    


# Example usage
bin_file_path1 = 'binaries/transmitted_bin_0520_1541.bin'
bin_file_path2 = 'binaries/received_binary_0520_1541.bin'
byte_data1 = load_bin_file(bin_file_path1)
byte_data2 = load_bin_file(bin_file_path2)
binary_string1 = bytes_to_binary_string(byte_data1)
binary_string2 = bytes_to_binary_string(byte_data2)

print(len(binary_string1)/4094)
print(len(binary_string2)/4096)


transmitted_chunks = extract_OFDM_bins(binary_string1)
actual_block_nums = len(transmitted_chunks)
received_chunks = extract_OFDM_bins(binary_string2)[0:actual_block_nums]

# concatenate the chunks into a single binary string
transmitted_binary_string = ''.join(transmitted_chunks[-2:])
received_binary_string = ''.join(received_chunks[-2:])

print(len(transmitted_binary_string), len(received_binary_string))

# calculate the bit error rate
num_errors = sum([1 for i in range(len(transmitted_binary_string)) if transmitted_binary_string[i] != received_binary_string[i]])
ber = num_errors / len(transmitted_binary_string)
print(f"Bit error rate: {ber:.2f}")

# plot the ber as a function of OFDM block number
ber_list = []
for i in range(actual_block_nums):
    num_errors = sum(1 for j in range(len(transmitted_chunks[i])) if transmitted_chunks[i][j] != received_chunks[i][j])
    ber = num_errors / len(transmitted_chunks[i])
    ber_list.append(ber)
plt.plot(ber_list)
plt.xlabel('OFDM block number')
plt.ylabel('Bit error rate')
plt.title('Bit error rate vs OFDM block number')
plt.show()