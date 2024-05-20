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
    
    for i, chunk in enumerate(chunks):
        # print("Chunk", i, ":", chunk)
        start_index = f_low * num_bins // fs
        end_index = f_high * num_bins // fs
        chunk = chunk[start_index:end_index]
        # print(chunk[start_index:end_index])
        
        # count the number of 1s in the chunk
        num_ones = chunk.count('1')
        num_zeros = chunk.count('0')
        if num_ones > num_zeros:
            print(f"Chunk {i} is decoded as 1")
        else:
            print(f"Chunk {i} is decoded as 0")
    return chunks
    
    


# Example usage
bin_file_path1 = './files/binary_blocks_test_file.bin'
bin_file_path2 = './files/received_image.bin'
byte_data1 = load_bin_file(bin_file_path1)
byte_data2 = load_bin_file(bin_file_path2)
binary_string1 = bytes_to_binary_string(byte_data1)
binary_string2 = bytes_to_binary_string(byte_data2)

# # Output the binary string (first 100 bits for demonstration)
# print("Binary String1 (first 500 bits):", binary_string1[0:2000])
# print("Binary String2 (first 500 bits):", binary_string2[0:2000])

# Extract OFDM bins
extract_OFDM_bins(binary_string2[0:100000])
