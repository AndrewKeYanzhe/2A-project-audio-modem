def load_bin_file(file_path):
    """Load a .bin file and return its contents as a byte array."""
    with open(file_path, 'rb') as bin_file:
        byte_data = bin_file.read()
    return byte_data

def bytes_to_binary_string(byte_data):
    """Convert byte data to a binary string."""
    binary_string = ''.join(format(byte, '08b') for byte in byte_data)
    return binary_string

# Example usage
bin_file_path1 = './files/binary_blocks_test_file.bin'
bin_file_path2 = './files/received_image.bin'
byte_data1 = load_bin_file(bin_file_path1)
byte_data2 = load_bin_file(bin_file_path2)
binary_string1 = bytes_to_binary_string(byte_data1)
binary_string2 = bytes_to_binary_string(byte_data2)

# Output the binary string (first 100 bits for demonstration)
print("Binary String1 (first 500 bits):", binary_string1[0:170])
print("Binary String2 (first 500 bits):", binary_string2[0:170])

