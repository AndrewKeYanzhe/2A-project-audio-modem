from matplotlib import pyplot as plt
from utils import cut_freq_bins

def load_bin_file(file_path):
    """Load a .bin file and return its contents as a byte array."""
    with open(file_path, 'rb') as bin_file:
        byte_data = bin_file.read()
    return byte_data

def bytes_to_binary_string(byte_data):
    """Convert byte data to a binary string."""
    binary_string = ''.join(format(byte, '08b') for byte in byte_data)
    return binary_string

def bin_string2chunks(binary_string, already_freq_truncated=True,
                      f_low=1000, f_high=8000, num_bins=4096, fs=48000):
    """
    Convert a binary string to a list of chunks, each corresponding the information bits in an OFDM block.
    """
    
    start_index, end_index = cut_freq_bins(f_low, f_high, fs, num_bins)
    
    if not already_freq_truncated:
        # chunk size is the number of bins
        chunk_size = num_bins
        # The number of chunks is the number of OFDM blocks
        num_chunks = len(binary_string) // chunk_size
        # Extract the chunks - only the bins corresponding to the specified frequency range
        chunks = [binary_string[i*chunk_size+start_index: i*chunk_size+end_index+1] for i in range(num_chunks)]
    
    if already_freq_truncated:
        # Already truncated to the frequency range, so chunk size is end_index - start_index
        chunks_size = end_index - start_index + 1
        num_chunks = len(binary_string) // chunk_size
        # no need to truncate again
        chunks = [binary_string[i*chunks_size:(i+1)*chunks_size] for i in range(num_chunks)]
    
    return num_chunks, chunks
       
def binary_to_bytes(binary_data):
    """Convert binary data to bytes."""
    # Pad the binary string to make its length a multiple of 8
    padded_binary = binary_data + '0' * ((8 - len(binary_data) % 8) % 8)
    byte_array = bytearray()
    for i in range(0, len(padded_binary), 8):
        byte_part = padded_binary[i:i + 8]
        byte_array.append(int(byte_part, 2))
    return bytes(byte_array) 

def save_file(file_path, content):
    """Save the content to a file."""
    with open(file_path, 'wb') as file:
        file.write(content)
    print(f"File has been saved to {file_path}. Please check the file to see if the image is correctly reconstructed.")


    
if __name__ == "__main__":


    # Example usage
    
    # Parameters
    f_low = 1000
    f_high = 8000
    num_bins = 4096
    fs = 48000
    time = '0525_1558'
    
    bin_file_path1 = 'binaries/transmitted_bin_0520_1541.bin'
    bin_file_path2 = './binaries/received_binary_0523_1300.bin'
    byte_data1 = load_bin_file(bin_file_path1)
    byte_data2 = load_bin_file(bin_file_path2)
    binary_string1 = bytes_to_binary_string(byte_data1)
    binary_string2 = bytes_to_binary_string(byte_data2)

    num_transmitted_chunks, transmitted_chunks = bin_string2chunks(binary_string1,
                                                                   already_freq_truncated=False,
                                                                   f_low=f_low, f_high=f_high,
                                                                   num_bins=num_bins, fs=fs)
    
    num_received_chunks, received_chunks = bin_string2chunks(binary_string2,
                                                             already_freq_truncated=True,
                                                             f_low=f_low, f_high=f_high,
                                                             num_bins=num_bins, fs=fs)

    # concatenate the chunks into a single binary string
    transmitted_binary_string = ''.join(transmitted_chunks)
    received_binary_string = ''.join(received_chunks)
    
    # save the received binary strings to a file
    byte_data2 = binary_to_bytes(received_binary_string)
    save_file('text/received_file_'+time+'.txt', byte_data2)
    
    
    
    
    
    
    ###################### Error rate calculation ######################

    # print(len(transmitted_binary_string), len(received_binary_string))

    # # calculate the bit error rate
    # num_errors = sum([1 for i in range(len(transmitted_binary_string)) if transmitted_binary_string[i] != received_binary_string[i]])
    # ber = num_errors / len(transmitted_binary_string)
    # print(f"Bit error rate: {ber:.2f}")

    # # plot the ber as a function of OFDM block number
    # ber_list = []
    # for i in range(actual_block_nums):
    #     num_errors = sum(1 for j in range(len(transmitted_chunks[i])) if transmitted_chunks[i][j] != received_chunks[i][j])
    #     ber = num_errors / len(transmitted_chunks[i])
    #     ber_list.append(ber)
    # plt.plot(ber_list)
    # plt.xlabel('OFDM block number')
    # plt.ylabel('Bit error rate')
    # plt.title('Bit error rate vs OFDM block number')
    # plt.show()