"""
    The OFDMTransmitter class handles the process of encoding binary data into an OFDM signal,
    adding cyclic prefixes, converting audio data to binary, and handling file operations for
    transmitting and receiving signals.

    Methods:
        map_bits_to_symbols(binary_data): Maps binary data to QPSK symbols.
        inverse_dft(symbols): Computes the inverse DFT of the symbols.
        add_cyclic_prefix(signal, prefix_length): Adds a cyclic prefix to the signal.
        split_data_into_blocks(binary_data, block_size, prefix_length): Splits binary data into blocks, adds cyclic prefixes, and combines them.
        audio_to_binary(audio_data): Converts audio data to a binary string.
        file_data_to_binary_with_header(binary_data, filename): Adds a header to binary data.
        load_data(file_path): Loads data from a CSV file into a numpy array.
        load_binary_data(file_path): Loads binary data from a file.
        save_to_csv(file_path, data): Saves data to a CSV file.
        transmit_signal(binary_data, block_size, prefix_length): Encodes and transmits the binary data as an OFDM signal.
        receive_signal(channel_impulse_response, transmitted_signal): Simulates receiving the signal over a channel.(just to test)
        play_signal(signal, chirp_data, fs): Plays the combined chirp and transmitted signal.
    """

import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import chirp
from ChirpSignalGenerator import ChirpSignalGenerator
from utils import save_as_wav
import logging
from AudioProcessor import AudioProcessor
import matplotlib.pyplot as plt
from utils import save_as_wav, cut_freq_bins

import os
import math

from ldpc_function import *

class OFDMTransmitter:

    def __init__(self):
        self.constellation_points = []


    def map_bits_to_numbers(self, binary_data):
        """Map 2N information bits to N numbers using QPSK with Gray coding."""
        constellation = {
            '00': 0,
            '01': 1,
            '11': 2,
            '10': 3
        }
        numbers = []
        for i in range(0, len(binary_data), 2):
            bits = binary_data[i:i+2]
            numbers.append(constellation[bits])
        return numbers
    

    def map_numbers_to_symbols(self, numbers):
        """Map N information numbers to N constellation symbols using QPSK with Gray coding."""
        # Define the QPSK Gray coding constellation symbols
        constellation = {
            0: complex(1, 1),    # '00'
            1: complex(-1, 1),   # '01'
            2: complex(-1, -1),  # '11'
            3: complex(1, -1)    # '10'
        }    
        # List to store the constellation symbols
        symbols = []
        # Iterate over the list of numbers
        for number in numbers:
            # Append the corresponding symbol to the list
            symbols.append(constellation[number])
        return symbols

    def inverse_dft(self, symbols):
        """Take the inverse DFT of the block of N constellation symbols."""
        return np.fft.ifft(symbols)

    def add_cyclic_prefix(self, signal, prefix_length):
        """Copy the last k symbols in the block and append them to the beginning of the block."""
        return np.concatenate((signal[-prefix_length:], signal))


    def split_data_into_blocks(self, binary_data, block_size, prefix_length,fs, f_low, f_high):
        """Split binary data into blocks, append cyclic prefix, and combine new blocks."""
        # Determine the frequency bin range
        n_bins = (block_size * 2) + 2
        n_low, n_high = cut_freq_bins(f_low, f_high, fs, n_bins)
        usable_subcarriers = n_high - n_low+1
        bits_per_block = usable_subcarriers * 2

        blocks_with_prefix = []
        
        if use_ldpc:
     
            if use_pilot_tone: 
                np.random.seed(1)
                constellation_points = np.array([0,1,2,3])
                number_extended = np.random.choice(constellation_points, n_bins)
                symbols_extended = self.map_numbers_to_symbols(number_extended)
                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))

                time_domain_signal = self.inverse_dft(symbols_extended)
                
                # Add cyclic prefix
                transmitted_signal = self.add_cyclic_prefix(time_domain_signal, prefix_length)
                
                # Append the block with cyclic prefix to the list
                for i in range(5):
                    blocks_with_prefix.append(transmitted_signal)
                logging.info(f"transmitted_signal length pilot {len(transmitted_signal)}")


            print("length of binary data", len(binary_data))
            print("bits_per_block",bits_per_block) #1194
            
            ldpc_encoded_length = (bits_per_block//24)*24

            ldpc_data_length = int(ldpc_encoded_length/2)

            print("ldpc_encoded_length",ldpc_encoded_length)
                

            # Calculate the total bits needed to fit the binary data into complete OFDM blocks
            # total_bits_needed = ldpc_data_length * ((len(binary_data) + ldpc_data_length - 1) // bits_per_block)
            total_bits_needed = ldpc_data_length*math.ceil(len(binary_data)/ldpc_data_length)
            binary_data_padded = binary_data.rjust(int(total_bits_needed), '0')
 
            num_blocks = len(binary_data_padded) // ldpc_data_length

            for i in range(num_blocks):
                start_index = i * ldpc_data_length
                end_index = start_index + ldpc_data_length
                block_data = binary_data_padded[start_index:end_index]


                #convert string to list
                block_data_ldpc = encode_ldpc(list(block_data))

                #convert list to string
                block_data_ldpc = ''.join(str(x) for x in block_data_ldpc)

                # block_data_ldpc_padded=block_data_ldpc.rjust(int(bits_per_block), '0')
                block_data_ldpc_padded = block_data_ldpc + '0' * (bits_per_block - len(block_data_ldpc)) if len(block_data_ldpc) < bits_per_block else block_data_ldpc



                # Map bits to symbols

                numbers = self.map_bits_to_numbers(block_data_ldpc_padded) #containing informations
                np.random.seed(1)
                constellation_points = np.array([0,1,2,3])
                number_extended = np.random.choice(constellation_points, n_bins)

                modulus_multiplication_result = list(number_extended)

                # Perform modulus 4 multiplication for the relevant indices
                for i in range(len(numbers)):
                    corresponding_index = i + 85
                    result = (numbers[i] + number_extended[corresponding_index]) % 4
                    modulus_multiplication_result[corresponding_index] = result

                symbols_extended = self.map_numbers_to_symbols(modulus_multiplication_result)
                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))

                # Perform the inverse DFT to convert to time domain
                time_domain_signal = self.inverse_dft(symbols_extended)
                
                # Add cyclic prefix
                transmitted_signal = self.add_cyclic_prefix(time_domain_signal, prefix_length)
                
                # Append the block with cyclic prefix to the list
                blocks_with_prefix.append(transmitted_signal)
                # print("transmitted_signal length",len(transmitted_signal))

        elif use_ldpc == False:

            # Calculate the total bits needed to fit the binary data into complete OFDM blocks
            total_bits_needed = bits_per_block * ((len(binary_data) + bits_per_block - 1) // bits_per_block)
            binary_data_padded = binary_data.rjust(total_bits_needed, '0')
            num_blocks = len(binary_data_padded) // bits_per_block
            blocks_with_prefix = []

            #manipulating the first block
            if use_pilot_tone: 
                np.random.seed(1)
                constellation_points = np.array([0,1,2,3])
                number_extended = np.random.choice(constellation_points, n_bins)
                symbols_extended = self.map_numbers_to_symbols(number_extended)
                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))

                time_domain_signal = self.inverse_dft(symbols_extended)
                
                # Add cyclic prefix
                transmitted_signal = self.add_cyclic_prefix(time_domain_signal, prefix_length)
                
                # Append the block with cyclic prefix to the list
                for i in range(5):
                    blocks_with_prefix.append(transmitted_signal)

                print("transmitted_block length pilot",len(transmitted_signal))


            for i in range(num_blocks):
                start_index = i * bits_per_block
                end_index = start_index + bits_per_block
                block_data = binary_data_padded[start_index:end_index]

                # Map bits to numbers
                numbers = self.map_bits_to_numbers(block_data) #containing informations
                np.random.seed(1)
                constellation_points = np.array([0,1,2,3])
                number_extended = np.random.choice(constellation_points, n_bins)


                modulus_multiplication_result = list(number_extended)

                # Perform modulus 4 multiplication for the relevant indices
                for i in range(len(numbers)):
                    corresponding_index = i + 85
                    result = (numbers[i] + number_extended[corresponding_index]) % 4
                    modulus_multiplication_result[corresponding_index] = result

                symbols_extended = self.map_numbers_to_symbols(modulus_multiplication_result)
                symbols_extended[0] = 0
                symbols_extended[n_bins // 2] = 0

                #symbols_extended[n_bins-n_high-usable_subcarriers:n_bins-n_high] = np.conj(np.flip(symbols[:usable_subcarriers]))
                symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))

                # Perform the inverse DFT to convert to time domain
                time_domain_signal = self.inverse_dft(symbols_extended)
                
                # Add cyclic prefix
                transmitted_signal = self.add_cyclic_prefix(time_domain_signal, prefix_length)
                
                # Append the block with cyclic prefix to the list
                blocks_with_prefix.append(transmitted_signal)

        print("number of blocks", len(blocks_with_prefix))

        return np.concatenate(blocks_with_prefix)
    
    def initialize_block(self, n_bins=4096, seed=1):
        """Initialize every block for the OFDM signal."""
        np.random.seed(seed)
        constellation_points = np.array([1+1j, 1-1j, -1+1j, -1-1j])
        symbols_extended = np.random.choice(constellation_points, n_bins)
        symbols_extended[0] = 0
        symbols_extended[n_bins // 2] = 0
        symbols_extended[n_bins//2+1:] = np.conj(np.flip(symbols_extended[1:n_bins//2]))
        np.random.seed(None) 
        return symbols_extended

    
    def plot_constellation(self):
        """Plot the QPSK constellation diagram."""
        if not self.constellation_points:
            print("No constellation points to plot.")
            return

        plt.scatter(np.real(self.constellation_points), np.imag(self.constellation_points))
        plt.axhline(0, color='black', lw=0.5)
        plt.axvline(0, color='black', lw=0.5)
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.title('QPSK Constellation')
        plt.grid()
        plt.show()



    def get_constellation_label(self, point):
        """Get label for constellation point."""
        if point == complex(1, 1):
            return 0
        elif point == complex(-1, 1):
            return 1
        elif point == complex(-1, -1):
            return 2
        elif point == complex(1, -1):
            return 3
        else:
            return -1  # Unknown point



    def audio_to_binary(self, audio_data):
        """Convert audio data to a binary string.this change the data from raw bytes to binary string."""
        binary_data = ''.join(format(byte, '08b') for byte in audio_data)
        return binary_data

    def file_data_to_binary_with_header(self, binary_data, filename):
        """Convert file binary data to a binary string with a header."""
        file_size = len(binary_data)
        header = f"\0\0{filename}\0\0{file_size*8}\0\0"
        header_binary = ''.join(format(ord(char), '08b') for char in header)
        binary_data_string = ''.join(format(byte, '08b') for byte in binary_data)
        complete_binary_data = header_binary + binary_data_string
        return complete_binary_data

    def load_data(self, file_path):
        """Load data from a CSV file into a numpy array."""
        return pd.read_csv(file_path, header=None).values.flatten()

    def load_binary_data(self, file_path):
        """Load binary data from a file."""
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        return binary_data

    def save_to_csv(self, file_path, data):
        """Save data to a CSV file."""
        np.savetxt(file_path, data, delimiter=',', fmt='%1.4f')
        print(f"Data has been written to {file_path}.")

    def transmit_signal(self, binary_data, block_size, prefix_length,fs, f_low, f_high):
        """Encode and transmit the binary data as an OFDM signal."""
        transmitted_signal = self.split_data_into_blocks(binary_data, block_size, prefix_length,fs, f_low, f_high)
        transmitted_signal = transmitted_signal.real
        print(f"Length of transmitted signal: {len(transmitted_signal)}")
        return transmitted_signal

    def receive_signal(self, channel_impulse_response, transmitted_signal):
        """Simulate receiving the signal over a channel."""
        received_signal = np.convolve(transmitted_signal, channel_impulse_response, mode='full')
        return received_signal

    def play_signal(self, signal, chirp_data, fs, save_path=None):
        """Play the combined chirp and transmitted signal."""
        _,normalized_signal = self.normalize_signal(signal)
        # combined_signal = np.concatenate((chirp_data, normalized_signal))
        combined_signal = np.concatenate((0.1*chirp_data, normalized_signal, 0.1*chirp_data))
        # sd.play(combined_signal, samplerate=fs)
        sd.wait()
        if save_path:
            save_as_wav(combined_signal, save_path, fs)
        
    def just_play_signal(self, signal, fs):
        sd.play(signal, samplerate=fs)
        sd.wait()
        
    def normalize_signal(self, signal):
        """Normalizes the signal to 16-bit integer values and floating-point values."""
        signal_int = np.int16(signal / np.max(np.abs(signal)) * 32767)
        signal_normalized = signal / np.max(np.abs(signal))
        return signal_int, signal_normalized

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Parameters
    fs = 48000
    block_size = (4096-2)//2
    prefix_length = 1024
    suffix_length = 1024
    chirp_name = '1k_8k_0523_suffix'

    # Example usage
    transmitter = OFDMTransmitter()

    # Load the binary data from file
    # transmitted_binary_path = 'binaries/P1017125.tif'
    transmitted_binary_path = 'text/article_2_iceland.txt'
    logging.info(f"Loading binary data from {transmitted_binary_path}.")
    data = transmitter.load_binary_data(transmitted_binary_path)

    recording_name = os.path.splitext(os.path.basename(transmitted_binary_path))[0]
    recording_name_with_extension = os.path.basename(transmitted_binary_path)


    use_pilot_tone = True
    use_ldpc = True
    use_header=True

    # Convert file data to binary with header
    #filename = "transmitted_5.26pm.wav"
    ## if with header
    #binary_data = transmitter.file_data_to_binary_with_header(data, filename)
    #if withouth header
    if use_header:
        binary_data = transmitter.file_data_to_binary_with_header(data, recording_name_with_extension)
    else:
        binary_data = transmitter.audio_to_binary(data)
    


    # print(type(binary_data))

    # use_header = False

    # if use_header:
    #     binary_data = "0"*16+binary_data


    # Transmit the signal
    transmitted_signal = transmitter.transmit_signal(binary_data, block_size, prefix_length,fs, 1000, 8000) # Don't worry about this frequency range for now

    # Save the transmitted signal to a CSV file
    output_csv_path = './files/transmitted_data_' + recording_name + '.csv'
    transmitter.save_to_csv(output_csv_path, transmitted_signal)
    logging.info(f"Transmitted signal has been saved to {output_csv_path}.")


    # Generate the chirp signal with ChirpSignalGenerator and save it
    generator = ChirpSignalGenerator(t_chirp=1.365333333 ,f_low=761.72, f_high=8824.22,
                                     prefix_size=prefix_length, suffix_size=suffix_length)
    generator.generate_chirp_signal()
    chirp_path = 'chirps/' + chirp_name + '.wav'
    generator.save_as_wav(chirp_path)

    # Load the chirp data from the saved file
    chirp_data, chirp_sr = librosa.load(chirp_path, sr=None)

    # Play the combined transmitted signal with chirp
    save_path ='recordings/transmitted_' + recording_name + '_pilot'+str(int(use_pilot_tone))+'_ldpc'+str(int(use_ldpc)) +'.wav'
    save_path =f'recordings/transmitted_{recording_name}_pilot{int(use_pilot_tone)}_ldpc{int(use_ldpc)}_header{int(use_header)}.wav'
    transmitter.play_signal(transmitted_signal,chirp_data, fs, save_path=save_path)
    logging.info(f"Saving the combined transmitted signal with chirp to{save_path}.")


