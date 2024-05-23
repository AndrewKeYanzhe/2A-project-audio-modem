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

class OFDMTransmitter:

    def __init__(self):
        self.constellation_points = []


    def map_bits_to_symbols(self, binary_data):
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

    def inverse_dft(self, symbols):
        """Take the inverse DFT of the block of N constellation symbols."""
        return np.fft.ifft(symbols)

    def add_cyclic_prefix(self, signal, prefix_length):
        """Copy the last k symbols in the block and append them to the beginning of the block."""
        return np.concatenate((signal[-prefix_length:], signal))
    #block_size here is 
   
    # (real_blocksize-2)/2
    def split_data_into_blocks(self, binary_data, block_size, prefix_length):
        """Split binary data into blocks, append cyclic prefix, and combine new blocks."""
        total_bits_needed = (block_size * 2) * ((len(binary_data) + block_size * 2 - 1) // (block_size * 2))
        binary_data_padded = binary_data.ljust(total_bits_needed, '0')
        num_blocks = len(binary_data_padded) // (block_size * 2)
        blocks_with_prefix = []

        for i in range(num_blocks):
            start_index = i * block_size * 2
            end_index = start_index + block_size * 2
            block_data = binary_data_padded[start_index:end_index]
            symbols = self.map_bits_to_symbols(block_data)
            self.constellation_points.extend(symbols)  # Save constellation points for visualization
            symbols_extended = np.zeros(block_size * 2 + 2, dtype=complex)
            symbols_extended[1:block_size+1] = symbols[:block_size]
            symbols_extended[block_size+2:] = np.conj(np.flip(symbols[:block_size]))
            time_domain_signal = self.inverse_dft(symbols_extended)
            transmitted_signal = self.add_cyclic_prefix(time_domain_signal, prefix_length)
            blocks_with_prefix.append(transmitted_signal)
        
        return np.concatenate(blocks_with_prefix)
    
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
        header = f"{filename}\0{file_size}\0"
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

    def transmit_signal(self, binary_data, block_size, prefix_length):
        """Encode and transmit the binary data as an OFDM signal."""
        transmitted_signal = self.split_data_into_blocks(binary_data, block_size, prefix_length)
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
        combined_signal = np.concatenate((chirp_data, normalized_signal))
        sd.play(combined_signal, samplerate=fs)
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
    prefix_length = 512
    recording_name = '0523_1237'
    chirp_name = '1k_8k_0523'

    # Example usage
    transmitter = OFDMTransmitter()

    # Load the binary data from file
    transmitted_binary_path = './binaries/transmitted_bin_0520_1541.bin'
    logging.info(f"Loading binary data from {transmitted_binary_path}.")
    data = transmitter.load_binary_data(transmitted_binary_path)

    # Convert file data to binary with header
    #filename = "transmitted_5.26pm.wav"
    ## if with header
    #binary_data = transmitter.file_data_to_binary_with_header(data, filename)
    #if withouth header
    binary_data = transmitter.audio_to_binary(data)

    # Transmit the signal
    transmitted_signal = transmitter.transmit_signal(binary_data, block_size, prefix_length)

    # Save the transmitted signal to a CSV file
    output_csv_path = './files/transmitted_data_' + recording_name + '.csv'
    transmitter.save_to_csv(output_csv_path, transmitted_signal)
    logging.info(f"Transmitted signal has been saved to {output_csv_path}.")

    # Plot the QPSK constellation diagram
    transmitter.plot_constellation()

    # Generate the chirp signal with ChirpSignalGenerator and save it
    generator = ChirpSignalGenerator(f_low=1000, f_high=8000)
    generator.generate_chirp_signal()
    chirp_path = 'chirps/' + chirp_name + '.wav'
    generator.save_as_wav(chirp_path)

    # Load the chirp data from the saved file
    chirp_data, chirp_sr = librosa.load(chirp_path, sr=None)

    # Play the combined transmitted signal with chirp
    save_path ='recordings/transmitted_signal_with_chirp_' + recording_name + '.wav'
    transmitter.play_signal(transmitted_signal,chirp_data, fs, save_path=save_path)
    logging.info(f"Saving the combined transmitted signal with chirp to{save_path}.")
    # Simulate receiving the signal

    # channel_impulse_response = transmitter.load_data('./files/channel.csv')
    # received_signal = transmitter.receive_signal(channel_impulse_response, transmitted_signal)

    # # Save the received signal to a CSV file
    # receive_csv_path = './files/received_with_channel.csv'
    # transmitter.save_to_csv(receive_csv_path, received_signal)
    # audio_processor = AudioProcessor(save_path)
    # csv_file_path = './files/no_channel.csv'
    # audio_processor.load_audio()
    # audio_processor.save_to_csv(csv_file_path)

