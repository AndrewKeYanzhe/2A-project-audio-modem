from OFDMTransmitter import OFDMTransmitter

# Example usage
transmitter = OFDMTransmitter()

# Load the binary data from file
file_path1 = './recordings/transmitted_linear_chirp_with_prefix_and_silence.wav'
data = transmitter.load_binary_data(file_path1)

# Convert file data to binary with header
filename = "transmitted_linear_chirp_with_prefix_and_silence.wav"

binary_data = transmitter.file_data_to_binary_with_header(data, filename)
# Transmit the signal
block_size = 32767 #(real_blocksize-2)/2
prefix_length = 65536
transmitted_signal = transmitter.transmit_signal(binary_data, block_size, prefix_length)

# Save the transmitted signal to a CSV file
output_csv_path = './files/transmitted_data.csv'
transmitter.save_to_csv(output_csv_path, transmitted_signal)

# Generate the chirp signal with ChirpSignalGenerator and save it
# generator = ChirpSignalGenerator()
# generator.generate_chirp_signal()
# generator.save_as_wav('recordings/transmitted_linear_chirp_with_prefix_and_silence.wav')

# # Load the chirp data from the saved file
# chirp_data, chirp_sr = librosa.load('recordings/transmitted_linear_chirp_with_prefix_and_silence.wav', sr=None)

# # Play the combined transmitted signal with chirp
# fs = 48000
# transmitter.play_signal(transmitted_signal,chirp_data, fs)

# Simulate receiving the signal

channel_impulse_response = transmitter.load_data('./FIR_filters/channel.csv')
received_signal = transmitter.receive_signal(channel_impulse_response, transmitted_signal)

# # Save the received signal to a CSV file
receive_csv_path = './files/received_with_channel.csv'
transmitter.save_to_csv(receive_csv_path, received_signal)
