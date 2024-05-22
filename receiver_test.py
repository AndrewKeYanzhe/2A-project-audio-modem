from receiver import Receiver
impulse_response=0
receiver = Receiver(channel_file='./FIR_filters/channel.csv', received_file='./files/received_with_channel.csv',channel_impulse_response=impulse_response, prefix_length=65536, block_size=65536)
binary_data = receiver.process_signal()
bytes_data = receiver.binary_to_bytes(binary_data)
filename, file_size, content = receiver.parse_bytes_data(bytes_data)
print("Filename:", filename)
print("File Size:", file_size)
print(content[0:10])

    # Save the byte array to a file
output_file_path = './files/receiver_test.wav'
receiver.save_file(output_file_path, content)