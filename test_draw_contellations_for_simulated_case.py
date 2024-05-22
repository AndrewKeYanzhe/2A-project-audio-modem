from receiver import Receiver

receiver = Receiver(channel_file='./files/channel.csv', received_file='./files/received_with_channel.csv',channel_impulse_response=0, prefix_length=32, block_size=1024)
binary_data = receiver.process_signal()
bytes_data = receiver.binary_to_bytes(binary_data)
filename, file_size, content = receiver.parse_bytes_data(bytes_data)
print("Filename:", filename)
print("File Size:", file_size)
print(content[0:10])

# Save the byte array to a file
output_file_path = './files/test_image_received.tiff'
receiver.save_file(output_file_path, content)