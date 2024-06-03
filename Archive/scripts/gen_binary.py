import os

def generate_binary_file(file_path, file_size):
    # Generate random bytes
    random_bytes = os.urandom(file_size)
    
    # Write to a file
    with open(file_path, 'wb') as file:
        file.write(random_bytes)

# Usage example: create a binary file 'output.bin' of size 1024 bytes (1 KB)
generate_binary_file('files/1535pm.bin', 1022*32)
