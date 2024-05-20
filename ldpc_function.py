"""
if src/c_ldpc.c and src/results2csv.c dont exist, use the following instructions

on mac:
1. install command line tools
2. create 'bin' folder
3. run the following as stated in readme
gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c
gcc -o bin/results2csv src/results2csv.c

Usage
-------------------------

the transmitted block is 2x the size of the data block

the transmitted block has to be a multiple of 24 (see /py/ldpc.py)
504 is chosen so that it fits within the 511 ofdm data block

hence the data block has to be of length 252

encode_ldpc(data) returns the 504 length transmission block
decode_ldpc(signal) does the inverse
"""


import random
import py.ldpc as ldpc
import numpy as np

# Create an LDPC code object
c = ldpc.code(standard = '802.16', rate = '1/2', z=21, ptype='A')

def flip_bits(bit_list, percentage=5):
    # Calculate the number of bits to flip
    num_bits_to_flip = int(len(bit_list) * (percentage / 100))
    
    # Generate a list of unique random indices to flip
    indices_to_flip = random.sample(range(len(bit_list)), num_bits_to_flip)
    
    # Flip the bits at the selected indices
    for index in indices_to_flip:
        bit_list[index] = 1 - bit_list[index]  # Flip the bit: 0 becomes 1, 1 becomes 0
        # bit_list[index] = - bit_list[index]  # Flip the bit: 0 becomes 1, 1 becomes 0

    return bit_list


def encode_ldpc(data): #data must be of length 324
    x = c.encode(data)
    # print(len(x), "seems to be twice as long")
    # print("x\n",x)
    return x


def decode_ldpc(signal):
    y = 10 * (.5 - signal)

    # Decode the received signal
    app, it = c.decode(y)

    # Check the number of iterations taken by the decoder
    # print("iterations",it)  # Output: 0

    output = np.where(app < 0, 1, 0)
    output=output[0:252]

    return output



if __name__ == "__main__":
    # Check the standard used
    print(c.standard)  # Output: '802.11n'

    # Generate a random message
    # data = np.random.randint(0, 2, c.K)
    data = [0,0,1,1]

    data = np.pad(data, (0, 252 - len(data)), 'constant', constant_values=(0, 0)) #length must be 252

    print("padded data\n",data)

    y=encode_ldpc(data)

    

    print("y before flipping\n",y)
    y=flip_bits(y,6) 
    print("y after flipping\n",y)

    output = decode_ldpc(y)
    print(output)
    print("length of output ",len(output))
    print("length of encoded data ",len(y))

    #run 100 times
    errors = 0
    for i in range(100):
        y=encode_ldpc(data)
        y=flip_bits(y,6) #6% of bits flipped results in no errors for 100 runs
        output = decode_ldpc(y)
        errors = errors + max(output !=data)

    print("errors across 100 trials: ",errors)