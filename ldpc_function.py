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

1. set the encoded_block_length (smaller than OFDM block size.)


    the transmitted block has to be a multiple of 24 (see /py/ldpc.py)
    e.g 4032 fits within the 4088 bits ofdm data block

2. encode_ldpc(data) 

    data is half of encoded_block_length
    e.g. 2016

3. decode_ldpc(signal)

    signal is encoded_block_length
"""



"""
TODO need to calculate signal to noise ratio at different frequencies and provide snr to ldpc for better performance

higher performance with longer blocks according to jossy    

"""


import random
import ldpc.py.ldpc as ldpc
import numpy as np

np.set_printoptions(threshold=np.inf)


# Create an LDPC code object

encoded_block_length = 1176
data_block_length = int(encoded_block_length/2)
z=int(encoded_block_length/24)

c = ldpc.code(standard = '802.16', rate = '1/2', z=z, ptype='A')



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


def encode_ldpc(data): 
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
    output=output[0:data_block_length]

    return output



if __name__ == "__main__":
    # Check the standard used
    print(c.standard)  # Output: '802.11n'

    # Generate a random message
    # data = np.random.randint(0, 2, c.K)
    data = [0,0,1,1]

    data = np.pad(data, (0, data_block_length - len(data)), 'constant', constant_values=(0, 0)) 

    print("padded data\n",data)

    y=encode_ldpc(data)

    

    print("y before flipping\n",y)

    
    y=flip_bits(y,6) 


    print("y after flipping\n",y)

    output = decode_ldpc(y)
    print("decoded data\n",output)
    print("length of output ",len(output))
    print("length of encoded data ",len(y))

    #run 100 times
    errors = 0
    for i in range(100):
        y=encode_ldpc(data)
        y=flip_bits(y,7) 
        #6% of bits flipped results in no errors for 100 runs. with 1008 encoded length
        #7% of bits flipped results in no errors for 100 runs. with 4032 encoded length

        output = decode_ldpc(y)
        errors = errors + max(output !=data)

    print("errors across 100 trials: ",errors)