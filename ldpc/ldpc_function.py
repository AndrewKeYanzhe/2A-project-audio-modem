"""
on mac:
1. install command line tools
2. create 'bin' folder
3. run the following as stated in readme
gcc -lm -shared -fPIC -o bin/c_ldpc.so src/c_ldpc.c
gcc -o bin/results2csv src/results2csv.c

"""


import random
import py.ldpc as ldpc
import numpy as np

# Create an LDPC code object
c = ldpc.code()

# Check the standard used
print(c.standard)  # Output: '802.11n'

# Generate a random message
data = np.random.randint(0, 2, c.K)
data = [0,0,1,1]

data = np.pad(data, (0, 324 - len(data)), 'constant', constant_values=(0, 0)) #length must be 324

print("data\n",data)
# print(c.K)
# print(u)

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

    


    # Encode the message
    x = c.encode(data)

    # print(len(x), "seems to be twice as long")

    # print("x\n",x)

    return x

# # Verify the encoding by checking the parity-check matrix
# verification = np.mod(np.matmul(x, np.transpose(c.pcmat())), 2)
# # print(verification)  # Output: array([0, 0, ..., 0])

# # Simulate the received signal


y=encode_ldpc(data)


# y=encode_ldpc(data)
print("y before flipping\n",y)


y=flip_bits(y,8) #8% of bits flipped results in no errors most of the time


print("y after flipping\n",y)


def decode_ldpc(signal):
    y = 10 * (.5 - signal)

    # Decode the received signal
    app, it = c.decode(y)


    # Check the number of iterations taken by the decoder
    print("iterations",it)  # Output: 0

    # output = app
    # print(c.decode(output))

    output = np.where(app < 0, 1, 0)

    output=output[0:324]


    return output

output = decode_ldpc(y)

# # Verify if the decoded message matches the original codeword
# errors = np.nonzero((app < 0) != x)
# print(errors)  # Output: (array([], dtype=int64),)

# output = c.decode(app < 0)
# print(app)
# output = np.where(app >0.5 , 1, 0)



print(output)
print(len(output))
# print(c.decode(output))
print(max(output !=data))



errors = 0
for i in range(100):
    y=encode_ldpc(data)
    y=flip_bits(y,6) #6% of bits flipped results in no errors for 100 runs
    output = decode_ldpc(y)
    errors = errors + max(output !=data)

print("errors across 100 trials: ",errors)
