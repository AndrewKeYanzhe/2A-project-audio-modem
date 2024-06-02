import numpy as np
import logging

from receiver import *



# Configure logging
logging.basicConfig(level=logging.DEBUG)

def map_bits_to_numbers(binary_data):
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

def map_numbers_to_symbols(numbers):
    constellation = {
        0: complex(1, 1),
        1: complex(-1, 1),
        2: complex(-1, -1),
        3: complex(1, -1)
    }
    symbols = []
    for number in numbers:
        symbols.append(constellation[number])
    return symbols

def qpsk_demapper(compensated_symbols, n_bins=4096, seed=1, offset=85, sequence_length=648):
    """Decode compensated QPSK symbols to original binary data."""
    # Define the QPSK Gray coding constellation mapping
    constellation = {
        complex(1, 1): 0,    # '00'
        complex(-1, 1): 1,   # '01'
        complex(-1, -1): 2,  # '11'
        complex(1, -1): 3    # '10'
    }

    # Demap QPSK symbols to numbers {0, 1, 2, 3}
    demapped_numbers = []
    for symbol in compensated_symbols:
        min_dist = float('inf')
        number = None
        for point, mapping in constellation.items():
            dist = np.abs(symbol - point)
            if dist < min_dist:
                min_dist = dist
                number = mapping
        if number is not None:
            demapped_numbers.append(number)
        else:
            logging.warning(f"No matching constellation point found for symbol {symbol}")

    logging.debug(f"Demapped numbers: {demapped_numbers}")

    # Generate the pseudo-random sequence used in the transmitter
    np.random.seed(seed)
    constellation_points = np.array([0, 1, 2, 3])
    number_extended = np.random.choice(constellation_points, n_bins)[offset:offset + sequence_length]

    logging.debug(f"Pseudo-random sequence: {number_extended}")

    original_numbers = []

    for i in range(len(demapped_numbers)):
        corresponding_index = (i)%684
        for x in range(4):
            if (x + number_extended[corresponding_index]) % 4 == demapped_numbers[i]:
                original_numbers.append(x)
                break

    logging.debug(f"Original numbers: {original_numbers}")

    # Map numbers back to binary data using QPSK with Gray coding
    reverse_constellation = {
        0: '00',
        1: '01',
        2: '11',
        3: '10'
    }

    binary_data = ''
    for number in original_numbers:
        binary_data += reverse_constellation[number]

    return binary_data

# Test the function
original_binary_data = '1101010111001010110101110001001100101010111001011010101110010010'  # Example binary data

# 1. Map original binary data to numbers
numbers = map_bits_to_numbers(original_binary_data)
logging.debug(f"Original numbers: {numbers}")

# 2. Generate the pseudo-random sequence used in the transmitter
np.random.seed(1)
constellation_points = np.array([0, 1, 2, 3])
number_extended = np.random.choice(constellation_points, 4096)[85:85+648]
logging.debug(f"Pseudo-random sequence: {number_extended}")

# 3. Apply modulus multiplication
modulus_multiplication_result = []
for i in range(len(numbers)):
    corresponding_index = i % 648
    result = (numbers[i] + number_extended[corresponding_index]) % 4
    modulus_multiplication_result.append(result)
logging.debug(f"Modulus multiplication result: {modulus_multiplication_result}")

# 4. Map the result to QPSK symbols
qpsk_symbols = map_numbers_to_symbols(modulus_multiplication_result)
logging.debug(f"QPSK symbols: {qpsk_symbols}")

# 5. Demodulate to recover original binary data
# recovered_binary_data = qpsk_demapper(qpsk_symbols)
print(len(qpsk_symbols))
# recovered_binary_data = qpsk_demapper(qpsk_symbols)
print("Original binary data:  ", original_binary_data)
recovered_binary_data = qpsk_demap_probabilities(qpsk_symbols, 1.41, bins_used=32, start_bin=0)
# recovered_binary_data = int(recovered_binary_data)
# recovered_binary_data = [int(x) for x in recovered_binary_data]

logging.debug(f"Recovered binary data: {recovered_binary_data}")

# 6. Compare the recovered binary data with the original binary data

print("Recovered binary data: ", recovered_binary_data)

# Check if the recovered binary data matches the original binary data
assert original_binary_data == recovered_binary_data, "The recovered binary data does not match the original!"
print("Test passed! The recovered binary data matches the original binary data.")
