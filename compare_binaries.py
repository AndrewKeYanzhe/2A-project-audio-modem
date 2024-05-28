from received_bin2txt import *


def read_binary_file(file_path):
    with open(file_path, 'rb') as file:
        return file.read()

def compare_files(file1, file2, max_shift=30):
    data1 = read_binary_file(file1)
    data2 = read_binary_file(file2)

    len1, len2 = len(data1), len(data2)
    max_len = max(len1, len2)
    best_shift = 0
    max_identical_bytes = 0

    for shift in range(-max_shift, max_shift + 1):
        identical_bytes = 0
        for i in range(max_len):
            if 0 <= i < len1 and 0 <= i - shift < len2:
                if data1[i] == data2[i - shift]:
                    identical_bytes += 1

        if identical_bytes > max_identical_bytes:
            max_identical_bytes = identical_bytes
            best_shift = shift

    percentage_identical = (max_identical_bytes / max_len) * 100

    return best_shift, percentage_identical


def compare_files_bitwise(file1, file2, max_shift=100):
    data1 = read_binary_file(file1) # this is reference
    data2 = read_binary_file(file2)

    len1, len2 = len(data1) * 8, len(data2) * 8
    max_len = max(len1, len2)
    best_shift = 0
    max_identical_bits = 0

    for shift in range(-max_shift, max_shift + 1):
        identical_bits = 0
        for i in range(max_len):
            idx1 = i // 8
            bit1 = (data1[idx1] >> (i % 8)) & 1 if idx1 < len(data1) else 0

            idx2 = (i - shift) // 8
            bit2 = (data2[idx2] >> ((i - shift) % 8)) & 1 if idx2 < len(data2) else 0

            if bit1 == bit2:
                identical_bits += 1

        if identical_bits > max_identical_bits:
            max_identical_bits = identical_bits
            best_shift = shift

    percentage_identical = (max_identical_bits / max_len) * 100

    return best_shift, percentage_identical

#compare 2 will shift the received file +- max shift and compare with reference file. extra corrupted bits in front of and behind main text in received file is ignored
def compare_2(file1, file2, max_shift=100):
    byte_data1 = load_bin_file(file1)
    byte_data2 = load_bin_file(file2)
    reference_binary_string = bytes_to_binary_string(byte_data1)
    received_binary_string = bytes_to_binary_string(byte_data2)



    # reference_binary_string = bytes_to_binary_string(file1)
    # received_binary_string = bytes_to_binary_string(file2)

    # print("length of reference", len(reference_binary_string))

    # print(reference_binary_string[:100])
    # print(received_binary_string[0:100])

    best_match = 0
    shift_at_best_match = 0

    for shift in range(-max_shift, max_shift + 1):
        compared_length = 0
        matched_bits = 0
        for i in range(len(reference_binary_string)):

            compared_length = compared_length + 1
            if i+shift >=0 and i+shift <= len(received_binary_string)-1:
                if reference_binary_string[i] == received_binary_string[i+shift]:
                    matched_bits = matched_bits+1
        # print(compared_length)

        if matched_bits>best_match:
            best_match = matched_bits
            shift_at_best_match = shift
    
    print("bit shift at best match", shift_at_best_match)
    print("percentage of bits that are the same",best_match/len(reference_binary_string))
# Example usage:
# file1_path = 'binaries/received_binary_0525_1749 constellation shifted.bin'
ref_path = 'text/article_2.txt'
# ref_path = 'text/article.txt'
# unshifted_path = 'binaries/received_binary_0525_1749 constellation non shifted.bin'
# unshifted_path = 'binaries/received_binary_0525_1749 constellation non shifted.bin'
unshifted_path ='binaries/received_0527_2103_pilot_iceland_decodeUsingPilot.bin'



# shift, percentage = compare_files_bitwise(ref_path, unshifted_path)

# print(f'Percentage of identical bits: {percentage:.2f}%')
# print(f'Best shift: {shift}')
print('\nchannel from chirp')
unshifted_path ='binaries/received_0527_2103_pilot_iceland_decodeUsingChirp.bin'
compare_2(ref_path, unshifted_path)

print('\nchannel from pilot')
unshifted_path ='binaries/received_0527_2103_pilot_iceland_decodeUsingPilot.bin'
compare_2(ref_path, unshifted_path)


# shift2, percentage2 = compare_files_bitwise(ref_path, shifted_path)

# print(f'Percentage of identical bits with constellation phase rotation: {percentage2:.2f}%')
# print(f'Best shift2: {shift}')





