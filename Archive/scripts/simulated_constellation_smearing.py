import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, std, shift):
    # Define the constellations
    constellations = np.array([1+1j, -1+1j, -1-1j, 1-1j])
    delta_t = shift / 48000  # Time step

    # Generate n random indices to select constellations
    indices = np.random.randint(0, len(constellations), size=n)

    # Select constellations and add Gaussian noise
    noise_real = np.random.normal(0, std, n)
    noise_imag = np.random.normal(0, std, n)
    complex_list = constellations[indices] + noise_real + 1j*noise_imag

    # Generate n real frequencies, uniformly distributed from 1000 Hz to 8000 Hz
    frequency_list = np.random.uniform(1000, 8000, n)

    # Calculate rotation for each complex number based on its frequency
    rotation_angles = 2 * np.pi * frequency_list * delta_t  # Convert frequency to radians
    rotations = np.exp(1j * rotation_angles)  # Euler's formula for complex rotation

    # Rotate each complex number
    rotated_complex_list = complex_list * rotations

    return constellations[indices], rotated_complex_list, frequency_list

def plot_complex_numbers(true_constellations, complex_numbers, shift):
    # Map the constellations to colors and labels
    constellation_map = {
        1+1j: ('red', '00'),
        -1+1j: ('blue', '01'),
        -1-1j: ('green', '11'),
        1-1j: ('yellow', '10')
    }

    plt.figure(figsize=(7, 5))  # Adjust the figure size as needed

    # Collecting labels to avoid duplicates in legend
    labels_collected = {}
    for const, color_label in zip(true_constellations, [constellation_map[c] for c in true_constellations]):
        color, label = color_label
        if label not in labels_collected:
            plt.scatter(complex_numbers.real[true_constellations == const], complex_numbers.imag[true_constellations == const],
                        c=color, s=20, label=label)  # Reduced size with 's=20'
            labels_collected[label] = color
        else:
            plt.scatter(complex_numbers.real[true_constellations == const], complex_numbers.imag[true_constellations == const],
                        c=color, s=20)

    font_size = 16
    plt.xlabel('Real Part', fontsize=font_size)
    plt.ylabel('Imaginary Part', fontsize=font_size)
    plt.title(f'Simulated Received Constellations\nwith {shift} Timestamps Delay', fontsize=font_size)
    plt.grid(True)
    plt.axis('equal')
    legend = plt.legend(title="Constellation", fontsize=font_size)  # Add a title to the legend
    plt.setp(legend.get_title(), fontsize=font_size)  # or a specific size like 15
    plt.tick_params(axis='both', which='major', labelsize=font_size)


    plt.show()



# Example usage
n = 1000  # number of values
shift = 2
true_constellations, complex_numbers, frequencies = generate_data(n,0.3,shift)

# Plotting
plot_complex_numbers(true_constellations, complex_numbers,shift)
