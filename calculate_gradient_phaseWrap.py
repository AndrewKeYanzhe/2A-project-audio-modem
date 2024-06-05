import numpy as np
import matplotlib.pyplot as plt
import statistics



gradient = 3

# Generate example data
n = np.arange(0, 700)
phase = np.mod(n * gradient, 360)  # Simulating phase wraparound

# # Scatter plot
# plt.scatter(n, phase)
# plt.title('Phase of g[n] at bin 85')
# plt.xlabel('n')
# plt.ylabel('Phase (degrees)')
# plt.show()

def calculate_gradients(phase, plot=False):
    # Generate example data
    n = np.arange(0, len(phase))
    # phase = np.mod(n * gradient, 360)  # Simulating phase wraparound

    # Scatter plot
    plt.scatter(n, phase)
    plt.title('Phases')
    plt.xlabel('n')
    plt.ylabel('Phase (degrees)')
    plt.show()


    



    gradients = []
    window_size = 10
    half_window = window_size // 2

    for i in range(len(phase) - window_size + 1):
        window = phase[i:i + window_size]
        avg_first_half = sum(window[:half_window]) / half_window
        avg_second_half = sum(window[half_window:]) / half_window
        gradient = (avg_second_half - avg_first_half)/5
        gradients.append(gradient)
        
    # Plotting the gradients
    plt.figure(figsize=(10, 5))
    plt.plot(gradients, marker='o', linestyle='-', color='b')
    plt.title('Gradients Plot')
    plt.xlabel('Index')
    plt.ylabel('Gradient Value')
    plt.grid(True)
    plt.show()

    gradient_mode = statistics.mode(gradients)


    
    return gradient_mode




if __name__ == "__main__":

    print(" ")
    print("mode of calculated gradients         ",calculate_gradients(phase, plot=True))
    print("gradient setting used for testing    ", gradient)




