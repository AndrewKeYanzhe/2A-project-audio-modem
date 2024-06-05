import numpy as np
import matplotlib.pyplot as plt

def calculate_gradients(phase, plot_phases=False, plot_gradients=False, plot_std=False):
    # Generate example data
    n = np.arange(0, len(phase))
    # phase = np.mod(n * gradient, 360)  # Simulating phase wraparound
    
    if plot_phases:
        # Scatter plot
        plt.scatter(n, phase)
        plt.title('Phases')
        plt.xlabel('n')
        plt.ylabel('Phase (degrees)')
        plt.title('Phase v.s. OFDM Symbol Index')
        plt.show()

    gradients = []
    std_devs = []
    if len(phase)<100:
        window_size = 10
    else:
        window_size=20
    half_window = window_size // 2

    for i in range(len(phase) - window_size + 1):
        window = phase[i:i + window_size]
        # Calculate gradient and std using OLS
        gradient = np.polyfit(np.arange(i, i + window_size), window, 1)[0]
        std_dev = np.std(window)
        gradients.append(gradient)
        std_devs.append(std_dev)        
        
    if plot_gradients:
        # Plotting the gradients
        plt.figure(figsize=(10, 5))
        plt.plot(gradients, marker='o', linestyle='-', color='b')
        plt.title('Gradients Plot')
        plt.xlabel('Index')
        plt.ylabel('Gradient Value')
        plt.grid(True)
        plt.show()
        
    if plot_std:
        # Plotting the gradients
        plt.figure(figsize=(10, 5))
        plt.plot(std_devs, marker='o', linestyle='-', color='b')
        plt.title('Standard Deviation Plot')
        plt.xlabel('Index')
        plt.ylabel('Standard Deviation')
        plt.grid(True)
        plt.show()

    def find_outliers(data_np, threshold=2):
        data_np = np.array(data_np)

        # Calculate quartiles
        Q1 = np.percentile(data_np, 25)
        Q3 = np.percentile(data_np, 75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = (data_np < lower_bound) | (data_np > upper_bound)

        return outliers
    
    gradients = np.array(gradients)
    outliers = find_outliers(std_devs)
    # Reject outliers
    filtered_gradients = gradients[~outliers]
    # Calculate mean of remaining values
    mean_gradient = np.mean(filtered_gradients)
    
    return mean_gradient



if __name__ == "__main__":
    
    gradient = 3

    # Generate example data
    n = np.arange(0, 500)
    phase = np.mod(n * gradient, 360)  # Simulating phase wraparound
    phase = [x + np.random.normal(0, 50) for x in phase]

    print("mode of calculated gradients         ",calculate_gradients(phase, plot_phases=True, plot_gradients=True, plot_std=True))
    print("gradient setting used for testing    ", gradient)




