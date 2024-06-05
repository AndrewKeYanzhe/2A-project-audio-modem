import numpy as np
import matplotlib.pyplot as plt

def calculate_gradients(phase, plot=False):
    # Generate example data
    n = np.arange(0, len(phase))
    # phase = np.mod(n * gradient, 360)  # Simulating phase wraparound
    


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

    def find_outliers(data_np, threshold=0.5):
        data_np = np.array(data_np)
        mean = np.mean(data_np)
        std_dev = np.std(data_np)
        outliers = data_np - mean > threshold * std_dev
        return outliers
    
    gradients = np.array(gradients)
    std_devs = np.array(std_devs)
    outliers = find_outliers(std_devs)
    # Reject outliers
    filtered_gradients = gradients[~outliers]
    # Calculate mean of remaining values
    mean_gradient = np.mean(filtered_gradients)
    
    if plot:
        # Create a figure with three subplots arranged vertically
        plt.figure(figsize=(6, 8))
        # Scatter plot of phases
        plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
        plt.scatter(n, phase)
        plt.title('Phase v.s. OFDM Symbol Index')
        plt.xlabel('n')
        plt.ylabel('Phase (degrees)')
        # Line plot of gradients
        plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
        plt.plot(gradients)
        plt.scatter(np.where(outliers)[0], gradients[outliers], color='r', label='Outliers')
        plt.title('Gradients Plot')
        plt.xlabel('Index')
        plt.ylabel('Gradient Value')
        plt.grid(True)
        # Line plot of standard deviations
        plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
        plt.plot(std_devs)
        plt.scatter(np.where(outliers)[0], std_devs[outliers], color='r', label='Outliers')
        plt.title('Standard Deviation Plot')
        plt.xlabel('Index')
        plt.ylabel('Standard Deviation')
        plt.grid(True)
        # Display the combined figure
        plt.tight_layout()
        plt.show()
    
    
    return mean_gradient



if __name__ == "__main__":
    
    gradient = 3

    # Generate example data
    n = np.arange(0, 500)
    phase = np.mod(n * gradient, 360)  # Simulating phase wraparound
    phase = [x + np.random.normal(0, 50) for x in phase]

    print("mode of calculated gradients         ",calculate_gradients(phase, plot=True))
    print("gradient setting used for testing    ", gradient)




