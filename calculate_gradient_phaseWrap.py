import numpy as np
import matplotlib.pyplot as plt
import statistics



gradient = 3

# Generate example data
n = np.arange(0, 700)
phase = np.mod(n * gradient, 360)  # Simulating phase wraparound

phase = [x + np.random.normal(0, 10) for x in phase]


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
    if len(phase)<100:
        window_size = 10
    else:
        window_size=30
    half_window = window_size // 2

    for i in range(len(phase) - window_size + 1):
        window = phase[i:i + window_size]
        avg_first_half = sum(window[:half_window]) / half_window
        avg_second_half = sum(window[half_window:]) / half_window
        gradient = (avg_second_half - avg_first_half)/(window_size/2)
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

    # import numpy as np

    print(gradients)

    def reject_outliers(data_np, threshold=2):
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

        # Filter out outliers
        # filtered_data = [item for item in data_np if item not in outliers]

        filtered_data = data_np[~outliers]


        # Calculate mean of filtered data
        mean_data = np.average(filtered_data)
        return mean_data




    # Example list of gradients
    # gradients = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]

    # Reject outliers
    filtered_gradients = reject_outliers(gradients)

    # Calculate mean of remaining values
    mean_gradient = np.mean(filtered_gradients)

    # print("Original Gradients:", gradients)
    # print("Filtered Gradients:", filtered_gradients)
    # print("Mean of Remaining Gradients:", mean_gradient)



    
    return mean_gradient




if __name__ == "__main__":

    print(" ")
    print("mode of calculated gradients         ",calculate_gradients(phase, plot=True))
    print("gradient setting used for testing    ", gradient)




