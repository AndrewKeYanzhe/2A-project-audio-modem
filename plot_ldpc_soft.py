import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-1.5, 1.5, 100)

# Calculate y values for the confidence curve
confidence = 0.5 - 0.5 * x

# Clip confidence values to range [0, 1]
confidence = np.clip(confidence, 0, 1)

# Define the second trend function
def second_trend(x):
    return np.where(x <= 0, 1, 0)

# Calculate y values for the second trend
y_second_trend = second_trend(x)

# Plot the curves
plt.figure(figsize=(3, 3))

# Plot the confidence curve
plt.plot(x, y_second_trend, label='Hard', linewidth=2.5)
plt.plot(x, confidence, label='Soft', linewidth=2.5)

# Plot the second trend


# Add labels and title
plt.xlabel('Input')
plt.ylabel('Value')
plt.title('De-mapping function')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
