import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-1.5, 1.5, 100)

# Calculate y values based on the equation
confidence = 0.5 - 0.5 * x

# Clip confidence values to range [0, 1]
confidence = np.clip(confidence, 0, 1)

# Plot the curve

plt.figure(figsize=(3, 3))

# plt.plot(x, confidence, label='Confidence')
plt.plot(x, confidence)

# Add labels and title
plt.xlabel('input')
plt.ylabel('Confidence')
plt.title('Confidence')
plt.grid(True)
# plt.legend()
plt.tight_layout()

# Show the plot
plt.show()
