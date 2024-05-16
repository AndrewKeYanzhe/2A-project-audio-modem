import pandas as pd
import numpy as np
#load the fir filter
fir_filter = pd.read_csv('fir_filter.csv').values.flatten()
# calculate the sum of the squared values of the FIR filter
sum_squared_values = np.sum(fir_filter**2)
print(sum_squared_values)

# plot the FIR filter
import matplotlib.pyplot as plt
plt.plot(fir_filter)
plt.show()