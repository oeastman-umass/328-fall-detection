import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
df = pd.read_csv('accelerometer.csv')

# Create a figure with subplots for each axis
fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 6))
fig.suptitle('Accelerometer Data')

# Plot the z-axis data
axs[0].plot(df['time'], df['z'])
axs[0].set_ylabel('Z-axis (m/s^2)')

# Plot the y-axis data
axs[1].plot(df['time'], df['y'])
axs[1].set_ylabel('Y-axis (m/s^2)')

# Plot the x-axis data
axs[2].plot(df['time'], df['x'])
axs[2].set_ylabel('X-axis (m/s^2)')
axs[2].set_xlabel('Time (s)')

# Show the plot
plt.show()