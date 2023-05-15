import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# period of 10 ms
sampling_rate = 100

# Load CSV file into a pandas DataFrame
df = pd.read_csv('data/3300-4200.csv')
print(df)
# Calculate magnitude of x, y, and z signals
magnitude = (df['x']**2 + df['y']**2 + df['z']**2)**0.5

# Add magnitude column to DataFrame
df['magnitude'] = magnitude

# Low pass filter
order = 4
fs = sampling_rate  # sample rate, Hz
cutoff = 4  # desired cutoff frequency of the filter, Hz. 
nyq = 0.5 * fs
normal_cutoff = cutoff / nyq
b, a = butter(order, normal_cutoff, btype='low', analog=False)

lowpass = filtfilt(b, a, magnitude)

# Plot the magnitude data
print(len(df))
fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(df['time'], magnitude)
# ax.plot(df['time'], lowpass)
ax.plot(range(len(magnitude)), magnitude)
ax.set_title('Magnitude of Accelerometer Data')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Magnitude (m/s^2)')

# Show the plot
plt.show()
