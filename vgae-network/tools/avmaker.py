import sys
import os
import numpy as np

# Command-line arguments
file_prefix = sys.argv[1]  # File prefix without iteration number and extension
num_iterations = int(sys.argv[2])  # Number of iterations to average

# Initialize the list to store data arrays
data_arrays = []

# Iterate over the number of iterations
for i in range(1, num_iterations + 1):
    # Construct the file name for each iteration
    file_name = f"{file_prefix}_{i}.txt"
    
    # Read the data from the current iteration file
    current_array = np.loadtxt(file_name)
    
    # Add the current iteration array to the list
    data_arrays.append(current_array)

# Compute the average of the values across iterations
averaged_array = np.mean(data_arrays, axis=0)

# Output file name
output_file = f"{file_prefix}_averaged.txt"

# Save the averaged array to a new text file
np.savetxt(output_file, averaged_array)

print(f"Averaged array saved to {output_file}")

