import matplotlib.pyplot as plt
import numpy as np

# List of file paths to read
file_paths = ["nb_1_loss.txt", "nb_2_loss.txt", "nb_4_loss.txt", "nb_8_loss.txt"]

# Initialize empty lists to store data
data = []

# Read and parse data from each file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming each line contains a single value (e.g., a number)
        data.append([float(line.strip()) for line in lines])

# Convert data to a NumPy array for plotting
data = np.array(data)

# Create x values (assuming x has a length of 400)
x = np.arange(1, 401)

# Create a single figure and plot data from each file
plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
line_styles = ['-', '--',  '-.', ':',]  # Specify line styles for each file
j_list = [1,2,4,8]
for i, file_path in enumerate(file_paths):
    plt.plot(x, data[i], label=f'number of batch {j_list[i]}', linestyle=line_styles[i % len(line_styles)])

# for i, file_path in enumerate(file_paths):
#     plt.plot(x, data[i], label=f'File {i+1}')

# Add labels, legend, and title
plt.xlabel('# epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.title('Multiple Files Data')
plt.savefig('output_figure.png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

