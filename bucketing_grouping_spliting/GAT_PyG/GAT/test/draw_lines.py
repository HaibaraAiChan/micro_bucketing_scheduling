import matplotlib.pyplot as plt

# Step 1: Read the contents of the two files and construct lists.
# Replace 'file1.txt' and 'file2.txt' with the actual file names.

file1_contents = []
file2_contents = []



with open('loss.txt', 'r') as file1:
    for line in file1:
        try:
            value = float(line.strip())
            file1_contents.append(value)
        except ValueError:
            # Handle non-numeric lines here, e.g., by skipping or providing a default value.
            pass

with open('loss_012.txt', 'r') as file2:
    for line in file2:
        try:
            value = float(line.strip())
            file2_contents.append(value)
        except ValueError:
            # Handle non-numeric lines here, e.g., by skipping or providing a default value.
            pass

x_values = list(range(1, len(file1_contents) + 1))

plt.plot(x_values, file1_contents, label='sum lr=0.005')
plt.plot(x_values, file2_contents, label='lstm lr=0.012')

# Add labels, title, legend, etc.
plt.xlabel('# epoch')
plt.ylabel('training loss')
plt.title('Line Plot of sum lr=5e-3 and lstmlr=12e-3')
plt.legend()

# Save the plot to a file (e.g., as a PNG image).
plt.savefig('line_plot.png')

# Optionally, you can also specify the file format and adjust other parameters:
# plt.savefig('line_plot.png', format='png', dpi=300)

# Close the plot to release resources (optional).
plt.close()
