# import matplotlib.pyplot as plt

# # Data for the first line
# x1 = [17.8, 15.01, 13.77]
# y1 = [3.77, 3.88, 4,]

# # Data for the second line
# x2 = [21.56,19.66,14.8]
# y2 = [2.56,2.48,2.44]

# # Plotting the lines with dots
# plt.plot(x1, y1, 'o-', label='Betty')
# plt.plot(x2, y2, 'o-', label='Micro-Bucketing')

# # Adding labels and title
# plt.xlabel('CUDA memory (GB)')
# plt.ylabel('Pure training time (sec)')
# plt.title('Pure train time v.s. CUDA mem')

# plt.legend()

# # Saving the plot as a PNG file
# plt.savefig('plot.png', dpi=300)  # Specify the file name and dpi (dots per inch)

# # Displaying the plot
# plt.show()

# import matplotlib.pyplot as plt

# # Data for the first line
# x1 = [17.8, 15.01, 13.77]
# y1 = [23.27, 23.68, 24.01]

# # Data for the second line
# x2 = [21.56, 19.66, 14.8]
# y2 = [9.74, 10.25, 10.81]

# # Plotting the lines with dots
# plt.plot(x1, y1, 'o-', label='Betty')
# plt.plot(x2, y2, 'o-', label='Micro-Bucketing')

# # Adding labels and title
# plt.xlabel('CUDA memory (GB)')
# plt.ylabel('End to end  time (sec)')
# plt.title('End to end  time v.s. CUDA mem')

# plt.legend()

# # Saving the plot as a PNG file
# plt.savefig('plot1.png', dpi=300)  # Specify the file name and dpi (dots per inch)

# # Displaying the plot
# plt.show()

import matplotlib.pyplot as plt

# # Data for the first line
x1 = [17.8, 15.01, 13.77]
y1 = [3.77, 3.88, 4,]

# Data for the second line
x2 = [21.56,19.66,14.8]
y2 = [2.56,2.48,2.44]

# create some data
x3 = [17.8, 15.01, 13.77]
y3 = [23.27, 23.68, 24.01]

# Data for the second line
x4 = [21.56, 19.66, 14.8]
y4 = [9.74, 10.25, 10.81]

fig = plt.figure()

# subplot 1
plt.subplot(2, 1, 1)  # 2 rows, 1 column, first plot
plt.plot(x1, y1,'o-', label='Betty')
plt.plot(x2, y2, 'o-', label='Micro-Bucketing')
plt.title('Pure train time v.s. CUDA mem')
plt.xlabel('CUDA memory (GB)')
plt.ylabel('Pure training time (sec)')
plt.legend() # added legend

# subplot 2
plt.subplot(2, 1, 2)  # 2 rows, 1 column, second plot
plt.plot(x3, y3,'o-', label='Betty')
plt.plot(x4, y4, 'o-', label='Micro-Bucketing')
plt.title('End to end  time v.s. CUDA mem')
plt.xlabel('CUDA memory (GB)')
plt.ylabel('End to end  time (sec)')
plt.legend() # added legend
# adjust spacing
plt.tight_layout()
plt.savefig('plot2.png', dpi=300)
# display the plot
plt.show()
