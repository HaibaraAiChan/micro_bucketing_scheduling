import matplotlib.pyplot as plt

def draw_lines(y_dict, string):
    print('y_dict ', y_dict)
    plt.clf()
    for method, y in y_dict.items():  # Fix: Use .items() to iterate over the dictionary
        ll = len(y)
        x = list(range(1, ll+1))  # Generates a list from 1 to ll

        # Create the line plot
        plt.plot(x, y, label=method)

    # Add labels and title
    plt.xlabel('# epoch')
    plt.ylabel(string)
    plt.title('Line Plot Example')

    # Save the plot to a PNG file
    plt.savefig(str(string) + '_line_plot.png')  # Fix: Convert string to a string using str()

# Example usage:
y_dict = {'Method1': [1, 2, 3, 4, 5], 'Method2': [2, 3, 4, 5, 6]}
draw_lines(y_dict, 'ExampleString')
