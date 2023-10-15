import os
import matplotlib.pyplot as plt
import statistics

def draw(y, string):
    plt.clf()
    ll = len(y)
    x = list(range(1, ll+1))  # Generates a list from 1 to 200

    # Create the line plot
    plt.plot(x, y)

    # Add labels and title
    plt.xlabel('# epoch')
    plt.ylabel(string)
    plt.title('Line Plot Example')

    # Save the plot to a PNG file
    plt.savefig(str(string) +'_line_plot.png')
    
    # Show the plot
    plt.show()

    
def read_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    f1_list=[]
    for file_name in file_list:
        if not 'log' in file_name:
            continue
        if not 'arxiv' in file_name:
            continue
        file_path = os.path.join(folder_path, file_name)
        print('file------ ', file_name)
        
        
        if os.path.isfile(file_path) :  # Ensure it is a file, not a subfolder
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("Micro"):
                        # print(line.split(" "))
                        f1 = float(line.split(" ")[3].strip())
                        
                        f1_list.append(f1)
                        
        
        # print('f1_list ', f1_list)
        # draw(f1_list, 'f1_list')
        # print()
        # print('max f1_list ',max(f1_list))
        print('max f1_list ',max(f1_list))
        return f1_list
        



read_files_in_folder('./')
numbers = read_files_in_folder('./')
mean = statistics.mean(numbers)
std_dev = statistics.stdev(numbers)
formatted_result = f"{mean:.2f} ± {std_dev:.2f}"
print(formatted_result)