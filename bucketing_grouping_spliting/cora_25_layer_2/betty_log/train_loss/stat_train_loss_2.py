
import statistics
import os
import matplotlib.pyplot as plt

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

def draw_lines(y_dict, string):
    print('y_dict ', y_dict)
    plt.clf()
    for method,y in y_dict.items():
        ll = len(y)
        x = list(range(1, ll+1))  # Generates a list from 1 to 200

        # Create the line plot
        plt.plot(x, y, label=method)

    # Add labels and title
    plt.xlabel('# epoch')
    plt.ylabel(string)
    plt.title('Line Plot Example')
    plt.legend()
    # Save the plot to a PNG file
    plt.savefig(str(string) +'_line_plot.png')
    
    # Show the plot
    plt.show()

def read_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)
    dataset = ''
    loss_dict={}
    for file_name in file_list:
        if not 'log' in file_name:
            continue
        if not 'train_loss' in file_name:
            continue
        # if not 'lr_0.01' in file_name:
        #     continue
        # if not 'cora' in file_name:
        #     continue
        dataset = 'cora'
        # if not 'arxiv' in file_name:
        #     continue
        
        # dataset = 'arxiv'
        # if not 'pubmed' in file_name:
        #     continue
        # dataset = 'pubmed'
        file_path = os.path.join(folder_path, file_name)
        print('file------ ', file_name)
        method =''
        if 'range' in file_name:
            method = 'range'
        elif 'random' in file_name:
            method = 'random'
        elif 'REG' in file_name:
            method = 'REG'
        else:
            method = 'full batch'
        if 'nb_2' in file_name:
            method = method + ' nb_2'
        if 'nb_4' in file_name:
            method = method + ' nb_4'
        loss_list_cur=[]
        if os.path.isfile(file_path) :  # Ensure it is a file, not a subfolder
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("----------------------------------------------------------pseudo_mini_loss sum"):
                        # print(line.split(" "))
                        print(line.split(" ")[2].strip())
                        loss = float(line.split(" ")[2].strip())
                        loss_list_cur.append(loss)
                        # print(loss_list_cur)
        # draw(loss_list_cur, str(dataset) + ' training loss ')
        loss_dict[method]=loss_list_cur  
    print(loss_dict)
    draw_lines(loss_dict, str(dataset) + ' training loss ')                 
    return loss_dict



loss_dict= read_files_in_folder('./')
numbers_list = loss_dict.values()
for numbers in numbers_list:
    print('max loss_list ',max(numbers))
    print('min loss_list ',min(numbers))
    length = len(numbers)
    start = 1
    end = 10
    mean = statistics.mean(numbers[start: end])
    std_dev = statistics.stdev(numbers[start: end])
    formatted_result = f"{mean:.8f} ± {std_dev:.8f}"
    print(formatted_result)