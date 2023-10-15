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

    
def read_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        if not 'log' in file_name:
            continue
        file_path = os.path.join(folder_path, file_name)
        print('file------ ', file_name)
        loss=[]
        train_acc = []
        test_acc = []
        if os.path.isfile(file_path) :  # Ensure it is a file, not a subfolder
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("Run"):
                        t_loss = float(line.split(" ")[7].strip())
                        t_acc = float(line.split(" ")[10].strip())
                        acc = float(line.split(" ")[16].strip())
                        train_acc.append(t_acc)
                        test_acc.append(acc)
                        loss.append(t_loss)
        print('loss ' , loss)
        draw(loss, 'loss')
        print()
        print('train_acc ', train_acc)
        draw(train_acc, 'train_acc')
        print('test_acc ', test_acc)
        draw(test_acc, 'test_acc')
        print()
        print('max train acc ',max(train_acc))
        



read_files_in_folder('./')