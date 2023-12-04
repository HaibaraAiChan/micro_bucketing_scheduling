import os

# Specify the folder path
folder_path = './'

# List all files in the folder
file_list = os.listdir(folder_path)

# Iterate through each file in the folder
for filename in file_list:
    # Check if the item is a file (not a subdirectory)
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Open and read the file
        with open(os.path.join(folder_path, filename), "r") as file:
            # Read the numbers from the file into a list
            print(filename)
            lines = file.readlines()
            for line in lines:
                if line.startswith("num_input_list"):
                    # print(line.split(" "))
                    # print(line.split(" ")[1].strip())
                    list_ = line.split("[")[1].strip()
                    list__ = list_.split("]")[0].strip()
                    numbers = list__.split(",")
                    numbers = [float(n) for n in numbers]
                    # print(numbers)
        


            # Calculate the mean
            mean = sum(numbers) / len(numbers)

            # Print the mean
            print("Mean:", mean)
            print()