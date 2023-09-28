import os

def read_files_in_folder(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        print('file------ ', file_name)
        end_time=[]
        if os.path.isfile(file_path):  # Ensure it is a file, not a subfolder
            with open(file_path, 'r') as file:
                
                lines = file.readlines()
                for line in lines:
                    if line.startswith("----------------------------------------------------------pseudo_mini_loss sum"):
                        number = float(line.split(" ")[2].strip())
                        print(number)
                        end_time.append(number)
                # if len(end_time)>3:
                #     print('avg end to end time ')
                #     print( ' ---------------- '+file_name + '  '+str(sum(end_time[3:])/len(end_time[3:])))
        print()
read_files_in_folder('./')