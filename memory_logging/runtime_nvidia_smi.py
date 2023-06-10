import time
import subprocess

def start_memory_logging(log_file, update_interval=1):
    """
    Start logging GPU memory consumption using nvidia-smi.
    
    Parameters:
        log_file (str): Name of the log file to save memory consumption data.
        update_interval (int, optional): Interval (in seconds) at which memory consumption is logged. Defaults to 1.
    """
    # Start the nvidia-smi command with appropriate arguments
    command = ['nvidia-smi', 'dmon', '-s', 'mu', '-d', str(update_interval), '-o', 'DT']

    # Open the log file for writing
    with open(log_file, 'w') as f:
        # Run the nvidia-smi command and redirect the output to the log file
        process = subprocess.Popen(command, stdout=f, universal_newlines=True)

    return process

def stop_memory_logging(process):
    """
    Stop logging GPU memory consumption by terminating the nvidia-smi process.
    
    Parameters:
        process (subprocess.Popen): The process object representing the nvidia-smi command.
    """
    # Terminate the nvidia-smi process
    process.terminate()



if __name__=='__main__':
	
    # Usage example
    log_file = 'memory.log'
    update_interval = 1  # in seconds

    # Start memory logging
    logging_process = start_memory_logging(log_file, update_interval)

    # Your main code here
    print('hello'*100)
    print("Running a long process...")
    time.sleep(5)  # Sleep for 5 seconds to simulate a long process

    # Stop memory logging when desired
    stop_memory_logging(logging_process)
    print("Memory logging stopped.")