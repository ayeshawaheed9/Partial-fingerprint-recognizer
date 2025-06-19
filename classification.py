import os
import shutil
import time
import psutil

# Define the main directory containing the subfolders with fingerprint files
# Adjust this path to where your fingerprint data is stored
main_directory = r'NISTSpecialDatabase4GrayScaleImagesofFIGS\sd04\png_txt'

# Define the directory where the classified folders will be created
# Adjust this path to where you want the classified data to be stored
classified_directory = r'classes'


# Define the five classification categories and their corresponding folders
categories = {
    'L': 'left_loop',
    'W': 'whirl',
    'R': 'right_loop',
    'T': 'tented_arch',
    'A': 'arch'
}

# Create folders for each category in the classified directory if they don't exist
for category in categories.values():
    os.makedirs(os.path.join(classified_directory, category), exist_ok=True)

# Function to copy files with retry mechanism and closing file handles if necessary
def copy_file(src, dst, retries=3, delay=1):
    for attempt in range(retries):
        try:
            shutil.copy2(src, dst)
            return
        except PermissionError:
            if attempt < retries - 1:
                time.sleep(delay)
                close_file_handles(src)
            else:
                raise

# Function to close file handles using psutil
def close_file_handles(file_path):
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            for file in proc.open_files():
                if file.path == file_path:
                    print(f"Closing handle for file {file_path} held by process {proc.info['name']} (PID: {proc.info['pid']})")
                    proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue

# Loop through each subdirectory in the main directory
for subdir in os.listdir(main_directory):
    subdir_path = os.path.join(main_directory, subdir)
    
    if os.path.isdir(subdir_path):
        # Loop through each file in the subdirectory
        for file in os.listdir(subdir_path):
            if file.endswith('.txt'):
                txt_file_path = os.path.join(subdir_path, file)
                png_file_path = os.path.join(subdir_path, file.replace('.txt', '.png'))
                
                # Read the classification from the text file
                with open(txt_file_path, 'r') as f:
                    content = f.readlines()
                    # Extract the class from the specific line
                    for line in content:
                        if line.startswith('Class:'):
                            classification = line.split(': ')[1].strip()
                            # Copy the files to the corresponding category folder
                            if classification in categories:
                                category_folder = os.path.join(classified_directory, categories[classification])
                                try:
                                    copy_file(txt_file_path, os.path.join(category_folder, file))
                                    copy_file(png_file_path, os.path.join(category_folder, file.replace('.txt', '.png')))
                                except PermissionError as e:
                                    print(f"Failed to copy {file} after multiple retries: {e}")
                                break

print("Files have been classified and copied to respective folders.")
