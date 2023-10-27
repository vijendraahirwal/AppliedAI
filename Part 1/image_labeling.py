import os

# Define the source directory
source_dir = "AppliedAI/Data"

# Function to rename files in a directory
def rename_files_in_directory(directory_path):
    # Get the directory name
    directory_name = os.path.basename(directory_path)

    # Get a list of all files in the directory
    files = os.listdir(directory_path)

    # Initialize a sequence number
    sequence_number = 1

    for file in files:
        # Construct the new file name
        new_name = f"{directory_name}_{sequence_number:03}"
        
        # Get the file extension (if any)
        file_name, file_extension = os.path.splitext(file)

        # Construct the full path to the original file
        original_file_path = os.path.join(directory_path, file)

        # Construct the full path to the new file
        new_file_path = os.path.join(directory_path, f"{new_name}{file_extension}")

        # Rename the file
        os.rename(original_file_path, new_file_path)

        # Increment the sequence number
        sequence_number += 1

# Use os.walk to traverse all subdirectories and their contents
for root, dirs, files in os.walk(source_dir):
    for directory in dirs:
        # Construct the full path to the subdirectory
        directory_path = os.path.join(root, directory)

        # Rename files in the subdirectory
        rename_files_in_directory(directory_path)

print("File renaming completed.")
