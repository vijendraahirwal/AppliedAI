import os
import csv
from PIL import Image

# Function to get user input for gender and age
def get_user_input(image_path):
    # Display the image
    image = Image.open(image_path)
    image.show()

    gender = input("Enter gender (m/f): ").lower()
    age = input("Enter age category (1 for young, 2 for middle_age, 3 for senior): ")

    # Close the image window
    image.close()

    return gender, age

# Path to the main directory
main_directory = "AppliedAI/Part_2/dataset_inshape"

# Output CSV file
csv_file_path = "labels.csv"

# Open CSV file for writing
with open(csv_file_path, 'w', newline='') as csvfile:
    # Create CSV writer
    csvwriter = csv.writer(csvfile)

    # Write header
    csvwriter.writerow(['Image Path', 'Gender', 'Age Category'])

    # Iterate through each category folder in the train directory
    for category_folder in os.listdir(os.path.join(main_directory, 'train')):
        category_folder_path = os.path.join(main_directory, 'train', category_folder)

        # Iterate through each image in the category folder
        for image_file in os.listdir(category_folder_path):
            # Get the full path of the image
            image_path = os.path.join(category_folder_path, image_file)

            # Get user input for gender and age
            gender, age = get_user_input(image_path)

            print("WRITING IT NOW")
            # Write the information to the CSV file
            csvwriter.writerow([image_path, gender, age])
            
            

print(f"Labels saved to {csv_file_path}")
