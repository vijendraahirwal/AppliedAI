from PIL import Image
import os

# Specify the source and destination folders
source_folder = r"Part 1\Unprocessed\greyscale_bored_faces"
destination_folder = r"Part 1\Unprocessed\resized_bored_faces"

# Set the desired size in pixels and DPI
new_size = (48, 48)
dpi = (96, 96)
# Set the desired size in pixels and DPI
new_size = (48, 48)
dpi = (96, 96)

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Get a list of all files in the source folder
files = os.listdir(source_folder)

# Loop through the files in the source folder
for file in files:
    # Check if the file is an image (you can add more image extensions if needed)
    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        # Open the image
        with Image.open(os.path.join(source_folder, file)) as img:
            # Resize the image to the desired size
            resized_img = img.resize(new_size)

            # Save the resized image to the destination folder with DPI
            resized_img.save(os.path.join(destination_folder, file), dpi=dpi)

print("Images resized to 48x48 pixels and set to 96 DPI, and saved to the destination folder.")