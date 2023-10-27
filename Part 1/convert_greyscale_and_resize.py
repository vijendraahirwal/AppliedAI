from PIL import Image
import os

# Define the source folder
source_dir = "AppliedAI/Data"

# Set the desired size of image in pixels and DPI
new_image_size = (48, 48)
dpi = (96, 96)

# Use os.walk to traverse all subdirectories and their contents
for root, dirs, files in os.walk(source_dir):
    for img_file in files:
        # Check if the file is an image and image extensions can be added when needed.
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Construct the full path to the image
            image_path = os.path.join(root, img_file)
            
            # Open the image
            with Image.open(image_path) as img:
                # Resize the image to the desired size
                resized_img = img.resize(new_image_size)
                
                # Convert the image to grayscale
                grayscale_img = resized_img.convert("L")

                # Save the grayscale image back to the same path, overwriting the original
                grayscale_img.save(image_path, dpi=dpi)

print("Successfully resized and converted images to grayscale in-place.")
