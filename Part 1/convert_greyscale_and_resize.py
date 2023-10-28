from PIL import Image
import os

# the source folder
source_dir = "AppliedAI/Data"

# desired size of image
new_image_size = (48, 48)
dpi = (96, 96)

# traversing all subdirectories
for root, dirs, files in os.walk(source_dir):
    for img_file in files:
        
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Constructing the full path
            image_path = os.path.join(root, img_file)
        
            with Image.open(image_path) as img:
                # Resizing
                resized_img = img.resize(new_image_size)
                
                # Converting to Grayscale
                grayscale_img = resized_img.convert("L")

                #saving
                grayscale_img.save(image_path, dpi=dpi)

print("Successfully resized and converted images to grayscale in-place.")
