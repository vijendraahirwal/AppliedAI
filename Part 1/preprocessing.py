import os
import cv2

def preprocess_images_in_directories(base_dir):
   
    def image_preprocessing(image_path):
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #blurring the image before sharping it to avoid any unwanted edges in the sharp image
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        
        #appling unsharp masking technique to sharpen the image
        gauss = cv2.GaussianBlur(image, (0, 0), 2.0)
        image_unsharp_masking_final_image = cv2.addWeighted(image, 2.0, gauss, -1.0, 0)
        
        return image_unsharp_masking_final_image

    for dir_name in ["angry","bored","neutral","focused"]:
        directory_path = os.path.join(base_dir, dir_name)
        
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
          
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    image_path = os.path.join(directory_path, filename)
                    
                    
                    final_image = image_preprocessing(image_path)
                    cv2.imwrite(image_path, final_image)  
                    print(f'Processed and saved: {filename}')
        else:
            print(f"The directory '{directory_path}' does not exist.")

base_directory = 'C:\\Users\\pa_ronak\\Downloads'
preprocess_images_in_directories(base_directory)
