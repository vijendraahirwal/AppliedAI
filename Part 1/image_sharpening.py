import os
import cv2


def preprocess_images_in_directories(base_dir):
   
    def preprossesing_image(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Applying Gaussian blur
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        # Apply unsharp masking to enhance image sharpness
        gauss = cv2.GaussianBlur(blurred_image, (0, 0), 2.0)
        final_image = cv2.addWeighted(image, 2.0, gauss, -1.0, 0)
        return final_image

    for emotion_folder in ["angry", "bored", "neutral", "focused"]:
        directory_path = os.path.join(base_dir, emotion_folder)
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            for filename in os.listdir(directory_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(directory_path, filename)
                    final_image = preprossesing_image(image_path)
                    cv2.imwrite(image_path, final_image)
                
        else:
            print(f"The file '{directory_path}' does not exist.")


source_dir = "AppliedAI/Data"
preprocess_images_in_directories(source_dir)
