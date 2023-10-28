import cv2
import os

# source path
source_dir = 'AppliedAI/Data'

# XML cascade classifier file for detecting face
cascade_classifier_path = "AppliedAI/Part 1/haarcascade_frontalface_alt2.xml"

# Loading cascade file
face_cascade_classifier = cv2.CascadeClassifier(cascade_classifier_path)

# Function creation to detect and save faces in an image
def detect_and_save_faces_in_place(image_path):
    try:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade_classifier.detectMultiScale(gray_image, 1.1, 4)
        
        # If faces are detected
        if len(detected_faces) > 0:
            # Loop through detected faces and save them
            for i, (x, y, w, h) in enumerate(detected_faces):
                face = image[y:y + h, x:x + w]
                face_filename = f"{os.path.splitext(os.path.basename(image_path))[0]}_face{i}.jpg"
                face_path = os.path.join(os.path.dirname(image_path), face_filename)
                cv2.imwrite(face_path, face)

            # Removal
            os.remove(image_path)
        else:
            # If no faces, delete the photo
            os.remove(image_path)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")


for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(root, file)
            detect_and_save_faces_in_place(image_path)
