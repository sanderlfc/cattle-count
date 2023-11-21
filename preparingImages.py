import os
import numpy as np
import cv2


#declare paths to working directory:
work_path = r'C:\Users\amand\Desktop\training_data_cattle'
output_path = os.path.join(work_path, 'preprocessed_images')
train_path = os.path.join(work_path, 'images', 'train')


# Create the output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

#this function is for preprocessing the images and augmentating (rotation, flip, resize):
def img_preprocessing(img_path, folder_result, augmentations=3):
    # Read the image
    original_image = cv2.imread(img_path)

    # Using Open CV to resize the images to 640x640:
    resized_image = cv2.resize(original_image, (640, 640))

    # Iterate and repeate augmentation and save the resulting images:
    for i in range(augmentations):

        flipped_image = cv2.flip(resized_image, 1) if np.random.rand() < 0.5 else resized_image         # Horizontal flipping with 50% probability

        angle_img = np.random.uniform(-10, 10)         # Rotation with a random angle between -10 and 10 degrees
        rotation_matrix = cv2.getRotationMatrix2D((flipped_image.shape[1] // 2, flipped_image.shape[0] // 2), angle_img, 1) #rotate image
        rotated_img = cv2.warpAffine(flipped_image, rotation_matrix, (flipped_image.shape[1], flipped_image.shape[0])) #using trasformation

        # Save images:
        output_file = os.path.join(folder_result, f"{os.path.basename(img_path)[:-4]}_aug_{i + 1}.jpg")
        cv2.imwrite(output_file, rotated_img)

# Process each image in the training directory
for filename in os.listdir(train_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(train_path, filename)
        img_preprocessing(image_path, output_path)

print("Preprocessing and augmentation completed.")