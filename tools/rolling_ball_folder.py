import os
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from tqdm import tqdm

def transform_images_in_folder(input_folder, output_folder):
    # Walk through all files and folders in input directory
    for root, dirs, files in os.walk(input_folder):
        # Create subfolder structure in output directory
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Process each image file in current folder
        for filename in tqdm(files, desc=f"Processing {relative_path}"):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                input_image_path = os.path.join(root, filename)
                output_image_path = os.path.join(output_subfolder, filename)

                image = cv2.imread(input_image_path, 0)

                # Rolling ball algorithm radius
                radius = 180

                # Background subtraction using rolling ball
                final_img, background = subtract_background_rolling_ball(image, radius, light_background=False,
                                                                         use_paraboloid=True, do_presmooth=True)

                # Save processed image
                cv2.imwrite(output_image_path, final_img)

if __name__ == '__main__':
    input_folder = "/mnt/ssd/brunoscholles/GigaSistemica/Datasets/DXA_Osteo_Images"
    output_folder = "/mnt/ssd/brunoscholles/GigaSistemica/Datasets/RB_DXA_Osteo_Images"
    transform_images_in_folder(input_folder, output_folder)