from skimage import io, restoration
import numpy as np

# Load the image
image_path = '/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Grave/img_OPHUB2018-146.jpg_bbox_[1733.27, 1109.41, 485.17, 58.4].jpg'  # Replace with the correct image path
image = io.imread(image_path)
image = np.array(image)

# Perform Rolling Ball
print('Performing Rolling Ball')
background = restoration.rolling_ball(image)
arr = image - background

# Path to save the images
output_path = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/tools/image_folder_test/'

# Save the images with different names
io.imsave(output_path + 'original_image.jpg', image)
io.imsave(output_path + 'estimated_background.jpg', background)
io.imsave(output_path + 'output_image.jpg', arr)

print('Images saved successfully!')