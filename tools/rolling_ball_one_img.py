import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle

output_path = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/tools/image_folder_test_cv/'
image_path = "/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Grave/img_OPHUB2018-146.jpg_bbox_[1733.27, 1109.41, 485.17, 58.4].jpg"
#image_path= '/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Saudavel/img_OPHUB2015-77.jpg_bbox_[721.07, 1124.36, 378.42, 76.73].jpg'

# Read the image
image = cv2.imread(image_path, 0)
# Save the original image
cv2.imwrite(output_path + 'original_image.jpg', image)

# Set the radius for rolling ball algorithm
radius = 180
# Subtract the background using the rolling ball algorithm
print('Performing Rolling Ball')
final_img = cle.top_hat_sphere(image, radius_x=400, radius_y=300)

# Converta o objeto final_img para um array do NumPy
final_img = cle.pull(final_img)

# Save the estimated background and the final image with background removed
cv2.imwrite(output_path + 'cv_output_image.jpg', final_img)