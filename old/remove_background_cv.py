import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt

output_path = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/tools/image_folder_test_cv/'
image_path = "/d01/scholles/gigasistemica/datasets/CVAT_raw/RAW_Osteo_CVAT_Croped/C1/OPHUB2019-398[1767.31, 999.82, 475.85, 83.18].jpg"
#image_path= '/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Saudavel/img_OPHUB2015-77.jpg_bbox_[721.07, 1124.36, 378.42, 76.73].jpg'

# Read the image
image = cv2.imread(image_path, 0)
# Save the original image
cv2.imwrite(output_path + 'original_image.jpg', image)

# Set the radius for rolling ball algorithm
radius = 180
# Subtract the background using the rolling ball algorithm
print('Performing Rolling Ball')
final_img, background = subtract_background_rolling_ball(image, radius, light_background=False,
                                         use_paraboloid=True, do_presmooth=True)

# Save the estimated background and the final image with background removed
cv2.imwrite(output_path + 'estimated_background.jpg', background)
cv2.imwrite(output_path + 'cv_output_image.jpg', final_img)