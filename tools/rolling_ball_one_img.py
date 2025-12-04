import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
from matplotlib import pyplot as plt
import pyclesperanto_prototype as cle

output_path = '/d01/scholles/gigasistemica/gigasistemica_sandbox_scholles/tools/image_folder_test_cv/'
image_path = "/d01/scholles/gigasistemica/datasets/CVAT_train/CVAT_Train_Saudavel_Grave_Croped_600x600/train/Grave/img_OPHUB2018-146.jpg_bbox_[1733.27, 1109.41, 485.17, 58.4].jpg"

image = cv2.imread(image_path, 0)
cv2.imwrite(output_path + 'original_image.jpg', image)

radius = 180
print('Performing Rolling Ball')
final_img = cle.top_hat_sphere(image, radius_x=400, radius_y=300)

# Convert final_img to NumPy array
final_img = cle.pull(final_img)

cv2.imwrite(output_path + 'cv_output_image.jpg', final_img)