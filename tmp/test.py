import numpy as np
import cv2
from utils.conversion import darknet2px

image_path = r'D:\facultate\Disertatie\Datasets\mask\crop_test\Mask_366.jpg'
label_path = r'D:\facultate\Disertatie\Datasets\mask\crop_test\Mask_366.txt'


image = cv2.imread(image_path)
gt = np.loadtxt(label_path).reshape(-1,5)

h, w,_ = image.shape
for label in gt:
    # Returns C2, bbox_w, bbox_h
    x2, y2, bbox_w, bbox_h = darknet2px(label, img_h=h, img_w=w)
    crop = image[y2:-bbox_h, x2-bbox_w:x2+bbox_w, :]

cv2.imshow('window',crop)
cv2.waitKey(0)
cv2.destroyAllWindows()
