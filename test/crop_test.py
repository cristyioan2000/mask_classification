import numpy as np
import cv2
from utils.conversion import darknet2px
import os

def get_crop( img, label):
    h, w, _ = img.shape
    # x2, y2, bbox_w, bbox_h = darknet2px(label, img_h=h, img_w=w)
    c1,c2 = darknet2px(label, img_h=h, img_w=w)

    # img = cv2.putText(img, "Point 1", (c1), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
    # img = cv2.putText(img, "Point 2", (c2), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
    # cv2.rectangle(img,(c1),(c2),(255,255,255),2)

    crop = img[c2[1]:c1[1], c1[0]:c2[0],:]
    return crop
    # return img

def __main__():
    img_name = r'D:\facultate\Disertatie\Datasets\mask\yolo\Mask_116.jpg'
    lbl_name = img_name.replace('.jpg','.txt')
    img_crop_name = r'D:\facultate\Disertatie\Datasets\mask\crop_test'
    img = cv2.imread(img_name)

    gt = np.loadtxt(lbl_name).reshape(-1,5)
    for index,label in enumerate(gt):
        crop = get_crop(img, label)
        out_file = os.path.join(img_crop_name,f"out_{index}.jpg")
        cv2.imwrite(out_file,crop)

if __name__ =="__main__":
    __main__()