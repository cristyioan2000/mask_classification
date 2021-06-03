import os
import cv2
import numpy as np
from utils.bbox_utils import get_crop

images_path = r'D:\facultate\Disertatie\Datasets\mask\mask_images.txt'

# Creates folder for the output dir
output_root = r'D:\facultate\Disertatie\Datasets\mask\crop_dataset'
if not os.path.exists(output_root):
    os.mkdir(output_root)
    print(f"Created dir: {output_root}..")

# Read all image paths
with open(images_path,'r') as reader:
    images_list = [line.strip() for line in reader.readlines()]

# Return label path
def get_label(image):
    return image.replace('.jpg','.txt').replace('.png','.txt')
# Create gt file
def create_classification_gt(image_file_name, index, output_root, class_id):
    gt_file_name = f"{image_file_name.split('.')[0]}_{index}.txt"
    gt_output_file_name = os.path.join(output_root, gt_file_name)
    with open(gt_output_file_name,'w') as wr:
        wr.write(str(int(class_id)))

for image_file in images_list:
    img = cv2.imread(image_file)
    h, w, _ = img.shape
    # Get label
    label_file = get_label(image_file)
    gt = np.loadtxt(label_file).reshape(-1,5)
    image_file_name = os.path.basename(image_file)
    for index, label in enumerate(gt):
        crop = get_crop(img,label)
        create_classification_gt(image_file_name, index, output_root, label[0])
        crop_file_output = os.path.join(output_root,f"{image_file_name.split('.')[0]}_{index}.jpg")
        try:
            crop = cv2.resize(crop,(128,128))
        except Exception as e:
            print(e)

        cv2.imwrite(crop_file_output,crop)

