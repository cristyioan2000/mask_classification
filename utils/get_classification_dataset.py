import os
import numpy as np

images_root = r'D:\facultate\Disertatie\Datasets\mask\yolo'
labels_root = r''

# images_list = [os.path.join(images_root, image_file) for image_file in os.listdir(images_root)]
# labels_list = [os.path.join(labels_root, label_file) for label_file in os.listdir(labels_root)]
file_list = [os.path.join(images_root, image_file) for image_file in os.listdir(images_root)]

images_list = []

dataset_dict = {
    '0':[],
    '1':[]
}

for image_file in file_list:
    if '.png' in image_file or '.jpg' in image_file:
        # Get label
        label_file = image_file.replace('.png','.txt').replace('.jpg','.txt')
        # Load coords + cls index
        gt = np.loadtxt(label_file).reshape(-1,5)
        # Discard multiple classes in the same image
        unique_classes = np.unique(gt[:,0]).size
        if unique_classes > 1:
            continue
        else:
            class_id = str(int(gt[:,0][0]))
            # Store image path
            dataset_dict[class_id].append(image_file)

for image in file_list:
    print('')