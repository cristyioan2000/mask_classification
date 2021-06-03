# Created by tudorc at 01.07.2019
"""
    Plot yolo labels on images. Just to test labels values in txt files
"""

import cv2
import os
from glob import glob

def clean_img_lbl_numbers(g_img, g_lbl, g_att):
    """
        Delete images that don't have labels or labels that don't have images

        Args:
            g_img(lst): sorted glob with all images
            g_lbl(lst): sorted glob with all labels
            g_att(lst): sorted glob with all attributes
        Returns:
            None
    """

    img_lst = [os.path.basename(img[:img.rindex('.')]) for img in g_img]
    lbl_lst = [os.path.basename(lbl[:lbl.rindex('.')]) for lbl in g_lbl]
    att_lst = [os.path.basename(lbl[:lbl.rindex('.')]) for lbl in g_lbl]

    for img in img_lst:
        if img not in lbl_lst or img not in att_lst:
            # MARK: delete images w/o labels or attributes
            print(f'to delete img: {os.path.join(os.path.dirname(g_img[0]), img + ".jpg")}')
            os.remove(os.path.join(os.path.dirname(g_img[0]), img + ".jpg"))

    for lbl in lbl_lst:
        if lbl not in img_lst or lbl not in att_lst:
            # MARK: delete labels w/o images or attributes
            print(f'to delete lbl: {os.path.join(os.path.dirname(g_lbl[0]), lbl + ".txt")}')
            os.remove(os.path.join(os.path.dirname(g_lbl[0]), lbl + ".txt"))

    for att in att_lst:
        if att not in img_lst or att not in lbl_lst:
            # MARK: delete attributes w/o images or labels
            print(f'to delete att: {os.path.join(os.path.dirname(g_att[0]), att + ".att")}')
            os.remove(os.path.join(os.path.dirname(g_att[0]), att + ".att"))

    return None


def plot_single_image(img_pth, lbl_pth):
    """
        Plot w/ opencv all labels from a txt for a jpg

        Args:
            img_pth(str): path to image
            lbl_pth(str): path to label
        Returns:
            None
    """

    img = cv2.imread(img_pth, cv2.COLOR_BGR2RGB)
    img_h = img.shape[0]
    img_w = img.shape[1]

    with open(lbl_pth, 'r') as lbl:
        labels = lbl.readlines()

        for label in labels:
            tokens = label.split(' ')

            # cls = int(tokens[0])
            cls = tokens[0]
            bb_c_x = float(tokens[1])
            bb_c_y = float(tokens[2])
            bb_w = float(tokens[3])
            bb_h = float(tokens[4])
            # print('\nLabel: {} - at ({},{}) with shape ({}{})'.format(cls, bb_c_x, bb_c_y, bb_w, bb_h))

            real_bb_c_x = bb_c_x * img_w
            real_bb_c_y = bb_c_y * img_h
            real_bb_w = bb_w * img_w
            real_bb_h = bb_h * img_h

            corner_01 = (int(real_bb_c_x - (real_bb_w / 2)), int(real_bb_c_y + (real_bb_h / 2)))
            corner_02 = (int(real_bb_c_x + (real_bb_w / 2)), int(real_bb_c_y - (real_bb_h / 2)))

            cls_color = (0, 255, 0)

            cv2.rectangle(img, corner_01, corner_02, cls_color, 2)

    # cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow('image', 700, 700)

    # cv2.imshow(window, img)
    cv2.imshow('image', img)
    # cv2.imwrite(path,img)
    key = cv2.waitKey(0)

    # TODO: next prev exit keybindings
    if key == ord('d'):
        print('d')
    elif key == ord('a'):
        print('b')
    elif key == ord('w'):
        print('w')
    elif key == ord('s'):
        print('s')
    elif key == 27:
        exit()

    cv2.destroyAllWindows()

    return None


def main():


    img_path = r'D:\facultate\Disertatie\Datasets\mask\crop_test\Mask_366.jpg'
    label_path = img_path.replace('.jpg','.txt')
    # label_path = img_path.replace('.png','.txt').replace('/images/','/labels/')
    plot_single_image(img_path,label_path)

if __name__ == '__main__':
    main()
