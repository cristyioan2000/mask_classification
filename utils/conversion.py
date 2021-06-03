import numpy as np

def darknet2px(label, img_w, img_h):
    cls = int(label[0])
    bb_c_x = float(label[1])
    bb_c_y = float(label[2])
    bb_w = float(label[3])
    bb_h = float(label[4])
    # print('\nLabel: {} - at ({},{}) with shape ({}{})'.format(cls, bb_c_x, bb_c_y, bb_w, bb_h))

    real_bb_c_x = bb_c_x * img_w
    real_bb_c_y = bb_c_y * img_h
    real_bb_w = bb_w * img_w
    real_bb_h = bb_h * img_h

    corner_01 = (int(real_bb_c_x - (real_bb_w / 2)), int(real_bb_c_y + (real_bb_h / 2)))
    corner_02 = (int(real_bb_c_x + (real_bb_w / 2)), int(real_bb_c_y - (real_bb_h / 2)))
    # return corner_02[0], corner_02[1], int(real_bb_w), int(real_bb_h)
    return corner_01,corner_02
