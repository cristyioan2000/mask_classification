from utils.conversion import darknet2px

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