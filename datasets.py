import torch
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from utils.conversion import darknet2px
from torchvision import transforms
# Read pandas


class MaskDetection(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, files):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.files = files

    def get_crop(self,img, label):
        h, w, _ = img.shape
        # x2, y2, bbox_w, bbox_h = darknet2px(label, img_h=h, img_w=w)
        c1, c2 = darknet2px(label, img_h=h, img_w=w)

        # img = cv2.putText(img, "Point 1", (c1), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
        # img = cv2.putText(img, "Point 2", (c2), cv2.FONT_HERSHEY_COMPLEX,1, (255,255,255), 2, cv2.LINE_AA)
        # cv2.rectangle(img,(c1),(c2),(255,255,255),2)

        crop = img[c2[1]:c1[1], c1[0]:c2[0], :]
        return crop
        # return img
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        self.crop_list = []
        self.crop_id_list = []

        image_path = self.files[idx]
        label_path = image_path.replace('.jpg','.txt')
        image = cv2.imread(image_path)
        # image = cv2.resize(image,(416,416))

        gt = np.loadtxt(label_path)

        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)



        return torch.from_numpy(image).float(), gt

def __main__():
    file_paths = r'D:\facultate\Disertatie\Datasets\mask\face_mask_crop_dataset.txt'
    with open(file_paths, 'r') as reader:
        train_files = [file.strip() for file in reader.readlines()]
    dataset = MaskDetection(train_files)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    train_image, label = next(iter(train_dataloader))
    # batch_size, channels, height, width
    # h,w,c
    # image = train_image.detach().cpu().numpy().transpose(2,3,1,0).squeeze(3)
    image = train_image[0].cpu().float().numpy().transpose(1, 2, 0)
    # cv2.imshow('w',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(r'D:\facultate\Disertatie\Datasets\mask\crop_test\out.jpg',cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

if __name__=="__main__":
    __main__()