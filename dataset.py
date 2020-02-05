# -*- coding=utf8 -*-
'''
@Filename  : dataset.py
@Author    : Gaozong
@Date      : 2020-02-04
@Contact   : zong209@163.com
@Describe  : Dataset class of cats and dogs
'''

import os
# import cv2
# import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    def __init__(self, root_dir, suffix, transform=None):
        """
        Args:
        root_dir: root directory of image
        """
        self.root_dir = root_dir
        self.suffix = suffix
        self.transform = transform
        self.dataset = self._list_images()

    def _list_images(self):
        '''
        return files list in directory
        '''
        image_list = list()
        class_index = 0
        for category in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, category)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith(self.suffix):
                        image_list.append(
                            [os.path.join(class_dir, file), class_index])
            class_index += 1
        return image_list

    def _load_image(self, path):
        img = Image.open(path)
        # 填充图片至正方形，防止形变
        # shape = img.shape
        # size = max(shape[0], shape[1])
        # new_array = np.ones((size, size)) * img[0][0][0]
        # background = Image.fromarray(new_array)
        # background.paste(Image.fromarray(img), (0, 0, shape[1], shape[0]))
        # return background.convert("RGB")
        return img.convert("RGB")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        imageinfo = self.dataset[index]
        image_data = self._load_image(imageinfo[0])
        process_data = self.transform(image_data)
        return process_data, imageinfo[1]


if __name__ == "__main__":
    dataset = AnimalDataset("data/train", "jpg")
    print(dataset[0][0])
