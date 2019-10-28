import PIL
from torchvision import datasets, transforms
import h5py
import torch
import numpy as np
from config import options


class Data:
    def __init__(self, mode='train'):

        h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/glomerulus_classification/dataset/data.h5', 'r')
        self.mode = mode
        if mode == 'train':
            self.images = np.transpose(h5f['x_train'][:], [0, 3, 1, 2])
            self.labels = h5f['y_train'][:].astype(int)
        elif mode == 'test':
            self.images = np.transpose(h5f['x_test'][:], [0, 3, 1, 2])
            self.labels = h5f['y_test'][:].astype(int)
        h5f.close()

    def __getitem__(self, index):

        img = torch.tensor(self.images[index]).div(255.).float()
        if self.labels[index] == 2:
            self.labels[index] = 1
        if self.labels[index] == 4:
            self.labels[index] = 2
        if self.labels[index] == 3:
            self.labels[index] = 2
        target = torch.tensor(self.labels[index])

        if self.mode == 'train':
            # normalization & augmentation
            img = transforms.ToPILImage()(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.RandomVerticalFlip()(img)
            # img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
            # img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)

        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.labels)
