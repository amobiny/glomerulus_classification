import PIL
from torchvision import transforms
import h5py
import torch
import numpy as np
from config import options
from torch.utils.data import Dataset


class Data(Dataset):
    def __init__(self, mode='train', data_len=None):

        self.mode = mode
        if mode == 'train':
            print('Loading the training data...')
            train_idx = list(range(10))
            train_idx.remove(options.loos)
            images = np.zeros((0, options.img_c, options.img_h, options.img_w))
            labels = np.array([])
            for idx in train_idx:
                h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/'
                                'glomerulus_classification/dataset/data_5C_{}.h5'.format(idx), 'r')
                x = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)
                y = h5f['y'][:].astype(int)
                images = np.concatenate((images, x), axis=0)
                labels = np.append(labels, y)
                h5f.close()
            self.images = images
            self.labels = labels
        elif mode == 'test':
            print('Loading the test data...')
            h5f = h5py.File('/home/cougarnet.uh.edu/amobiny/Desktop/'
                            'glomerulus_classification/dataset/data_5C_{}.h5'.format(options.loos), 'r')
            self.images = np.transpose(h5f['x'][:], [0, 3, 1, 2]).astype(int)[:data_len]
            self.labels = h5f['y'][:].astype(int)[:data_len]
            h5f.close()

    def __getitem__(self, index):

        # img = torch.tensor(self.images[index]).div(255.).float()

        img = torch.tensor(self.images[index]).float()
        img = (img - img.min()) / (img.max() - img.min())

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
            img = transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=.05, saturation=.05)(img)
            # img = transforms.RandomResizedCrop(options.img_h, scale=(0.7, 1.))(img)
            # img = transforms.RandomRotation(90, resample=PIL.Image.BICUBIC)(img)
            img = transforms.ToTensor()(img)

        # img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        return len(self.labels)
