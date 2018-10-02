import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
import os
import torch
import numpy as np

class PneumoniaDataset(Dataset):
    """Pneumonia opacity dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pneumonia_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pneumonia_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.pneumonia_frame.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name, as_gray=True)
        image = np.expand_dims(image, 2)
        pneumonia = self.pneumonia_frame.iloc[idx, 1:].values
        #pneumonia = pneumonia.astype('float')#.reshape(1, 1)

        sample = {'image': image, 'pneumonia': pneumonia[0]}

        if self.transform:
            sample = self.transform(sample)

        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, pneumonia = sample['image'], sample['pneumonia']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'pneumonia': pneumonia}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, pneumonia = sample['image'], sample['pneumonia']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'pneumonia': pneumonia}

trainset = PneumoniaDataset(csv_file='data/img_labels.csv',
                                           root_dir='datasets/images',
                                           transform=transforms.Compose([
                                           RandomCrop(32), ToTensor()])
                            )

trainloader = DataLoader(trainset, batch_size=4,
                        shuffle=True, num_workers=4)

classes = ('0', '1')