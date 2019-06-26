import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings('ignore')

plt.ion()

visual_path = './data/visual/'

def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(0.001)

def preview_landmarks():
    landmarks_frame = pd.read_csv('data/faces/face_landmarks.csv')

    n = 65 # arbitrary img index
    img_name = landmarks_frame.iloc[n, 0]
    landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
    landmarks = landmarks.astype('float').reshape(-1, 2)

    print('Image name: {}'.format(img_name))
    print('Landmakrs shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    plt.figure()
    show_landmarks(io.imread(os.path.join('data/faces/', img_name)), landmarks)
    plt.savefig(visual_path+'show_landmarks.png')

#preview_landmarks()


## Dataset class

class FaceLandmarksDataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self. root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


# create dataset instance
face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')

def inspect_dataset():
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv', root_dir='data/faces/')
    fig = plt.figure()
    for i in range(len(face_dataset)):
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)
        ax = plt.subplot(1, 4, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks(**sample)
        if i==3:
            plt.savefig(visual_path+'inspect_dataset.png')
            break

#inspect_dataset()

## Transforms

class Rescale(object):
    
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            # fit short side, preserve aspect ratio
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        landmarks = landmarks * [new_w / w, new_h / h] # sneaky!
        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top] # sneaky!

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image.transpose((2, 0, 1)) # H x W x C -> C x H x W
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

def transform_demo():
    scale = Rescale(256)
    crop = RandomCrop(128)
    composed = transforms.Compose([Rescale(256), RandomCrop(224)])

    fig = plt.figure()
    sample = face_dataset[65]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i+1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        show_landmarks(**transformed_sample)

    plt.savefig(visual_path + 'transform_demo.png')

transform_demo()
