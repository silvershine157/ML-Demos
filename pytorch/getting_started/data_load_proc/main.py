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

inspect_dataset()
