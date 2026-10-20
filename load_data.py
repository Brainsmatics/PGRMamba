import torch
import os
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np



class dataset0(Dataset):
    def __init__(self, src, require_labels=True):
        super(dataset0, self).__init__()
        self.src = src
        self.require_labels = require_labels
        self.image_list, self.label_list = self._get_name()
        if not self.image_list:
            raise FileNotFoundError(f'No TIFF images in {self.src}')
        missing = [p for p in self.label_list if not os.path.isfile(p)]
        if self.require_labels and missing:
            raise FileNotFoundError(f'Missing CSV label: {missing[0]}')
        self.image_size = sitk.ReadImage(self.image_list[0]).GetSize() 
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.ToTensor()

    def _get_name(self):
        files = sorted(os.listdir(self.src))
        image_list, label_list = [], []
        for f in files:
            if f.endswith('.tif'):
                image_list.append(os.path.join(self.src, f))
                label_list.append(os.path.join(self.src, f[:-3] + 'csv'))
        return image_list, label_list

    def _get_label(self, label_name):
        label = []
        try:
            if (not os.path.exists(label_name)) or os.path.getsize(label_name) == 0:
                return label
            df = pd.read_csv(label_name)
        except pd.errors.EmptyDataError:
            return label
        if all(col in df.columns for col in ['X', 'Y', 'Slice']):
            x_col, y_col, z_col = 'X', 'Y', 'Slice'
        elif all(col in df.columns for col in ['x', 'y', 'slice']):
            x_col, y_col, z_col = 'x', 'y', 'slice'
        else:
            try:
                _ = [float(c) for c in df.columns]
                df = pd.read_csv(label_name, header=None)
            except Exception:
                pass
            if df.shape[1] >= 3:
                x_col, y_col, z_col = df.columns[:3]
            else:
                raise ValueError(f"Label file {label_name} must have at least 3 columns for X, Y, Slice")
        for i in range(df.shape[0]):
            label.append([df.loc[i, x_col], df.loc[i, y_col], df.loc[i, z_col], 1])
        return label

    def __getitem__(self, item):
        # print(self.image_list[item])
        image = sitk.ReadImage(self.image_list[item])
        image = sitk.GetArrayFromImage(image).astype('float32')
        image = self.transform(image)
        image = image.permute(1,0,2).unsqueeze(0)
        # label = torch.tensor([])
        label = []
        if self.require_labels:
            if not os.path.isfile(self.label_list[item]):
                raise FileNotFoundError(f'Missing CSV label: {self.label_list[item]}')
            label = self._get_label(self.label_list[item])
        # return image, torch.tensor(label).float()
        label = np.asarray(label, dtype='float32').reshape(-1, 4)
        image = np.array(image, dtype='float32')
        return image, label


    def __len__(self):
        return len(self.image_list)

def dataset0_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    # Keep variable-length labels per sample, including zero-label samples.
    bboxes = tuple(bboxes)
    return images, bboxes

if __name__ == '__main__':
    data0 = dataset0(src='./train_data1')
    data_loader = DataLoader(dataset=data0, shuffle=True, batch_size=8, collate_fn=dataset0_collate)
    for image, label in data_loader:
        print(image.shape)
        # print(label)
