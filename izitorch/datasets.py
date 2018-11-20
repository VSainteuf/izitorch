import h5py
import torch
import torch.utils.data as data
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import ToTensor
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm


class Sequential_Dataset_from_h5folder(data.Dataset):

    def __init__(self, folder, labels='label_44class', extra_feature=None, im_transforms=None, extra_transform=None):
        super(Sequential_Dataset_from_h5folder, self).__init__()
        self.folder_path = folder
        self.labels = labels
        self.extra_feature = extra_feature
        self.im_transforms = im_transforms
        self.extra_transform = extra_transform
        self.dataset_list = np.sort([f for f in os.listdir(folder) if str(f).endswith('.h5')])
        with h5py.File(os.path.join(folder, self.dataset_list[0]), 'r') as h5:
            self.chunksize = h5['images'].shape[0]
        self.len = None

    def __getitem__(self, index):
        dataset_number = index // self.chunksize
        local_index = index % self.chunksize

        with h5py.File(os.path.join(self.folder_path, self.dataset_list[dataset_number]), 'r') as h5:

            image_series = (torch.from_numpy(h5['images'][local_index, :, :, :, :]).float())
            if self.im_transforms is not None:
                for i in range(image_series.size()[0]):
                    image_series[i, :, :, :] = (self.im_transforms[i])(image_series[i, :, :, :])

            if self.extra_feature is not None:
                extra = torch.from_numpy(np.array(h5[self.extra_feature][local_index], dtype=int)).float()
                if self.extra_transform is not None:
                    m, s = self.extra_transform
                    extra = (extra - m) / s
                data = (image_series, extra)
            else:
                data = image_series

            labels = torch.from_numpy(np.array(h5[self.labels][local_index], dtype=int))
        return data, labels

    def __len__(self):
        if self.len is None:
            l = 0
            for d in self.dataset_list:
                with h5py.File(os.path.join(self.folder_path, d), 'r') as h5:
                    l += h5['images'].shape[0]
            self.len = l

        return self.len

    def compute_stats(self, out_path='./meanstd.pkl'):
        dl = data.DataLoader(self, batch_size=100, shuffle=False, num_workers=6)

        m = []
        s = []

        xm = []
        xs = []

        for x, y in tqdm(dl):

            if self.extra_feature is not None:
                x, extra = x
                xm.append(np.mean(extra.numpy(), axis=0))
                xs.append(np.std(extra.numpy(), axis=0))

            m.append(np.mean(x.numpy(), axis=(0, 3, 4)))
            s.append(np.std(x.numpy(), axis=(0, 3, 4)))
        m = np.mean(m,axis=0)
        s = np.mean(s,axis=0)
        xm = np.mean(xm, axis=0)
        xs = np.mean(xs, axis=0)
        stats = {'images': (m, s), 'extra': (xm, xs)}
        pkl.dump(stats, open(out_path, 'wb'))


class Unitemporal_Dataset_from_h5folder(data.Dataset):

    def __init__(self, folder, date_number=12, labels='label_44class', extra_feature=None, im_transform=None,
                 extra_transform=None):
        super(Unitemporal_Dataset_from_h5folder, self).__init__()
        self.folder_path = folder
        self.date_number = date_number
        self.labels = labels
        self.extra_feature = extra_feature
        self.im_transform = im_transform
        self.extra_transform = extra_transform
        self.dataset_list = np.sort([f for f in os.listdir(folder) if str(f).endswith('.h5')])
        with h5py.File(os.path.join(folder, self.dataset_list[0]), 'r') as h5:
            self.chunksize = h5['images'].shape[0]
        self.len = None

    def __getitem__(self, index):
        dataset_number = index // self.chunksize
        local_index = index % self.chunksize

        with h5py.File(os.path.join(self.folder_path, self.dataset_list[dataset_number]), 'r') as h5:

            images = (torch.from_numpy(h5['images'][local_index, self.date_number, :, :, :]).float())
            if self.im_transform is not None:
                images = (self.im_transform)(images)

            if self.extra_feature is not None:
                extra = torch.from_numpy(np.array(h5[self.extra_feature][local_index], dtype=int)).float()
                if self.extra_transform is not None:
                    m, s = self.extra_transform
                    extra = (extra - m) / s
                data = (images, extra)
            else:
                data = images

            labels = torch.from_numpy(np.array(h5[self.labels][local_index], dtype=int))
        return data, labels

    def __len__(self):
        if self.len is None:
            l = 0
            for d in self.dataset_list:
                with h5py.File(os.path.join(self.folder_path, d), 'r') as h5:
                    l += h5['images'].shape[0]
            self.len = l

        return self.len

    def compute_stats(self, out_path=None):
        dl = data.DataLoader(self, batch_size=10000, shuffle=False, num_workers=6)

        means = []
        stds = []

        for x, y in tqdm(dl):
            means.append(np.mean(x.numpy(), axis=(0, 2, 3)))
            stds.append(np.std(x.numpy(), axis=(0, 2, 3)))
        self.means = np.mean(means, axis=0)
        self.stds = np.mean(stds, axis=0)

        if out_path is not None:
            stats = (self.means, self.stds)
            pkl.dump(stats, open(out_path, 'wb'))
