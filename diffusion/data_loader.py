import numpy as np
import torch
import time

def worker_init_fn(worker_id):
    np.random.seed(int(time.time()))


class Dataset:
    def __init__(self, data_path_list, transform=None, label=False):
        self.transform = transform
        self.label = label
        self.data = np.concatenate([np.load(path, allow_pickle=True).astype(np.float16) for path in data_path_list], axis=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        
        window = self.data[idx]
        if len(window.shape) == 1:
            x, y = window
        elif len(window.shape) == 2:
            x, y = window, None
        
        if self.transform is not None:
            x = self.transform(x)
            x = [xi.squeeze(0) for xi in x] if isinstance(x, list) else x.squeeze(0)
        
        if self.label and y is not None:
            return (x, y)
        else:
            return x


class NormalDataset:
    def __init__(self, x, y, name='', transform=None):
        self.transform = transform
        self.x = x
        self.y = y
        print(f'{name} set sample count: {len(self.x)}')
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        x = self.x[idx]
        y = self.y[idx]
        
        if self.transform is not None:
            x = self.transform(x)
            x = [xi.squeeze(0) for xi in x] if isinstance(x, list) else x.squeeze(0)
        
        return x, y


class SignalPathDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.signals = np.load(path, allow_pickle=True)['arr_0']
        self.transforms = transforms

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, i):
        signal = self.signals[i]
        if self.transforms is not None:
            signal = self.transforms(signal)[0]
        return signal