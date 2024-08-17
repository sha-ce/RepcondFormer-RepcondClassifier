
# copied https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool1d, softmax
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionForSignal

PRE_TRAINED_PATH = './results/bests.pth'

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


def get_activations(loader, model, num_classes, device='cpu'):
    model.eval()
    dims = [k for k, v in model.BLOCK_INDEX_BY_DIM.items() if v == model.output_blocks[0]][0]
    pred_arr = np.empty((len(loader.dataset), dims))
    pred_fc_arr = np.empty((len(loader.dataset), num_classes))

    start_idx = 0
    for batch in tqdm(loader):
        batch = batch.to(device, dtype=float)

        with torch.no_grad():
            pred, pred_fc = model.forward_eval(batch)
            pred_fc = softmax(pred_fc, dim=1)

        pred = pred.squeeze(2).squeeze(1).cpu().numpy()
        pred_fc = pred_fc.detach().cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        pred_fc_arr[start_idx:start_idx + pred.shape[0]] = pred_fc
        start_idx = start_idx + pred.shape[0]
    return pred_arr, pred_fc_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_fid(sample_act, origin_act):
    sample_m = np.mean(sample_act, axis=0)
    sample_s = np.cov(sample_act, rowvar=False)
    
    origin_m = np.mean(origin_act, axis=0)
    origin_s = np.cov(origin_act, rowvar=False)
    
    fid_value = calculate_frechet_distance(sample_m, sample_s, origin_m, origin_s)
    return fid_value


def calculate_inception_score(out_arr, split_size=5000):
    # calc inception score
    scores = []
    for i in range(0, len(out_arr), split_size):
        part = out_arr[i : i + split_size]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    inception_score = float(np.mean(scores))
    return inception_score


def calculate_prec_recall(sample, origin):
    from sklearn.metrics import precision_score, recall_score
    sample = torch.argmax(torch.tensor(sample), dim=1)
    origin = torch.argmax(torch.tensor(origin), dim=1)
    
    return (
        precision_score(origin, sample, average='macro'),
        recall_score(origin, sample, average='macro'),
    )


def main():
    args = parser.parse_args()
    device = torch.device(args.device)
    
    # setup inception model #
    block_idx = InceptionForSignal.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionForSignal([block_idx], num_classes=args.num_classes).from_pretrained(PRE_TRAINED_PATH)
    model.to(device, dtype=float).eval()
    
    # setup data #
    sample_idx = [i for i, p in enumerate(args.path) if 'sample' in p][0]
    origin_idx = 0 if sample_idx else 1
    
    sample_dataset = SignalPathDataset(args.path[sample_idx], transforms=TF.ToTensor())
    sample_loader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    origin_dataset = SignalPathDataset(args.path[origin_idx], transforms=TF.ToTensor())
    origin_loader = DataLoader(origin_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # calculate evaluation #
    sample_act, sample_fc = get_activations(sample_loader, model, args.num_classes, device)
    origin_act, origin_fc = get_activations(origin_loader, model, args.num_classes, device)
    
    fid_value = calculate_fid(sample_act, origin_act)
    inception_score = calculate_inception_score(sample_fc)
    precision, recall = calculate_prec_recall(sample_fc, origin_fc)
    
    # output #
    print('FID: ', fid_value)
    print('Inception Score: ', inception_score)
    print("Precision:", precision)
    print("Recall:", recall)


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-classes', type=int, default=4)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dims', type=int, default=2048, choices=list(InceptionForSignal.BLOCK_INDEX_BY_DIM))
parser.add_argument('--save-stats', action='store_true')
parser.add_argument('path', type=str, nargs=2, help=('Paths to the generated images or to .npz statistic files'))

if __name__ == '__main__':
    main()