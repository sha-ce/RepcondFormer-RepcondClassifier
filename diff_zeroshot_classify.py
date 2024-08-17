import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import argparse
import sys
import os

from diffusion import diffusion_transformer as dits
from diffusion import create_diffusion, Dataset, NormalDataset, worker_init_fn
from diffusion.gaussian_diffusion import mean_flat

import warnings
warnings.simplefilter('ignore', FutureWarning)

class Logger:
    def __init__(self, filename, mode='w'):
        self.console = sys.stdout
        self.file = open(filename, mode)
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
    def flush(self):
        self.console.flush()
        self.file.flush()
    def close(self):
        self.file.close()

def get_error(out, target, n_classes, n_samples):
    pred = torch.cat(out, dim=0)
    error = mean_flat((target-pred)**2).detach().cpu()
    error = error.reshape(n_classes, n_samples).mean(dim=1)
    
    hard = torch.argmin(error).item()
    soft = 1-torch.nn.functional.softmax(error, dim=0)
    return hard, soft


def main(cfg):
    seed = cfg.seed if cfg.seed is not None else np.random.randint(-999999, 999999)
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    output_dir = cfg.ckpt.replace('checkpoints/', 'classify_ckpt_').replace('.pt', '')
    os.makedirs(output_dir, exist_ok=True)
    logger = Logger(os.path.join(output_dir, 'timestep-'+str(cfg.timestep)+'.log'), mode='a')
    logger.write(
        f'#############################################\n'+
        f'diffusion samplign step: {cfg.sampling_steps}\n'+
        f'num batch per class: {cfg.num_per_class}\n'+
        f'seed: {seed}\n'
    )
    
    # setup model
    checkpoint = torch.load(cfg.ckpt)
    ckpt_args = checkpoint["args"] if "args" in checkpoint else None
    ckpt_args_ = torch.load(ckpt_args.ckpt)['args']

    model = dits.__dict__[ckpt_args_.model](
        input_size=ckpt_args_.window_size,
        patch_size=ckpt_args_.patch_size,
        num_classes=ckpt_args.num_classes,
    ).to(device, cfg.dtype)
    model.load_state_dict(checkpoint['ema'], strict=False)
    model.eval()

    diffusion = create_diffusion(str(cfg.sampling_steps))

    # setup data
    x_test = np.load(os.path.join(cfg.data_path, cfg.dataset, 'x_test.npy'))
    y_test = np.load(os.path.join(cfg.data_path, cfg.dataset, 'y_test.npy'))
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    dataset = NormalDataset(x_test, y_test, name="test", transform=transform)
    
    # run
    B,C,L = x_test.shape
    n = cfg.num_per_class
    batch = n*ckpt_args.num_classes
    eps = torch.randn((n, C, L), device=device)
    noise = torch.cat([eps for _ in range(ckpt_args.num_classes)], dim=0)

    t = torch.tensor(cfg.timestep, device=device)
    logger.write(f'#### {t} ####\n')
    trues = []
    x0preds, x0errors = [], []
    # epspreds, epserrors = [], []
    for idx in tqdm(range(len(dataset))):
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device, dtype=cfg.dtype)
        trues.append(y)

        x_batch = torch.cat([x for _ in range(batch)], dim=0)
        t_batch = torch.tensor([t]*batch, device=device)
        y_batch = torch.cat([torch.tensor([i for _ in range(n)]) for i in range(ckpt_args.num_classes)], dim=0).to(device)
        
        x0s, epss = [], []
        for i in range(0, batch, cfg.batch_size):
            x_ = x_batch[i:i+cfg.batch_size]
            t_ = t_batch[i:i+cfg.batch_size]
            n_ = noise[i:i+cfg.batch_size]
            model_kwargs = dict(y=y_batch[i:i+cfg.batch_size])
        
            x_t = diffusion.q_sample(x_, t_, noise=n_)
            x_t = x_t.to(cfg.dtype)

            with torch.no_grad():
                out = diffusion.p_sample(model, x_t, t_, clip_denoised=False, model_kwargs=model_kwargs, dtype=cfg.dtype)
            x0s.append(out['pred_xstart'])
        
        x0pred, x0error = get_error(x0s, target=x_batch, n_classes=ckpt_args.num_classes, n_samples=n)
        x0preds.append(x0pred)
        x0errors.append(x0error)

    from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, top_k_accuracy_score
    trues = np.array(trues)
    
    preds = np.array(x0preds)
    errors = np.array(x0errors)
    logger.write('### x0 prediction error ###\n'+
        f'acc: {accuracy_score(trues, preds):.6f}\n'+
        f'rec: {recall_score(trues, preds, average='macro'):.6f}\n'+
        f'pre: {precision_score(trues, preds, average='macro'):.6f}\n'+
        f'f1 : {f1_score(trues, preds, average='macro'):.6f}\n'+
        f'{confusion_matrix(trues, preds)}\n'+
        f'top-2 acc: {top_k_accuracy_score(trues, errors, k=2):.6f}\n'+
        f'top-3 acc: {top_k_accuracy_score(trues, errors, k=3):.6f}\n'+
        f'top-4 acc: {top_k_accuracy_score(trues, errors, k=4):.6f}\n'+
        f'top-5 acc: {top_k_accuracy_score(trues, errors, k=5):.6f}\n'+
        f'#############################################\n\n'
    )
    logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--data-path", type=str, default='../signal_data/downstream')
    parser.add_argument("--seed", type=int, default=None)         # if seed is None, seed is random val generated from numpy.random.randint()
    parser.add_argument("--num-per-class", type=int, default=512) # The number of noise removal samples per class label.
    parser.add_argument("--batch-size", type=int, default=128)    # We set a batch size to prevent the GPU from overflowing due to memory constraints.
    parser.add_argument("--sampling-steps", default=1000)         # This is the number of time steps in the diffusion model.
    parser.add_argument("--timestep", default=200)                # The time steps to which noise removal is applied.
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float)
    parser.set_defaults(
        seed=0,
        ckpt='./experiment_log/downstream/<dataset>/diffusion/<fine-tuned model>',
    )
    args = parser.parse_args()
    main(args)