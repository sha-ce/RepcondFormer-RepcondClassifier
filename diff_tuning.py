# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from datetime import datetime
from time import time
import argparse
import logging
import os

from diffusion import diffusion_transformer as dits
from diffusion import create_diffusion, Dataset, NormalDataset, worker_init_fn


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    seed = args.global_seed
    torch.manual_seed(seed)
    print(f"Starting seed={seed}, device={device}.")

    # Setup an experiment folder:
    args.results_dir = os.path.join(args.results_dir, args.dataset, 'diffusion')
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_dir = f"{args.results_dir}/{args.ckpt.split('/')[-3].split('-')[0]}-ckpt({args.ckpt.split('/')[-1][:-3]})-{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    checkpoint = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    a = checkpoint["args"] if "args" in checkpoint else None
    model = dits.__dict__[a.model](
        input_size=a.window_size,
        patch_size=a.patch_size,
        num_classes=args.num_classes,
        represent=False
    ).to(device, dtype=args.dtype)
    checkpoint['ema']['x_embedder.proj.weight'], _ = checkpoint['ema']['x_embedder.proj.weight'].chunk(2, dim=1)
    model.load_state_dict(checkpoint['ema'], strict=False)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Setup data:
    x_train = np.load(os.path.join(args.data_path, args.dataset, 'x_train.npy'))
    y_train = np.load(os.path.join(args.data_path, args.dataset, 'y_train.npy'))
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    train_dataset = NormalDataset(x_train, y_train, name="train", transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.global_batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in train_loader:
            x = x.to(device, dtype=args.dtype)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=dict(y=y.to(device)))
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device).item()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--data-path", type=str, default='../signal_data/downstream')
    parser.add_argument("--results-dir", type=str, default="experiment_log/downstream")
    parser.add_argument("--dataset", type=str, choices=['adl', 'oppo', 'pamap', 'realworld', 'wisdm'])
    parser.add_argument("--num-classes", type=int) # adl:5, oppo:4, pamap:8, realworld:8, wisdm:18
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float)
    parser.set_defaults(
        dataset='pamap',
        ckpt='./experiment_log/pre-train/<pre-trained model>',
    )
    args = parser.parse_args()
    args.num_classes = {'adl':5, 'oppo':4, 'pamap':8, 'realworld':8, 'wisdm':18}[args.dataset]
    main(args)
