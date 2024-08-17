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
from diffusion import create_diffusion, Dataset, worker_init_fn
from diffusion.diffusion_utils import bool_flag

import warnings
warnings.simplefilter('ignore', UserWarning)

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
    os.makedirs(args.results_dir, exist_ok=True)
    c_ = '_rep_' if args.rep_condition else ''
    experiment_dir = f"{args.results_dir}/{args.model}{c_}-{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    model = dits.__dict__[args.model](
        input_size=args.window_size,
        patch_size=args.patch_size,
        num_classes=0,
        represent=args.rep_condition,
    ).to(device, dtype=args.dtype)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    logger.info(f"optimizer: AdamW, lr: {args.lr}, weight decay: {args.wd}")

    # Setup data:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0., std=1., inplace=True)])
    dataset = Dataset(glob(args.data_paths), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} time windows ({args.data_paths})")

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
        for x in loader:
            x = x.to(device, dtype=args.dtype)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(x0=x) if args.rep_condition else None
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs=model_kwargs)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-paths", type=str)
    parser.add_argument("--results-dir", type=str, default='./experiment_log/pre-train')
    parser.add_argument("--model", type=str, default='XL')
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--rep-condition", type=bool_flag, default=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--dtype", type=torch.dtype, default=torch.float)
    parser.set_defaults(
        data_paths='../signal_data/experiment_data/30hz_10.0s_overlap0.0s/capture24/train/*.npy',
    )
    args = parser.parse_args()
    main(args)
