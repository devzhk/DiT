# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import DiT_models
from diffusion import create_diffusion

from omegaconf import OmegaConf
from train_utils.datasets import ImageNetLatentDataset


import webdataset as wds
import pickle
from itertools import islice

#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def sample(moments, scale_factor=0.18215):
    mean, logvar = torch.chunk(moments, 2, dim=1)
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    z = mean + std * torch.randn_like(mean)
    z = scale_factor * z
    return z


def onehot2int(label_vec):
    """
    Find one-hot vector index.
    Args:
        label_vec: A one-hot vector of shape (batch_size, num_classes).
    """
    return torch.argmax(label_vec, dim=1)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# WebDataset Helper Function
def nodesplitter(src, group=None):
    rank, world_size, worker, num_workers = wds.utils.pytorch_worker_info()
    if world_size > 1:
        for s in islice(src, rank, None, world_size):
            yield s
    else:
        for s in src:
            yield s


def get_file_paths(dir):
    return [os.path.join(dir, file) for file in os.listdir(dir)]



def decode_data(item):
    output = {}
    img = pickle.loads(item['latent'])
    output['latent'] = img
    label = int(item['cls'].decode('utf-8'))
    output['label'] = label
    return output


def make_loader(root, mode='train', batch_size=32, 
                num_workers=4, cache_dir=None, 
                resampled=False, world_size=1, total_num=1281167, 
                bufsize=1000, initial=100):
    data_list = get_file_paths(root)
    num_batches_in_total =  total_num // (batch_size * world_size)
    # paths = split_by_proc(data_list, rank, world_size)
    # print(f'rank: {rank}, world_size: {world_size}, len(paths): {len(paths)}')
    if resampled:
        repeat = True
        splitter = False
    else:
        repeat = False
        splitter = nodesplitter
    dataset = (
        wds.WebDataset(
        data_list, 
        cache_dir=cache_dir,
        repeat=repeat,
        resampled=resampled, 
        handler=wds.handlers.warn_and_stop, 
        nodesplitter=splitter,
        )
        .shuffle(bufsize, initial=initial)
        .map(decode_data, handler=wds.handlers.warn_and_stop)
        .to_tuple('latent label')
        .batched(batch_size, partial=False)
        )
    
    # mprint(f'dataset created from {paths}')
    loader = wds.WebLoader(dataset, batch_size=None, num_workers=num_workers, shuffle=False, persistent_workers=True)
    if resampled:
        loader = loader.with_epoch(num_batches_in_total)
    return loader

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={size}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    if args.ckpt is not None:  # Load from checkpoint if provided
        logger.info(f"Loading checkpoint from {args.ckpt}")
        train_steps = int(os.path.basename(args.ckpt).split(".")[0])  # Get the training step number from the ckpt name
        ckpt = torch.load(args.ckpt, map_location=torch.device('cuda'))
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        ema.load_state_dict(ckpt["ema"])
    else:
        train_steps = 0
        update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model = DDP(model, device_ids=[rank]) 
    # Setup data:

    loader = make_loader(args.data_path, batch_size=args.global_batch_size,
                         num_workers=args.num_workers, world_size=size)
    

    batch_size_per_gpu = int(args.global_batch_size // dist.get_world_size())
    num_grad_accum = batch_size_per_gpu // args.microbatch



    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = sample(x) 
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # model_kwargs = dict(y=y)
            opt.zero_grad()
            for i in range(num_grad_accum):
                x_ = x[i * args.microbatch: (i + 1) * args.microbatch]
                y_ = y[i * args.microbatch: (i + 1) * args.microbatch]
                t_ = t[i * args.microbatch: (i + 1) * args.microbatch]
                model_kwargs = dict(y=y_)
        
                loss_dict = diffusion.training_losses(model, x_, t_, model_kwargs)
                loss = loss_dict["loss"].sum().mul(1 / batch_size_per_gpu)
                loss.backward()
                running_loss += loss.item()
            opt.step()
            update_ema(ema, model.module)
            # Log loss values:
            
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                logger.info(f'Peak GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1024 ** 3:.2f} GB')
                logger.info(f'Reserved GPU memory: {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB')
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/imagenet-latent.yaml")

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global_batch_size", type=int, default=256)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    main(args)
