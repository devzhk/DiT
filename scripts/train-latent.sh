#! /bin/bash
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate
pip install git+https://github.com/huggingface/pytorch-image-models.git
pip install git+https://github.com/huggingface/diffusers
torchrun --nnodes=1 --nproc_per_node=8 train.py \ 
--config configs/imagenet-latent.yaml --data-path no --global-batch-size 256 --microbatch 32 \
--ckpt results/002-DiT-XL-2/checkpoints/1350000.pt
