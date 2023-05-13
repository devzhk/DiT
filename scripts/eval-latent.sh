#! /bin/bash
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate
pip install git+https://github.com/huggingface/pytorch-image-models.git
pip install git+https://github.com/huggingface/diffusers
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt_dir results/002-DiT-XL-2/checkpoints  --id_min 100000 --id_max 1300000 --id_step 100000