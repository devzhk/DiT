#! /bin/bash
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate transformers diffusers
pip install git+https://github.com/huggingface/pytorch-image-models.git
torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py --ckpt_dir results/002-DiT-XL-2/checkpoints  --id_min 100000 --id_max 1400000 --id_step 100000