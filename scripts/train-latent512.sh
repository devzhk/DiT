#! /bin/bash
pip install einops lmdb omegaconf wandb tqdm pyyaml accelerate
pip install diffusers["torch"] transformers timm webdataset
torchrun --nnodes=1 --nproc_per_node=4 train.py --config configs/imagenet-latent512.yaml --data_path ~/data/imagenet-wds --image_size 512 --microbatch 8 --global_batch_size 64