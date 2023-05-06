ngc batch run \
--name "ml-model.DiT-XL2-train" \
--commandline "cd /diffusion_ws/Code/DiT; git config --global --add safe.directory /diffusion_ws/Code/DiT; \
bash scripts/train-latent.sh" \
--image "nvidia/pytorch:23.03-py3" \
--ace nv-us-west-3 \
--instance dgxa100.80g.8.norm \
--priority LOW \
--total-runtime 168h \
--workspace ReY8v22xQoqdGPDyMwZ75g:/diffusion_ws \
--datasetid 1602645:/imagenet_256_latent_lmdb \
--result /results \
--label ml__vitdiff_hk --label _wl__computervision \
--port 6066 --port 1234 --port 8888