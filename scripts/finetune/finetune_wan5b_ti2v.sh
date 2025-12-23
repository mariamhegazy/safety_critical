export WANDB_DISABLED=true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online

pip3 install moviepy
mkdir videos
pip3 install huggingface_hub==0.24.0 
pip3 install tf-keras==2.19.0
pip3 install trl==0.16.0
pip3 install transformers==4.46.1
pip3 install protobuf==5.29.5

###Actually, we don't use the original pytorch torchrun in our internal environment, 
###so I just follow the official example of pytorch.
###Please adapt the torchrun scripts into your own environment
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 --master_addr=192.168.0.101 --master_port=29500 \
    fastvideo/train_grpo_skyreels_i2v.py \
    --seed 42 \
    --model_type "hunyuan_hf" \
    --pretrained_model_name_or_path data/SkyReels-I2V \
    --reference_model_path data/flux \
    --vae_model_path data/SkyReels-I2V \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 121 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 40 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo \
    --tracker_project_name skyreels_i2v \
    --h 400 \
    --w 640 \
    --t 53 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --fps 8 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --cfg_infer 5.0 \
    --shift 7 \
    --use_group \
    --timestep_fraction 0.6 \
    --use_videoalign \
    --init_same_noise 





torchrun --nnodes=1 --nproc_per_node=4 \
    fastvideo/train_grpo_wan5b_ti2v.py \
    --seed 42 \
    --model_type "wan2.2_ti2v" \
    --pretrained_model_name_or_path data/wan2.2-ti2v-5b-diffusers \
    --reference_model_path data/flux \
    --vae_model_path data/wan2.2-ti2v-5b-diffusers \
    --cache_dir data/.cache \
    --data_json_path data/rl_embeddings_wan22/videos2caption.json \
    --gradient_checkpointing \
    --train_batch_size 1 \
    --sp_size 1 \
    --train_sp_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 121 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --checkpointing_steps 40 \
    --validation_steps 100000000 \
    --checkpoints_total_limit 3 \
    --allow_tf32 \
    --cfg 0.0 \
    --output_dir data/outputs/grpo_wan5b \
    --tracker_project_name wan2.2_5B \
    --h 416 \
    --w 640 \
    --t 30 \
    --sampling_steps 16 \
    --eta 0.3 \
    --lr_warmup_steps 0 \
    --fps 24 \
    --sampler_seed 1237 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 8 \
    --cfg_infer 5.0 \
    --shift 3 \
    --use_group \
    --timestep_fraction 0.6 \
    --use_videoalign \
    --init_same_noise \



# export NCCL_IB_DISABLE=1
# export NCCL_NET=Socket
# export NCCL_SOCKET_IFNAME=eth0   # or whatever your host NIC is
# export NCCL_DEBUG=WARN
