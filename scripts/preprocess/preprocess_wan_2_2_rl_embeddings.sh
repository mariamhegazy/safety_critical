GPU_NUM=8 # 2,4,8
MODEL_PATH="./data/Wan2.1-T2V-1.3B"
OUTPUT_DIR="data/rl_embeddings"

pip install diffusers==0.35.0 peft==0.17.0 transformers==4.56.0

# torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
#     fastvideo/data_preprocess/preprocess_wan_2_1_embeddings.py \
#     --model_path $MODEL_PATH \
#     --output_dir $OUTPUT_DIR \
#     --prompt_dir "./assets/prompts.txt"



# torchrun --nproc_per_node=4 --master_port 19002 \
#     fastvideo/data_preprocess/preprocess_wan_2_1_embeddings.py \
#     --model_path "./data/Wan2.1-T2V-1.3B" \
#     --output_dir "data/rl_embeddings" \
#     --prompt_dir "./assets/prompts_subset.txt"

torchrun --nproc_per_node=4 fastvideo/data_preprocess/preprocess_wan_2_2_embeddings.py \
  --prompt_dir ./assets/consist-id_subset.txt \
  --output_dir data/rl_embeddings_wan22 \
  --model_path data/wan2.2-ti2v-5b-diffusers \
  --train_batch_size 32 \
  --dataloader_num_workers 4

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

python fastvideo/data_preprocess/preprocess_wan_2_2_embeddings.py \
  --prompt_dir what_if_qwen3_30b_sc_wan_new_imgs_600.jsonl \
  --output_dir data/rl_embeddings_wan22 \
  --caption_key future_caption \
  --image_key file_name \
  --image_root /capstor/store/cscs/swissai/a144/datasets/lingo/scenary/