CUDA_VISIBLE_DEVICES=0 python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --max_samples -1 \
    --max_new_tokens 2048 \
    2>&1 | tee results/baseline_4B_arc_easy.log

# 没有用vllm 
CUDA_VISIBLE_DEVICES=1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 2048 \
    2>&1 | tee results/latent_mas_sequential_4B_arc_easy.log

CUDA_VISIBLE_DEVICES=2 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --max_samples -1 \
    --latent_space_realign \
    --max_new_tokens 2048 \
    2>&1 | tee results/latent_mas_sequential_4B_arc_easy_latent_space_realign.log

# 用VLLM
CUDA_VISIBLE_DEVICES=3 python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --max_samples -1 \
    --use_vllm \
    --max_new_tokens 2048 \
    2>&1 | tee results/baseline_4B_arc_easy_vllm.log


CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /NAS/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 2048 \
    --use_vllm \
    --use_second_HF_model \
    --enable_prefix_caching \
    --device2 cuda:1 \
    2>&1 | tee results/latent_mas_sequential_4B_arc_easy_vllm.log   


CUDA_VISIBLE_DEVICES=2 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --prompt sequential \
    --max_samples -1 \
    --max_new_tokens 2048 \
    2>&1 | tee results/latent_mas_sequential_14B_arc_easy.log

CUDA_VISIBLE_DEVICES=7 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --max_samples -1 \
    --prompt sequential \
    --max_new_tokens 2048 \
    --latent_steps 20 \
    2>&1 | tee results/0310/latent_mas_sequential_4B_gsm8k_latent_steps_20.log

CUDA_VISIBLE_DEVICES=7 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --max_samples -1 \
    --prompt sequential \
    --max_new_tokens 2048 \
    2>&1 | tee results/0310/latent_mas_sequential_4B_gsm8k.log