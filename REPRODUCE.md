# LatentMAS 主实验复现指南

> 基于论文 [arXiv:2511.20639](https://arxiv.org/abs/2511.20639)《Latent Collaboration in Multi-Agent Systems》

---

## 实验概览

论文主实验分为三张表格：

| 表格 | 架构 | 任务 | 模型 |
|---|---|---|---|
| **Table 1** | Sequential（顺序） | ARC-E、ARC-C、GSM8K、MedQA、MBPP+、HumanEval+ | Qwen3-4B / 8B / 14B |
| **Table 2** | Hierarchical（分层） | ARC-E、ARC-C、GSM8K、MedQA、MBPP+、HumanEval+ | Qwen3-4B / 8B / 14B |
| **Table 3** | Sequential + Hierarchical | AIME24、AIME25、GPQA-Diamond | Qwen3-8B / 14B |

每个表格中，每个任务均需跑 3 种方法：**Baseline（单模型）、TextMAS、LatentMAS**。

---

## 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `--temperature` | 0.6 | 生成温度 |
| `--top_p` | 0.95 | 核采样 |
| `--latent_steps` | 40 | 潜在推理步数（论文实验范围 0/10/20/40/80，40~80 最优） |
| `--latent_space_realign` | 开启 | 潜在空间对齐（论文建议作为超参数，按需开关） |
| `--seed` | 42 | 论文取 3 次独立运行均值，可依次使用 42、43、44 |
| ARC-E/ARC-C/GSM8K `--max_new_tokens` | 2048 | |
| MedQA/MBPP+/HumanEval+ `--max_new_tokens` | 4096 | |
| GPQA `--max_new_tokens` | 8192 | |
| AIME24/25 `--max_new_tokens` | 20000 | |

---

## 环境准备

```bash
conda create -n latentmas python=3.10 -y
conda activate latentmas
pip install -r requirements.txt

# 可选：安装 vLLM 以加速推理
pip install vllm

# 推荐设置 HuggingFace 缓存路径
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

---

## Step 0：修复代码 Bug（运行 8B 模型前必须执行）

> `run.py` 中 `--model_name` 的 choices 目前有 bug，漏掉了 `Qwen3-8B`。

```bash
sed -i 's/choices=\["Qwen\/Qwen3-4B", "Qwen\/Qwen3-4B", "Qwen\/Qwen3-14B"\]/choices=["Qwen\/Qwen3-4B", "Qwen\/Qwen3-8B", "Qwen\/Qwen3-14B"]/' run.py
```

---

## Table 1 — Sequential 架构（6 个通用任务）

### Baseline（单模型）

```bash
# ARC-Easy
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_easy \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --max_new_tokens 2048

# ARC-Challenge
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_challenge \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_challenge \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_challenge \
    --max_new_tokens 2048

# GSM8K
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gsm8k \
    --max_new_tokens 2048

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --max_new_tokens 2048

# MedQA
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task medqa \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task medqa \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task medqa \
    --max_new_tokens 4096

# MBPP+
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task mbppplus \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task mbppplus \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task mbppplus \
    --max_new_tokens 4096

# HumanEval+
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task humanevalplus \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task humanevalplus \
    --max_new_tokens 4096

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task humanevalplus \
    --max_new_tokens 4096
```

### TextMAS（Sequential）

```bash
# ARC-Easy
CUDA_VISIBLE_DEVICES=1 python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --max_new_tokens 2048 \
    2>&1 | tee results/text_mas_sequential_4B_arc_easy.log

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_easy \
    --prompt sequential \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --prompt sequential \
    --max_new_tokens 2048

# ARC-Challenge
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_challenge \
    --prompt sequential \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_challenge \
    --prompt sequential \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_challenge \
    --prompt sequential \
    --max_new_tokens 2048

# GSM8K
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --prompt sequential \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gsm8k \
    --prompt sequential \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt sequential \
    --max_new_tokens 2048

# MedQA
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task medqa \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task medqa \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task medqa \
    --prompt sequential \
    --max_new_tokens 4096

# MBPP+
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task mbppplus \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task mbppplus \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task mbppplus \
    --prompt sequential \
    --max_new_tokens 4096

# HumanEval+
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task humanevalplus \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task humanevalplus \
    --prompt sequential \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task humanevalplus \
    --prompt sequential \
    --max_new_tokens 4096
```

### LatentMAS（Sequential）

```bash
# ARC-Easy
CUDA_VISIBLE_DEVICES=2 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048 \
    2>&1 | tee results/latent_mas_sequential_4B_arc_easy_2.log

CUDA_VISIBLE_DEVICES=1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt sequential \
    --max_new_tokens 2048 \
    2>&1 | tee results/latent_mas_sequential_4B_arc_easy_3.log


python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_easy \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# ARC-Challenge
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_challenge \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_challenge \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_challenge \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# GSM8K
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gsm8k \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# MedQA
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task medqa \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task medqa \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task medqa \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

# MBPP+
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task mbppplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task mbppplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task mbppplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

# HumanEval+
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task humanevalplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task humanevalplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task humanevalplus \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096
```

---

## Table 2 — Hierarchical 架构（6 个通用任务）

> Baseline 与 Table 1 完全相同，无需重复运行。

### TextMAS（Hierarchical）

```bash
# ARC-Easy
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_easy \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --prompt hierarchical \
    --max_new_tokens 2048

# ARC-Challenge
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_challenge \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_challenge \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_challenge \
    --prompt hierarchical \
    --max_new_tokens 2048

# GSM8K
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gsm8k \
    --prompt hierarchical \
    --max_new_tokens 2048

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt hierarchical \
    --max_new_tokens 2048

# MedQA
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task medqa \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task medqa \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task medqa \
    --prompt hierarchical \
    --max_new_tokens 4096

# MBPP+
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task mbppplus \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task mbppplus \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task mbppplus \
    --prompt hierarchical \
    --max_new_tokens 4096

# HumanEval+
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task humanevalplus \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task humanevalplus \
    --prompt hierarchical \
    --max_new_tokens 4096

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task humanevalplus \
    --prompt hierarchical \
    --max_new_tokens 4096
```

### LatentMAS（Hierarchical）

```bash
# ARC-Easy
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_easy \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_easy \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_easy \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# ARC-Challenge
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task arc_challenge \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task arc_challenge \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task arc_challenge \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# GSM8K
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task gsm8k \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gsm8k \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048

# MedQA
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task medqa \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task medqa \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task medqa \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

# MBPP+
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task mbppplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task mbppplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task mbppplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

# HumanEval+
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-4B \
    --model_path /sharedspace/models/Qwen3-4B \
    --task humanevalplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task humanevalplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task humanevalplus \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 4096
```

---

## Table 3 — 推理密集型任务（仅 8B 和 14B）

### AIME 2024

```bash
# Baseline
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2024 \
    --max_new_tokens 20000

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2024 \
    --max_new_tokens 20000

# TextMAS Sequential
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2024 \
    --prompt sequential \
    --max_new_tokens 20000

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2024 \
    --prompt sequential \
    --max_new_tokens 20000

# LatentMAS Sequential
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2024 \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2024 \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

# TextMAS Hierarchical
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2024 \
    --prompt hierarchical \
    --max_new_tokens 20000

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2024 \
    --prompt hierarchical \
    --max_new_tokens 20000

# LatentMAS Hierarchical
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2024 \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2024 \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000
```

### AIME 2025

```bash
# Baseline
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2025 \
    --max_new_tokens 20000

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2025 \
    --max_new_tokens 20000

# TextMAS Sequential
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2025 \
    --prompt sequential \
    --max_new_tokens 20000

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2025 \
    --prompt sequential \
    --max_new_tokens 20000

# LatentMAS Sequential
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2025 \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2025 \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

# TextMAS Hierarchical
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2025 \
    --prompt hierarchical \
    --max_new_tokens 20000

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2025 \
    --prompt hierarchical \
    --max_new_tokens 20000

# LatentMAS Hierarchical
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task aime2025 \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task aime2025 \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 20000
```

### GPQA Diamond

```bash
# Baseline
python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gpqa \
    --max_new_tokens 8192

python run.py \
    --method baseline \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gpqa \
    --max_new_tokens 8192

# TextMAS Sequential
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gpqa \
    --prompt sequential \
    --max_new_tokens 8192

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gpqa \
    --prompt sequential \
    --max_new_tokens 8192

# LatentMAS Sequential
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gpqa \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 8192

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gpqa \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 8192

# TextMAS Hierarchical
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gpqa \
    --prompt hierarchical \
    --max_new_tokens 8192

python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gpqa \
    --prompt hierarchical \
    --max_new_tokens 8192

# LatentMAS Hierarchical
python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-8B \
    --model_path /sharedspace/models/Qwen3-8B \
    --task gpqa \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 8192

python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gpqa \
    --prompt hierarchical \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 8192
```

---

## 补充说明

### 多次运行取均值
论文报告的是 3 次独立运行的均值，在每条命令末尾加 `--seed 42 / 43 / 44` 分别运行，最后对 `accuracy` 取均值即可。

### latent_steps 消融
若需复现 Figure 8（latent steps 消融实验），在以下 5 个值分别跑 LatentMAS：
```bash
--latent_steps 0
--latent_steps 10
--latent_steps 20
--latent_steps 40
--latent_steps 80
```

### 开启 vLLM 加速（TextMAS / Baseline）
```bash
python run.py \
    --method text_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt sequential \
    --max_new_tokens 2048 \
    --use_vllm
```

### 开启 vLLM 加速（LatentMAS，需双 GPU）
```bash
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --method latent_mas \
    --model_name Qwen/Qwen3-14B \
    --model_path /sharedspace/models/Qwen3-14B \
    --task gsm8k \
    --prompt sequential \
    --latent_steps 40 --latent_space_realign \
    --max_new_tokens 2048 \
    --use_vllm --use_second_HF_model --enable_prefix_caching \
    --device cuda:0 --device2 cuda:1
```

### 硬件要求
- 论文使用 **8 × NVIDIA A100-80G**
- 单卡运行时，建议从 4B 模型起步；14B 模型至少需要 2 × 80G 或使用 vLLM tensor parallel
