#!/usr/bin/env bash
set -e

EXAMPLES_DIR="C:\Users\hongw\OneDrive\문서\interior_lora\interior_lora"  # diffusers 실제 경로
V="boisversionsrai"

accelerate launch "$EXAMPLES_DIR/train_dreambooth_lora_sdxl.py" \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --instance_data_dir "./train_dataset/coastal" \
  --output_dir "./final-lora-weight/adalora_coastal" \
  --instance_prompt "a ${V} interior room" \
  --resolution 768 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 2 \
  --max_train_steps 5000 \
  --rank 16 \
  --mixed_precision fp16 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 1000 \
  --checkpoints_total_limit 3 \
  --learning_rate 1e-4 \
  --target_rank 8 \
  --use_adalora 