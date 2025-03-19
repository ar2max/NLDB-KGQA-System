#!/bin/bash

python finetune_model.py \
  --input_file data/sft/rubq_train.json \
  --output_dir ./drive/MyDrive/text2sparql/models/rubq_model \
  --pretrained_model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --max_length 1024 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 8 \
  --warmup_steps 500 \
  --weight_decay 0.01 \
  --learning_rate 2e-5 \
  --logging_dir ./logs \
  --logging_steps 1000 \
  --evaluation_strategy no \
  --save_steps 1000 \
  --save_total_limit 1 \
  --dataloader_num_workers 4 \
  --gradient_accumulation_steps 1 \
  --optim adamw_torch \
  --seed 42 \
