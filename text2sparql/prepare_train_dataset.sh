#!/bin/bash

python prepare_train_sft_dataset.py \
  --train_file "data/datasets/lcquad_2.0/lcquad_2.0_train.json" \
  --test_file "data/datasets/lcquad_2.0/lcquad_2.0_test.json" \
  --entities_file "data/wikidata_files/top5_entities_candidates.json" \
  --relations_file "data/wikidata_files/top5_relations_candidates.json" \
  --mode "e2e" \
  --output_dir "data/sft" \
  --dataset_name "lcquad_2.0" \
  --tokenizer_path "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
  --lang "en" \
  --aug_size 0.5