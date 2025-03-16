#!/bin/bash
python inference.py \
--model_name_or_path "./drive/MyDrive/text2sparql/models/rubq_model" \
--dataset_file "data/e2e_validation_datasets/rubq_input_dataset.json" \
--output_file "rubq_e2e_predictions.json" \
--batch_size 32 \
--max_length 1024