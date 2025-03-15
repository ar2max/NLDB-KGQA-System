#!/bin/bash
python dataset_wrappers.py \
    --input_rubq_train "../data/raw/rubq/rubq_train.json" \
    --input_rubq_test "../data/raw/rubq/rubq_test.json" \
    --input_qald_train "../data/raw/qald/qald_train.json" \
    --input_qald_test "../data/raw/qald/qald_test.json" \
    --input_lcquad_train "../data/raw/lcquad/lcquad_2_train.json" \
    --input_lcquad_test "../data/raw/lcquad/lcquad_2_test.json" \
    --input_pat_singlehop "../data/raw/pat/PAT-multihop.json" \
    --input_pat_multihop "../data/raw/pat/PAT-singlehop.json" \
    --output_rubq_train "../data/preprocessed/rubq/rubq_train.json" \
    --output_rubq_test "../data/preprocessed/rubq/rubq_test.json" \
    --output_qald_train "../data/preprocessed/qald/qald_train.json" \
    --output_qald_test "../data/preprocessed/qald/qald_test.json" \
    --output_lcquad_train "../data/preprocessed/lcquad/lcquad_2_train.json" \
    --output_lcquad_test "../data/preprocessed/lcquad/lcquad_2_test.json" \
    --output_pat_train "../data/preprocessed/pat/pat_train.json" \
    --output_pat_test "../data/preprocessed/pat/pat_test.json"

