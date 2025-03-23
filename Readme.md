# The Benefits of Query-Based KGQA Systems for Complex and Temporal Questions in the LLM Era

**Authors**  
Oleg Somov
Artem Alekseev
Mikhail Chaichuk
Miron Butko
Alexander Panchenko 
Elena Tutubalina

---

## Overview

This repository provides code and experiments for **query-based Knowledge Graph QA (KGQA)**, focusing on generating **executable SPARQL queries** rather than direct answers. Our approach addresses complex, multi-hop, and temporal question-answering on WikiData.

### Contact
For questions or feedback:  
[artem.alekseev@skoltech.ru](mailto:artem.alekseev@skoltech.ru)

---

## Datasets

We test on four datasets, each with different formats:

1. **QALD-10**: [https://github.com/KGQA/QALD-10](https://github.com/KGQA/QALD-10)  
2. **LC-QuAD 2.0**: [https://github.com/AskNowQA/LC-QuAD2.0](https://github.com/AskNowQA/LC-QuAD2.0)  
3. **RuBQ 2.0**: [https://github.com/vladislavneon/RuBQ](https://github.com/vladislavneon/RuBQ)  
4. **PAT**: [https://github.com/jannatmeem95/PAT-Questions](https://github.com/jannatmeem95/PAT-Questions)

> **Attention**  
> **Please download and place these raw datasets in your preferred directory (e.g., `data/raw/<dataset_name>`). Make sure to update paths inside scripts accordingly.**

---

## Installation & Requirements

1. **Clone this Repository**  
   ```bash
   git clone https://github.com/ar2max/NLDB-KGQA-System.git
   cd NLDB-KGQA-System
   ```

2. **Install Dependencies**  
   - Python 3.8+ recommended  
   - Install packages from `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - For entity linking, we also use [ReFinED](https://github.com/amazon-science/ReFinED).  
   - For LLM-based tasks: `transformers`, `torch`, `trl==0.12.0`, `openai`, etc.  
   - If you plan to use GPT-4 or DeepSeek-R1, ensure you have valid API credentials or local checkpoints.

---

## Step-by-Step Usage

Below is a concise guide. All commands refer to `.sh` scripts that call the relevant Python tools.

### 1. **Preprocess the Datasets**

Unify each dataset (QALD-10, LC-QuAD 2.0, RuBQ 2.0, PAT) into a consistent JSON format (question, SPARQL, entities, relations).  
- **Script**: `preprocessing/run_preprocessing.sh`  
- **Usage**:
  ```bash
  bash preprocessing/run_preprocessing.sh
  ```
  Inside this script, you’ll see parameters like:
  ```bash
  --input_rubq_train "../data/raw/rubq/rubq_train.json" \
  --input_rubq_test  "../data/raw/rubq/rubq_test.json" \
  ...
  --output_rubq_train "../data/preprocessed/rubq/rubq_train.json" \
  --output_rubq_test  "../data/preprocessed/rubq/rubq_test.json"
  ```
  Adjust those paths to where your raw data and output folders are located.

### 2. **Build the BM25 Index**

Create a BM25 index over Wikidata subsets:
- **Script**: `retrieval/2_create_bm25_index.sh`
  ```bash
  bash retrieval/2_create_bm25_index.sh
  ```
  This script uses `pyserini` to index JSON files in `data/combined_data/`, then saves the Lucene index to `data/combined_data_index/`.

### 3. **Entity Disambiguation**

Use LLM-based reasoning (DeepSeek-R1, ChatGPT, etc.) on retrieved candidates:
- **Notebooks**:  
  - `disambiguation/vllm.ipynb` (local LLM inference)  
  - `disambiguation/openai_api.ipynb` or `openai_batchapi.ipynb` (OpenAI GPT-4)  
- Update paths within the notebook for:
  - BM25 retrieval results  
  - Preprocessed datasets  
- Run all cells to produce disambiguated entities.

### 4. **Model Fine-Tuning for Text-to-SPARQL**

We finetune a small LLM (e.g., [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct)) to map questions + entities + relations → SPARQL.

> **Example Input Prompt**  
> ```text
> Question: What can cause a tsunami?
>
> Entities:
> has cause - P828
>
> Relations:
> tsunami - Q8070
> ```

1. **Prepare Training Data**  
   - **Script**: `text2sparql/prepare_train_dataset.sh`  
   - Run via:
     ```bash
     bash text2sparql/prepare_train_dataset.sh
     ```
   - It combines preprocessed questions with gold/predicted entities & relations.

2. **Prepare Validation Data**  
   - **Script**: `text2sparql/prepare_validation_dataset.sh`  
   - Run via:
     ```bash
     bash text2sparql/prepare_validation_dataset.sh
     ```

3. **Fine-Tune the Model**  
   - **Script**: `text2sparql/run_finetune.sh`  
   - **Usage**:
     ```bash
     bash text2sparql/run_finetune.sh
     ```
   - Parameters include:
     ```bash
     --input_file data/sft/rubq_train.json \
     --output_dir ./text2sparql/models/rubq_model \
     --pretrained_model Qwen/Qwen2.5-Coder-0.5B-Instruct \
     --num_train_epochs 3 \
     --per_device_train_batch_size 8 \
     ...
     ```

### 5. **Inference**

Generate SPARQL queries for test sets:
- **Script**: `text2sparql/run_inference.sh`
  ```bash
  bash text2sparql/run_inference.sh
  ```
  Key parameters:
  ```bash
  --model_name_or_path ./text2sparql/models/rubq_model \
  --dataset_file data/e2e_validation_datasets/rubq_input_dataset.json \
  --output_file rubq_e2e_predictions.json \
  --batch_size 32
  ```

### 6. **Evaluation & Metrics**

We execute gold vs. predicted SPARQL on Wikidata to measure performance.  
- **Notebook**: `metrics/evaluation_notebook.ipynb`  
- Compare outputs for precision, recall, F1, etc.

### GPT-4 Baseline

Notebooks under `gpt_kgqa/` demonstrate GPT-4:
- **Direct QA** (`gpt4_direct_qa.ipynb`)  
- **Text-to-SPARQL** (`gpt4_text2sparql.ipynb`)  

Requires an OpenAI API key.

---

## Data & Predictions

- **Preprocessed datasets**: Located in `data/preprocessed/` (after Step 1).
- **Predictions**: Will be stored in `data/predictions/` or a path of your choice.
- No model checkpoints are provided by default.

---

## Citation

```
@inproceedings{somov2023query,
  title={The benefits of query-based KGQA systems for complex and temporal questions in LLM era},
  author={Somov, Oleg and Alekseev, Artem and Chaichuk, Mikhail and Butko, Miron and Panchenko, Alexander and Tutubalina, Elena},
  year={2023},
  ...
}
```

For questions or further information, contact:  
[artem.alekseev@skoltech.ru](mailto:artem.alekseev@skoltech.ru)

---

