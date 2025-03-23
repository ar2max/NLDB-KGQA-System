import json
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
import random
from peft import PeftModel


def fix_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SparqlInferenceDataset(Dataset):
    def __init__(self, dataset_file: str, tokenizer, max_length: int = 1024):
        with open(dataset_file, "r", encoding="utf-8") as file:
            self.examples = json.load(file)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        sft_text = example["sft"]
        tokenized = self.tokenizer(
            sft_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt"
        )
        input_ids = tokenized['input_ids'][0].to(torch.long)
        attention_mask = tokenized['attention_mask'][0].to(torch.bool)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "id": example.get("id"),
            "question": example.get("question", ""),
            "gold_query": example.get("query", "")
        }


def predict_sparql_queries(model, tokenizer, dataset, batch_size=32):
    model.eval()
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Generating SPARQL Predictions"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                num_beams=2,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = outputs.cpu()
            prompt_length = input_ids.shape[1]
            decoded_outputs = tokenizer.batch_decode(
                outputs[:, prompt_length:],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

            for i, pred in enumerate(decoded_outputs):
                predictions.append({
                    "id": batch["id"][i],
                    "question": batch["question"][i],
                    "gold_query": batch["gold_query"][i],
                    "predicted_query": pred.strip()
                })
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPARQL query predictions for a given dataset."
    )
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path or model name for the fine-tuned model.")
    parser.add_argument("--dataset_file", type=str, required=True, help="Path to the JSON file containing the dataset.")
    parser.add_argument("--output_file", type=str, default="predictions_table.json", help="File to save the predictions table.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for generation.")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum input length for tokenization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_lora", type=bool, default=False)
    parser.add_argument("--base_model", type=str, default=None)

    args = parser.parse_args()

    fix_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"

    if args.use_lora:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16, device_map="auto")
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_name_or_path)

    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    
    model = model.to(device)"

    dataset = SparqlInferenceDataset(args.dataset_file, tokenizer, max_length=args.max_length)
    predictions = predict_sparql_queries(model, tokenizer, dataset, batch_size=args.batch_size)

    with open(args.output_file, "w", encoding="utf-8") as out_file:
        json.dump(predictions, out_file, indent=4)
    print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()