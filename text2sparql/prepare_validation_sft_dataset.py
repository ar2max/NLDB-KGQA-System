import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer

from nutils import INSTRUCTION, load_json, format_qid, format_refined_entities, create_prompt
def prepare_e2e_dataset(dataset, tokenizer, entities_path, predicates_path, output_path, mode="miron"):
    if mode not in {"refined", "miron"}:
        raise ValueError("Mode must be either 'refined' or 'miron'")

    entity_candidates = load_json(entities_path)
    predicate_candidates = load_json(predicates_path)

    validation_dataset = []

    for sample in tqdm(dataset, desc="Formatting dataset"):
        qid = str(sample['id'])

        refined_entities = sample.get('refined', [])
        miron_entities = entity_candidates.get(qid, {})
        miron_predicates = predicate_candidates.get(qid, {})

        if not miron_entities or not miron_predicates:
            continue

        assert miron_entities.get("question_eng") == miron_predicates.get("question_eng"), "Mismatched questions"
        assert miron_entities.get("query") == miron_predicates.get("query"), "Mismatched queries"
        if miron_entities["query"] != sample['query'] or miron_entities["question_eng"] != sample['en_question']:
            continue


        predicates_string = format_qid(miron_predicates["candidates"])
        if mode == "refined":
            entity_data = format_refined_entities(refined_entities)
        else:
            entity_data = miron_entities["candidates"]

        entities_string = format_qid(entity_data)
        user_task = create_prompt(sample["en_question"], entities_string, predicates_string)

        chat = [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": user_task}
        ]

        formatted_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        validation_dataset.append({
            "id": qid,
            "question": sample["en_question"],
            "query": sample["query"],
            "entities": entity_data,
            "relations": miron_predicates["candidates"],
            "sft": formatted_prompt
        })

    print(len(validation_dataset))
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(validation_dataset, file, ensure_ascii=False, indent=4)

    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process entity and relation candidates into a structured dataset.")
    parser.add_argument("--dataset", required=True, help="Path to test dataset JSON file with refined entities")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Coder-0.5B-Instruct",
                        help="Path or model name for the tokenizer")
    parser.add_argument("--entities", required=True, help="Path to entity candidates JSON file")
    parser.add_argument("--predicates", required=True, help="Path to predicate candidates JSON file")
    parser.add_argument("--output", required=True, help="Output path for processed dataset JSON file")
    parser.add_argument("--mode", choices=["refined", "miron"], default="miron",
                        help="Processing mode: 'refined' or 'miron'")

    args = parser.parse_args()

    # Load required components
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = load_json(args.dataset)

    # Process dataset
    prepare_e2e_dataset(dataset, tokenizer, args.entities, args.predicates, args.output, args.mode)