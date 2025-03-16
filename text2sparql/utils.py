import json
import random

INSTRUCTION = """Translate the following question with the given set of relevant entities and predicates into a SPARQL query."""

def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def format_gold_qid(id_map):
    if not id_map:
        return " - None\n"
    return "".join(f"{pid} - {label.get('en', 'N/A')}\n" for pid, label in id_map.items() if label)


def format_qid(id_map):
    if not id_map:
        return " - None\n"
    return "".join(f"{qid} - {label}\n" for qid, label in id_map.items() if label)

def format_refined_entities(refined_entities):
    return {
        entity['id']: {
            'en': entity['label']
        }
        for entity in refined_entities
    }

def create_prompt(question, entities_string, predicates_string):
    return f"Question: {question}\n\nEntities:\n{entities_string}\n\nRelations:\n{predicates_string}\n"


def add_extra_entities(entities, top5, n=2):
    entity_data = {}
    for entity_id, id_map in entities.items():
        label = id_map.get('en', None) if id_map is not None else None
        candidates = list(top5.get(entity_id, {}).items())
        n = min(n, len(candidates))
        aug_entities = dict(random.sample(candidates, n))
        if entity_id not in aug_entities:
            aug_entities[entity_id] = label
        entity_data.update(aug_entities)
    return entity_data

def is_schema_token(token, prefixes):
    # Check if token starts with any of the given prefixes.
    return any(token.startswith(prefix) for prefix in prefixes)

def preprocess_sparql(sparql: str) -> str:
    # Preprocess a SPARQL query to help tokenization and enforce a standard format.
    sparql = sparql.replace('SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }', '')

    sparql = sparql.replace('\n', ' ')
    sparql = sparql.replace('{', ' { ')
    sparql = sparql.replace('}', ' } ')
    sparql = sparql.replace('(', ' ( ')
    sparql = sparql.replace(')', ' ) ')
    sparql = sparql.replace('[', ' [ ')
    sparql = sparql.replace(']', ' ] ')
    sparql = sparql.replace(',', ' , ')
    sparql = sparql.replace('.', ' . ')
    sparql = sparql.replace('|', ' | ')
    sparql = sparql.replace('/', ' / ')
    sparql = sparql.replace(';', ' ; ')

    sparql = sparql.strip()
    sparql_tokens = sparql.split()
    updated_tokens = []
    # Lowercase non-schema tokens.
    for token in sparql_tokens:
        token = token.strip()
        if not is_schema_token(token, ['dr:', 'wd:', 'wdt:', 'p:', 'pq:', 'ps:', 'psn:']):
            updated_tokens.append(token.lower())
        else:
            updated_tokens.append(token)

    updated_sparql = " ".join(updated_tokens).strip()
    updated_sparql = updated_sparql.replace('. }', ' }')
    return updated_sparql

