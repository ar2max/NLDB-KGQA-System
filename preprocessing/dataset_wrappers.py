import re
import json
import asyncio
import aiohttp
import argparse
from tqdm import tqdm
from aiohttp import ClientSession
from sklearn.model_selection import train_test_split
from refined.inference.processor import Refined

from preprocessing_utils import (
    extract_wikidata_id_from_link,
    clean_sparql,
    map_wikidata_urls_to_prefix,
    fetch_with_semaphore,
    execute_wikidata_sparql_query,
    get_refined_entities
)

def load_data(file_path):
    """Load data from the given file path."""
    with open(file_path, 'r') as f:
        return json.load(f)

async def fetch_and_cache_labels(session, entities, cached_labels):
    """Fetch labels for a set of entities and update the cache."""
    new_labels = entities - cached_labels.keys()
    if new_labels:
        tasks = [fetch_with_semaphore(session, entity) for entity in new_labels]
        results = await asyncio.gather(*tasks)
        cached_labels.update({entity: result for entity, result in zip(new_labels, results) if result})

async def preprocess_rubq(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for item in tqdm(data):
                query = item.get('query')
                if not query:  # Skip if query is None
                    continue

                # Extract and deduplicate entities
                question_entities = list(
                    set(extract_wikidata_id_from_link(uri) for uri in item.get('question_uris', [])))
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(question_entities + query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                question_relations = list(set(
                    match.group(1) for value in item.get('question_props', [])
                    if (match := re.search(r'(P\d+)', value))
                ))
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(question_relations + query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                answer_entities = [extract_wikidata_id_from_link(ans['value']) for ans in item.get('answers', [])]

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': item['uid'],
                    'en_question': item.get('question_eng'),
                    'ru_question': item.get('question_text'),
                    'query': clean_sparql(query),
                    'entities': {
                        'question': {entity: cached_labels.get(entity) for entity in question_entities},
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': {relation: cached_labels.get(relation) for relation in question_relations},
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'refined': [ent['id'] for ent in get_refined_entities(item.get('question_eng'), refined=REFINED_MODEL)],
                    'answer_en': [cached_labels[entity].get('en') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels[entity].get('ru') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")

async def preprocess_qald(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)['questions']

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for item in tqdm(data):
                query = item.get('query').get('sparql')
                accepted_languages = [q['language'] for q in item['question']]
                if not query or not ('en' in accepted_languages and 'ru' in accepted_languages):
                    continue

                # clean query from prefixes and urls
                query = map_wikidata_urls_to_prefix(query)

                # Extract and deduplicate entities
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                # answer_entities = list(*(extract_answers_from_response(response) for response in item['answers']))
                answer_entities = execute_wikidata_sparql_query(query)
                if not answer_entities:
                    continue

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)
                en_question = next(filter(lambda q: q['language'] == 'en', item['question']))['string']
                # Prepare sample
                sample = {
                    'id': item['id'],
                    'en_question': en_question,
                    'ru_question': next(filter(lambda q: q['language'] == 'ru', item['question']))['string'],
                    'query': clean_sparql(query),
                    'entities': {
                        'question': None,
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': None,
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'refined': [ent['id'] for ent in get_refined_entities(en_question, refined=REFINED_MODEL)],
                    'answer_en': [cached_labels[entity].get('en') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels[entity].get('ru') for entity in answer_entities if
                                  cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")



async def preprocess_lcquad(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for i, item in enumerate(tqdm(data)):
                query = item.get('query')
                if not query:
                    continue

                # Extract and deduplicate entities
                query_entities = list(set(re.findall(r'(Q\d+)', query)))
                entities_to_fetch = set(query_entities)
                all_entities.update(entities_to_fetch)

                # Extract and deduplicate relations
                query_relations = list(set(re.findall(r'(P\d+)', query)))
                relations_to_fetch = set(query_relations)
                all_relations.update(relations_to_fetch)

                # Fetch and cache labels for entities and relations
                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                # Extract answers
                answer_entities = execute_wikidata_sparql_query(query)
                if not answer_entities:
                    continue

                # Fetch and cache labels for answer entities
                await fetch_and_cache_labels(session, set(answer_entities), cached_labels)

                # Prepare sample
                sample = {
                    'id': i,
                    'en_question': item.get('en_question'),
                    'ru_question': item.get('ru_question'),
                    'query': clean_sparql(query),
                    'entities': {
                        'question': None,
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': None,
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'refined': [ent['id'] for ent in get_refined_entities(item.get('en_question'), refined=REFINED_MODEL)],
                    'answer_en': [cached_labels.get(entity).get('en') for entity in answer_entities if cached_labels.get(entity, {}).get('en')],
                    'answer_ru': [cached_labels.get(entity).get('ru') for entity in answer_entities if cached_labels.get(entity, {}).get('ru')],
                    'answer_entities': answer_entities
                }
                dataset.append(sample)

    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:
        # Ensure data is saved even if an error occurs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': dataset,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {output_path}")

async def preprocess_pat(single_hop_path, multi_hop_path, train_output_path, test_output_path):
    dataset = []
    all_entities = set()
    all_relations = set()
    cached_labels = {}

    pat_singlehop = json.load(open(single_hop_path))
    pat_multihop = json.load(open(multi_hop_path))

    pat_data = pat_singlehop.copy()
    pat_data.update(pat_multihop)
    all_ids = set([10000])

    try:
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            for question, item in tqdm(list(pat_data.items())):
                sparql = item.get('sparql') or item.get('query')
                if not sparql:
                    continue

                id = item['uniq_id'] if item['uniq_id'] not in all_ids else max(all_ids) + 1

                multi_hop_sparql = None
                intermediate_entities = None
                if type(sparql) is list:
                    assert len(item['sparql']) == 2
                    sparql = item['sparql'][0]
                    multi_hop_sparql = item['sparql'][1]
                    intermediate_entities = [inter['ID'] for inter in item['intermediate entities']]

                question_entities = [item['subject']['subject']]
                query_entities = list(set(re.findall(r'(Q\d+)', sparql)))
                entities_to_fetch = set(question_entities + query_entities)
                all_entities.update(entities_to_fetch)

                question_relations = item['relations']
                query_relations = list(set(re.findall(r'(P\d+)', sparql)))
                relations_to_fetch = set(question_relations + query_relations)
                all_relations.update(relations_to_fetch)

                await fetch_and_cache_labels(session, entities_to_fetch.union(relations_to_fetch), cached_labels)

                sample = {
                    'id': id,
                    'en_question': item['question'],
                    'ru_question': None,
                    'query': sparql,
                    'multi_hop_query': multi_hop_sparql,
                    'intermediate_entities': intermediate_entities,
                    'entities': {
                        'question': {entity: cached_labels.get(entity) for entity in question_entities},
                        'query': {entity: cached_labels.get(entity) for entity in query_entities}
                    },
                    'relations': {
                        'question': {relation: cached_labels.get(relation) for relation in question_relations},
                        'query': {relation: cached_labels.get(relation) for relation in query_relations}
                    },
                    'refined': [ent['id'] for ent in get_refined_entities(item['question'], refined=REFINED_MODEL)],
                    'answer_en': None,
                    'answer_ru': None,
                    'answer_entities': None
                }
                dataset.append(sample)
                all_ids.add(id)
    except Exception as e:
        print(f"Error processing item: {item}")
        raise e

    finally:

        pat_train, pat_test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=42)

        # Ensure data is saved even if an error occurs
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': pat_train,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Train data saved to {train_output_path}")

        with open(test_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'dataset': pat_test,
                'entities': list(all_entities),
                'relations': list(all_relations)
            }, f, ensure_ascii=False, indent=4)
        print(f"Test data saved to {train_output_path}")

if __name__ == "__main__":
    REFINED_MODEL = Refined.from_pretrained(model_name='wikipedia_model', entity_set="wikidata")

    parser = argparse.ArgumentParser(description="Preprocess datasets for KGQA.")
    parser.add_argument("--input_rubq_train", required=True, help="Path to RuBQ train dataset")
    parser.add_argument("--input_rubq_test", required=True, help="Path to RuBQ test dataset")
    parser.add_argument("--output_rubq_train", required=True, help="Path to save preprocessed RuBQ train dataset")
    parser.add_argument("--output_rubq_test", required=True, help="Path to save preprocessed RuBQ test dataset")

    parser.add_argument("--input_qald_train", required=True, help="Path to QALD train dataset")
    parser.add_argument("--input_qald_test", required=True, help="Path to QALD test dataset")
    parser.add_argument("--output_qald_train", required=True, help="Path to save preprocessed QALD train dataset")
    parser.add_argument("--output_qald_test", required=True, help="Path to save preprocessed QALD test dataset")

    parser.add_argument("--input_lcquad_train", required=True, help="Path to LCQUAD_2.0 train dataset")
    parser.add_argument("--input_lcquad_test", required=True, help="Path to LCQUAD_2.0 test dataset")
    parser.add_argument("--output_lcquad_train", required=True, help="Path to save preprocessed LCQUAD_2.0 train dataset")
    parser.add_argument("--output_lcquad_test", required=True, help="Path to save preprocessed LCQUAD_2.0 test dataset")

    parser.add_argument("--input_pat_singlehop", required=True, help="Path to PAT singlehop dataset")
    parser.add_argument("--input_pat_multihop", required=True, help="Path to PAT multihop dataset")
    parser.add_argument("--output_pat_train", required=True, help="Path to save preprocessed PAT train dataset")
    parser.add_argument("--output_pat_test", required=True, help="Path to save preprocessed PAT test dataset")

    args = parser.parse_args()

    asyncio.run(preprocess_rubq(args.input_rubq_train, args.output_rubq_train))
    asyncio.run(preprocess_rubq(args.input_rubq_test, args.output_rubq_test))

    asyncio.run(preprocess_qald(args.input_qald_train, args.output_qald_train))
    asyncio.run(preprocess_qald(args.input_qald_test, args.output_qald_test))

    asyncio.run(preprocess_lcquad(args.input_lcquad_train, args.output_lcquad_train))
    asyncio.run(preprocess_lcquad(args.input_lcquad_test, args.output_lcquad_test))

    asyncio.run(preprocess_pat(args.input_pat_singlehop, args.input_pat_multihop, args.output_pat_train, args.output_pat_test))

