import re
import ssl
import time
import asyncio
import aiohttp
from aiohttp import ClientSession, ClientError
from SPARQLWrapper import SPARQLWrapper, JSON


# Utility Functions
def extract_wikidata_id_from_link(link):
    """
    Extract a Wikidata ID (e.g., Q12345) from a given link.
    """
    patterns = [
        r"https?://www\.wikidata\.org/entity/(Q\d+)",
        r"https?://www\.wikidata\.org/entity/(\d+)",
        r"https?://www\.wikidata\.org/wiki/(Q\d+)"
    ]
    for pattern in patterns:
        if match := re.search(pattern, link):
            return 'Q' + match.group(1) if pattern.endswith(r"/(\d+)") else match.group(1)
    return None


def extract_answers_from_response(response):
    """
    Extract answers from a SPARQL query response.
    """
    answers = []
    if 'results' in response:
        for binding in response['results']['bindings']:
            for key, sub_answer in binding.items():
                value = sub_answer.get('value')
                if re.match(r"^https?://www\.wikidata\.org/entity/Q\d+$", value):
                    answers.append(extract_wikidata_id_from_link(value))
                else:
                    answers.append(value)
    elif 'boolean' in response:
        answers.append(response['boolean'])
    return answers


async def fetch_wikidata_labels(session, entity_id, languages=['en', 'ru'], fallback_languages=['en-gb', 'en-ca', 'mul'], max_retries=3, timeout=30):
    entity_id = str(entity_id)

    # Ensure entity_id is valid (e.g., Q42 or P31)
    if not re.match(r'^[QP]\d+$', entity_id):
        return {lang: entity_id for lang in languages}

    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
    retries = 0

    while retries < max_retries:
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    entities = data.get('entities', {})

                    # Handle cases where the entity_id is missing or replaced
                    labels = entities.get(entity_id, {}).get('labels', {})
                    if not labels:
                        labels = entities[next(iter(entities))].get('labels', {})

                    # Build the result dictionary with fallback handling
                    result = {}
                    for lang in languages:
                        if lang in labels:
                            result[lang] = labels[lang]['value']
                        elif lang == 'en':
                            # Check for fallback languages
                            for fallback in fallback_languages:
                                if fallback in labels:
                                    result[lang] = labels[fallback]['value']
                                    break
                            else:
                                # Fallback to "default for all languages" if available
                                if 'en' in labels:
                                    result[lang] = labels['en']['value']
                                else:
                                    result[lang] = f"[Label missing for {entity_id}]"

                    return result

                elif response.status == 429:
                    time.sleep(1)
                    retries += 1
                else:
                    print('Response:', response.status)
                    return {}

        except (ClientError, asyncio.TimeoutError) as e:
            retries += 1
            time.sleep(1)
            print(f"Retrying {entity_id} ({retries}/{max_retries}) due to {type(e).__name__}...")
    print(f"Failed to fetch labels for {entity_id} after {max_retries} attempts.")
    return {}




def execute_wikidata_sparql_query(query, endpoint="https://query.wikidata.org/sparql", timeout=30, max_retries=3, retry_delay=1):
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(timeout)
    ssl._create_default_https_context = ssl._create_unverified_context

    for attempt in range(1, max_retries + 1):
        try:
            # Execute query and parse results
            results = sparql.query().convert()
            return extract_answers_from_response(results)
        except Exception as e:
            # print(f"Error executing query (Attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                return []


async def fetch_with_semaphore(session, entity_id, languages=['en', 'ru'], semaphore=asyncio.Semaphore(30)):
    """
    Fetch labels with concurrency control using a semaphore.
    """
    async with semaphore:
        return await fetch_wikidata_labels(session, entity_id, languages)


def clean_sparql(sparql):
    """
    Clean and tokenize a SPARQL query for easier processing.
    """
    replacements = {
        '\n': ' ', '{': ' { ', '}': ' } ',
        '(': ' ( ', ')': ' ) ', '[': ' [ ', ']': ' ] ',
        ',': ' , ', '.': ' . ', '|': ' | ', '/': ' / ', ';': ' ; '
    }
    for old, new in replacements.items():
        sparql = sparql.replace(old, new)

    sparql = sparql.strip()
    tokens = sparql.split()
    updated_tokens = [
        token.lower() if not token.startswith(('dr:', 'wd:', 'wdt:', 'p:', 'pq:', 'ps:', 'psn:')) else token
        for token in tokens
    ]

    cleaned_query = " ".join(updated_tokens).replace('. }', ' }').strip()
    return cleaned_query


def map_wikidata_urls_to_prefix(sparql_query):
    """
    Replace Wikidata URLs in a SPARQL query with appropriate prefixes.
    """
    prefix_pattern = r"PREFIX\s+(\w+):\s+<([^>]+)>"
    prefixes = dict(re.findall(prefix_pattern, sparql_query))

    known_prefixes = {
        "http://www.wikidata.org/entity/": "wd:",
        "http://www.wikidata.org/prop/direct/": "wdt:"
    }
    prefix_map = {**known_prefixes, **prefixes}

    def replace_url(match):
        full_url = match.group(1)
        for base_url, prefix in prefix_map.items():
            if full_url.startswith(base_url):
                return prefix + full_url[len(base_url):]
        raise ValueError(f"Unknown URL structure: {full_url}")

    sparql_query = re.sub(prefix_pattern, "", sparql_query)
    url_pattern = r"<(http://www\.wikidata\.org/(entity|prop/direct)/[A-Za-z0-9]+)>"
    return re.sub(url_pattern, replace_url, sparql_query).strip()

def get_refined_entities(text, refined):
    return [{
        'id': span.__dict__['predicted_entity'].wikidata_entity_id,
        'label': span.__dict__['predicted_entity'].wikipedia_entity_title,
        'text': span.__dict__['text'],
        'span': tuple([span.__dict__['start'], span.__dict__['ln']]),
        'score': span.__dict__['entity_linking_model_confidence_score']
    } for span in refined.process_text(text) if span.predicted_entity is not None]

# Main Function with Tests
def main():
    """Run tests for the functions."""
    # Test extract_wikidata_id_from_link
    assert extract_wikidata_id_from_link("http://www.wikidata.org/entity/Q8070") == "Q8070"
    assert extract_wikidata_id_from_link("https://www.wikidata.org/entity/12345") == "Q12345"
    assert extract_wikidata_id_from_link("invalid_url") is None

    # Test extract_answers_from_response
    response_1 = {'head': {'vars': ['o1']}, 'results': {'bindings': [{'o1': {'type': 'uri', 'value': 'http://www.wikidata.org/entity/Q102844'}}]}}
    response_2 = {'head': {'vars': ['o1']}, 'results': {'bindings': [{'o1': {'xml:lang': 'de', 'type': 'literal', 'value': 'Angela Dorothea Kasner'}}]}}

    assert extract_answers_from_response(response_1) == ["Q102844"]
    assert extract_answers_from_response(response_2) == ["Angela Dorothea Kasner"]

    # Test map_wikidata_urls_to_prefix
    sparql_with_urls = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    SELECT ?item WHERE { <http://www.wikidata.org/entity/Q42> wdt:P31 <http://www.wikidata.org/entity/Q5> . }
    """
    mapped = map_wikidata_urls_to_prefix(sparql_with_urls)
    assert "wd:Q42" in mapped and "wd:Q5" in mapped

    # Test fetch_wikidata_labels
    async def test_async_functions():
        async with ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            labels = await fetch_wikidata_labels(session, "Q38458113")
            print(labels)
            assert "en" in labels and "Douglas Adams" in labels.values()

    asyncio.run(test_async_functions())

    print("All tests passed!")


if __name__ == "__main__":
    main()
