from functools import reduce
import math
import re


def flatten(array):
    result = []

    def walk(subarray):
        for item in subarray:
            if isinstance(item, list):
                walk(item)
            else:
                result.append(item)
    walk(array)

    return result


def merge(dict1, dict2):
    keys = dict1.keys() | dict2.keys()

    merged_keys = {}

    for key in keys:
        if key in dict1 and key in dict2:
            value1 = dict1[key]
            value2 = dict2[key]
            merged_keys[key] = flatten([value1, value2])
        elif key in dict1:
            value = dict1[key]
            merged_keys[key] = value if isinstance(value, list) else [value]
        else:
            value = dict2[key]
            merged_keys[key] = value if isinstance(value, list) else [value]

    return merged_keys


def normalize_token(word):
    matched = re.findall(r'\w+', word)
    if matched:
        return ''.join(matched).lower()
    return None


def build_inverted_index(terms):
    inverted_index = {}

    for term in terms:
        value = inverted_index.get(term, 0)
        inverted_index[term] = value + 1

    return inverted_index


def calculate_IDF(docs_count, term_count):
    return math.log2(1 + (docs_count - term_count + 1) / (term_count + 0.5))


def search(documents, search_words):
    docs_count = len(documents)

    docs_terms = {}

    for document in documents:
        id = document['id']
        text = document['text']
        lines = text.split('\n')

        terms = [
            normalize_token(word)
            for line in lines
            for word in line.split()
            if word is not None
        ]

        docs_terms[id] = terms

    inverted_indexes = []

    for document in documents:
        doc_terms = docs_terms[document['id']]
        doc_inverted_index = build_inverted_index(doc_terms)

        terms = {}

        for term in doc_inverted_index:
            term_count = doc_inverted_index[term]
            term_frequency = term_count / len(doc_terms)

            terms.update({term: {
                'doc_id': document['id'],
                'term_frequency': term_frequency,
                'count': term_count}
            })
        inverted_indexes.append(terms)

    index = reduce(merge, inverted_indexes, {})

    for key in index.keys():
        term_docs = index[key]
        term_docs_count = len(term_docs)

        for doc in term_docs:
            term_frequency = doc['term_frequency']
            doc_idf = calculate_IDF(docs_count, term_docs_count)
            tf_idf = term_frequency * doc_idf
            doc['tf_idf'] = tf_idf

    def find(text):
        terms = [
            normalize_token(word) for word in text.split() if word is not None
        ]

        current_index = {
            term: index.get(term)
            for term
            in terms
        }

        group_by_doc_id = {}
        for group in flatten(current_index.values()):
            if group:
                id = group['doc_id']
                group_by_doc_id.setdefault(id, []).append(group)

        current_docs_id = group_by_doc_id.keys()

        weighted_docs = {}
        for doc_id in current_docs_id:
            values = group_by_doc_id[doc_id]
            sum_idf = sum(
                [value['tf_idf'] for value in flatten(values)]
            )
            weighted_docs[doc_id] = sum_idf

        return sorted(
            current_docs_id,
            key=lambda doc_id: weighted_docs[doc_id],
            reverse=True
        )

    return find(search_words)
