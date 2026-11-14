import json
import re

def _as_list(val):
    if val is None:
        return []
    return val if isinstance(val, list) else [val]

def _normalize(val):
    return re.sub(r"\s+", " ", str(val)).strip().casefold()

with open('data/day6_chunks_b.json', 'r', encoding='utf-8') as f:
    chunks_b = json.load(f)
with open('data/converted/chunks_a.json', 'r', encoding='utf-8') as f:
    chunks_a = json.load(f)
with open('data/converted/answerable_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

print("Checking Day 6 normalization on B:")
for q in questions:
    quotes = _as_list(q['gold'][0]['evidence'])
    gold_pages = _as_list(q['gold'][0]['page'])
    matches = [c for c in chunks_b if c['page'] in gold_pages and any(_normalize(quote) in _normalize(c['text']) for quote in quotes)]
    print(f"{q['id']}: page={gold_pages[0]}, matches in B={len(matches)}")
