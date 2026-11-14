import json
import re
import unicodedata

def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold()))

with open('data/converted/chunks_a.json', 'r', encoding='utf-8') as f:
    chunks_a = json.load(f)
with open('data/day6_chunks_b.json', 'r', encoding='utf-8') as f:
    chunks_b = json.load(f)
with open('data/converted/answerable_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

print("--- Testing Strategy A ---")
for q in questions:
    ev_norm = normalize(q['gold'][0]['evidence'])
    found = any(c['page'] == q['gold'][0]['page'] and ev_norm in normalize(c['text']) for c in chunks_a)
    print(f"{q['id']}: found in A = {found}")

print("--- Testing Strategy B ---")
for q in questions:
    ev_norm = normalize(q['gold'][0]['evidence'])
    found = any(c['page'] == q['gold'][0]['page'] and ev_norm in normalize(c['text']) for c in chunks_b)
    print(f"{q['id']}: found in B = {found}")
