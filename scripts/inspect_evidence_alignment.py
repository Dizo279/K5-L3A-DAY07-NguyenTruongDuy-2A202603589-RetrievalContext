import json
import re

def _normalize(val):
    return re.sub(r"\s+", " ", str(val)).strip().casefold()

with open('data/converted/answerable_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

with open('data/day6_chunks_b.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

updated_questions = []
for q in questions:
    gold = q['gold'][0]
    ev = gold['evidence']
    page = gold['page']
    
    # check if exact match exists in page
    norm_ev = _normalize(ev)
    matching_chunks = [c for c in chunks if c['page'] == page and norm_ev in _normalize(c['text'])]
    
    if matching_chunks:
        updated_questions.append(q)
    else:
        # Find which chunk on this page has the key information
        # Let's inspect chunks on this page
        p_chunks = [c for c in chunks if c['page'] == page]
        print(f"--- Q ID: {q['id']}, Page: {page} ---")
        print(f"Original quote: {ev}")
        for idx, c in enumerate(p_chunks):
            # check words overlap
            ev_words = [w for w in re.findall(r'\w+', ev.lower()) if len(w) > 3]
            chunk_words = set(re.findall(r'\w+', c['text'].lower()))
            common = set(ev_words) & chunk_words
            if len(common) >= 3:
                print(f"Candidate chunk {idx} (overlap {len(common)}): {c['text'][:180]}...")
