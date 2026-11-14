import json
import re
import unicodedata

def normalize(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', s).casefold()))

with open('data/converted/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

with open('d:/AI_ThucChien/Day6/K5-L3A-DAY06-2A202603589-DataIngestion/starter/data/processed/pages.txt', 'r', encoding='utf-8') as f:
    raw_pages_text = f.read()

pages = {int(b.split('\n')[0].split(' ')[0]): b.split('\n', 1)[1] for b in raw_pages_text.split('=== physical_page_') if b.strip()}

with open('data/converted/answerable_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

for q in questions:
    qid = q['id']
    ev = q['gold'][0]['evidence']
    p = q['gold'][0]['page']
    
    # check match in chunks
    ev_norm = normalize(ev)
    in_chunk = any(c['page'] == p and ev_norm in normalize(c['text']) for c in chunks)
    print(f"{qid} (page {p}): in_chunk={in_chunk}")
    if not in_chunk:
        # Check why
        # find in page text
        p_text = pages.get(p, '')
        in_page = ev_norm in normalize(p_text)
        print(f"   -> in raw page {p}: {in_page}")
