import json
import re

with open('d:/AI_ThucChien/Day6/K5-L3A-DAY06-2A202603589-DataIngestion/starter/data/processed/pages.txt', 'r', encoding='utf-8') as f:
    raw_pages_text = f.read()

page_blocks = raw_pages_text.split('=== physical_page_')
pages = {}
for block in page_blocks:
    if not block.strip():
        continue
    lines = block.split('\n', 1)
    p_num = int(lines[0].split(' ')[0])
    p_text = lines[1] if len(lines) > 1 else ''
    pages[p_num] = p_text

with open('d:/AI_ThucChien/Day6/K5-L3A-DAY06-2A202603589-DataIngestion/starter/benchmark/questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)

for qid in ['q06', 'q08', 'q17', 'q20']:
    q = next(item for item in questions if item['id'] == qid)
    print(f"=== {qid} ===")
    print("Quote:", q['evidence_quote'])
    words = [w for w in re.findall(r'\w+', q['evidence_quote'].lower()) if len(w) > 4]
    for p_num, text in pages.items():
        if all(w in text.lower() for w in words[:4]):
            print(f"Match in page {p_num}!")
            # print surrounding
            pos = text.lower().find(words[0])
            print("Snippet:", text[max(0, pos-50):pos+250])
