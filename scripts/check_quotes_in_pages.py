import json
import re
import unicodedata

def normalize(s):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', s).casefold()))

with open('d:/AI_ThucChien/Day6/K5-L3A-DAY06-2A202603589-DataIngestion/starter/data/processed/pages.txt', 'r', encoding='utf-8') as f:
    raw_pages_text = f.read()

# Parse pages
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

print("Checking questions in raw pages:")
for q in questions:
    p = q['gold_page']
    quote = q['evidence_quote']
    p_text = pages.get(p, '')
    n_quote = normalize(quote)
    n_ptext = normalize(p_text)
    in_page = n_quote in n_ptext
    print(f"{q['id']}: page {p} in_page={in_page}")
    if not in_page:
        # Search which page actually contains it
        found_in = [p_idx for p_idx, text in pages.items() if n_quote in normalize(text)]
        print(f"   -> Found in pages: {found_in}")
