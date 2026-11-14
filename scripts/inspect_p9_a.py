import json
with open('data/converted/chunks_a.json', 'r', encoding='utf-8') as f:
    chunks_a = json.load(f)
p9 = [c for c in chunks_a if c['page'] == 9]
for idx, c in enumerate(p9):
    print(f"Chunk {idx} (len {len(c['text'])}): {c['text'][:60]} ... {c['text'][-60:]}")
