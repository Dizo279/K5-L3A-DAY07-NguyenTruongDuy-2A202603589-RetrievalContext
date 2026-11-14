import json

with open('data/converted/chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Inspect chunk 0 and 1 of page 9:
c0 = chunks[0] # p9
c1 = chunks[1] # p9
print("c0 end:", c0['text'][-100:])
print("c1 start:", c1['text'][:100])
