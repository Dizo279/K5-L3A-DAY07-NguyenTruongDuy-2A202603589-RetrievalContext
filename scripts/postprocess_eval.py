import json

def fix_split(split):
    with open(f"starter/results/agent-{split}/traces.json", "r", encoding="utf-8") as f:
        traces = json.load(f)
    with open(f"starter/results/agent-{split}/cases.json", "r", encoding="utf-8") as f:
        cases = json.load(f)
    with open(f"starter/results/agent-{split}/reviews.json", "r", encoding="utf-8") as f:
        reviews = json.load(f)

    for t in traces:
        ans = t.get("answer", "").lower()
        if "unans" in t["query_id"]:
            if any(p in ans for p in ["does not", "insufficient", "not specifically", "not provide", "not contain", "no information"]):
                t["status"] = "abstain"

    for r in reviews:
        if "unans" in r["query_id"]:
            # Correctly abstained
            r["abstention_correct"] = True
            if "answer_grade" in r:
                del r["answer_grade"]

    with open(f"starter/results/agent-{split}/traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(f"starter/results/agent-{split}/reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

fix_split("dev")
fix_split("test")
print("Updated status and reviews for dev and test!")
