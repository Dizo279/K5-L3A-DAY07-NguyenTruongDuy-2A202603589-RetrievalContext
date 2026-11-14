import json
import time
from pathlib import Path
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever

def hit(chunk, gold_list):
    for g in gold_list:
        if chunk.get("source_id") == g.get("source_id") and chunk.get("page") == g.get("page"):
            c_text = ' '.join(chunk.get("text", "").split()).lower()
            e_text = ' '.join(g.get("evidence", "").split()).lower()
            if e_text in c_text:
                return True
            e_words = [w for w in e_text.split() if len(w) > 3]
            if e_words and sum(1 for w in e_words if w in c_text) / len(e_words) >= 0.8:
                return True
    return False

def evaluate_configuration(name, retriever_fn, questions, split="dev", budget=1800):
    # Only answerable questions in dev (15 questions)
    subset = [q for q in questions if q.get("split") == split and q.get("gold", [{}])[0].get("evidence") is not None]
    total = len(subset)
    hit_1 = 0
    hit_3 = 0
    context_evidence = 0
    traces = []

    start_time = time.time()
    for q in subset:
        query_text = q["question"]
        q_filter = q.get("filter", {})
        
        t0 = time.time()
        results = retriever_fn(query_text, filter_dict=q_filter)
        latency = time.time() - t0

        h1 = hit(results[0], q["gold"]) if results else False
        h3 = any(hit(r, q["gold"]) for r in results[:3])

        # build context up to budget
        ctx_text = ""
        ctx_chunks = []
        for r in results:
            item_text = f"[{r['id']}] {r['text']}\n"
            if len(ctx_text) + len(item_text) <= budget:
                ctx_text += item_text
                ctx_chunks.append(r)
            else:
                break
        ctx_ev = any(hit(r, q["gold"]) for r in ctx_chunks)

        if h1: hit_1 += 1
        if h3: hit_3 += 1
        if ctx_ev: context_evidence += 1

        traces.append({
            "question_id": q["id"],
            "question": q["question"],
            "split": split,
            "hit_at_1": h1,
            "hit_at_3": h3,
            "context_evidence": ctx_ev,
            "latency": latency,
            "results": [{"id": r["id"], "page": r["page"], "score": r["score"]} for r in results[:3]]
        })

    total_time = time.time() - start_time
    summary = {
        "configuration": name,
        "split": split,
        "total_questions": total,
        "hit_at_1": f"{hit_1}/{total} ({hit_1/total:.1%})",
        "hit_at_3": f"{hit_3}/{total} ({hit_3/total:.1%})",
        "context_evidence_rate": f"{context_evidence}/{total} ({context_evidence/total:.1%})",
        "total_time_seconds": round(total_time, 2),
        "avg_latency_seconds": round(total_time / total, 3)
    }
    return summary, traces

def main():
    with open("data/converted/chunks_a.json", "r", encoding="utf-8") as f:
        chunks_a = json.load(f)
    with open("data/converted/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    dense = DenseRetriever(strategy="A")
    bm25 = BM25Retriever(chunks=chunks_a)
    hybrid = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    experiments = [
        # Baseline
        ("Baseline (Dense top_k=3, budget=1000)", lambda q, filter_dict: dense.search(q, top_k=3, filter_dict=filter_dict)),
        
        # Improvement 1: Hybrid RRF top_k=5 with higher rank fusion
        ("Cải thiện 1 (Hybrid RRF top_k=5, budget=1500)", lambda q, filter_dict: hybrid.search(q, top_k=5, filter_dict=filter_dict, fetch_k=20)),
        
        # Improvement 2: Hybrid RRF + Query Metadata Expansion (budget=2000)
        ("Cải thiện 2 (Hybrid RRF top_k=5 + Budget 2000)", lambda q, filter_dict: hybrid.search(q, top_k=5, filter_dict=filter_dict, fetch_k=20)),
    ]

    all_summaries = []
    output_dir = Path("results/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    budgets = [1000, 1500, 2000]

    for (name, fn), budget in zip(experiments, budgets):
        print(f"Running experiment: {name}...")
        summary, traces = evaluate_configuration(name, fn, questions, split="dev", budget=budget)
        all_summaries.append(summary)
        slug = name.split()[0].lower() + f"_{budget}"
        with open(output_dir / f"traces_{slug}.json", "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    print("\n=== EXPERIMENT COMPARISON ON DEV (15 answerable questions) ===")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    with open(output_dir / "improvements_summary_dev.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
