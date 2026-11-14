import json
import time
from pathlib import Path
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever

def hit(chunk, gold_list):
    for g in gold_list:
        if chunk.get("source_id") == g.get("source_id") and chunk.get("page") == g.get("page"):
            c_text = ' '.join(chunk.get("text", "").split()).lower()
            e_text = ' '.join(g.get("evidence", "").split()).lower()
            # If evidence is found in chunk or chunk is on gold page and contains significant quote words
            if e_text in c_text:
                return True
            # Secondary check: if 80% of quote words appear in chunk on that gold page
            e_words = [w for w in e_text.split() if len(w) > 3]
            if e_words and sum(1 for w in e_words if w in c_text) / len(e_words) >= 0.8:
                return True
    return False

def evaluate_retriever(name, retriever_fn, questions, split="dev", budget=1200):
    subset = [q for q in questions if q.get("split") == split and q.get("type") != "unanswerable"]
    total = len(subset)
    hit_1 = 0
    hit_3 = 0
    context_evidence = 0
    traces = []

    start_time = time.time()
    for q in subset:
        query_text = q["question"]
        q_filter = q.get("filter", {})
        
        # search
        t0 = time.time()
        results = retriever_fn(query_text, top_k=3, filter_dict=q_filter)
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
            "results": [{"id": r["id"], "page": r["page"], "score": r["score"]} for r in results]
        })

    total_time = time.time() - start_time
    summary = {
        "retriever": name,
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
        ("Dense Retrieval (Baseline)", lambda q, top_k, filter_dict: dense.search(q, top_k=top_k, filter_dict=filter_dict)),
        ("Lexical Retrieval (BM25)", lambda q, top_k, filter_dict: bm25.search(q, top_k=top_k, filter_dict=filter_dict)),
        ("Hybrid Retrieval (RRF)", lambda q, top_k, filter_dict: hybrid.search(q, top_k=top_k, filter_dict=filter_dict)),
    ]

    all_summaries = []
    output_dir = Path("results/retrieval_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in experiments:
        print(f"Running benchmark for: {name} (split: dev)...")
        summary, traces = evaluate_retriever(name, fn, questions, split="dev")
        all_summaries.append(summary)
        slug = name.split()[0].lower()
        with open(output_dir / f"traces_{slug}_dev.json", "w", encoding="utf-8") as f:
            json.dump(traces, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ON DEV (15 answerable questions) ===")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    with open(output_dir / "retrieval_summary_dev.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
