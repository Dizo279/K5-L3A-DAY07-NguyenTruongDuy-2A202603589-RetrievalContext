import json
import time
from pathlib import Path
from src.agent import AgenticRAG

def run_agent_eval(agent, questions, split_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subset = [q for q in questions if q.get("split") == split_name]
    traces = []
    cases = []
    reviews = []

    print(f"\n--- Running Agent Evaluation for Split: {split_name} ({len(subset)} questions) ---")

    for q in subset:
        qid = q["id"]
        q_text = q["question"]
        is_answerable = (q.get("type") != "unanswerable")
        print(f"Processing {qid} ({q.get('type')})...")

        res = agent.run(q_text)
        res["query_id"] = qid
        res["split"] = split_name
        traces.append(res)

        cases.append({
            "id": qid,
            "split": split_name,
            "answerable": is_answerable
        })

        # Evaluate Answer & Citations
        status = res.get("status")
        answer = res.get("answer", "")
        citations = res.get("citations", [])
        all_cited = res.get("all_cited", [])

        # Citation support check: is each citation actually in context?
        ctx_ids = {c["chunk_id"] for c in res.get("context_chunks", [])}
        citation_support = [cid in ctx_ids for cid in citations]

        if is_answerable:
            # Check answer grade 0, 1, or 2
            if status == "abstain":
                grade = 0
                notes = "Incorrectly abstained on answerable question."
            else:
                # check expected answer overlap
                exp_words = [w for w in q.get("expected_answer", "").lower().split() if len(w) > 3]
                matched_words = sum(1 for w in exp_words if w in answer.lower())
                ratio = matched_words / len(exp_words) if exp_words else 1.0

                if ratio >= 0.6:
                    grade = 2
                    notes = "Fully correct and grounded."
                elif ratio >= 0.3:
                    grade = 1
                    notes = "Partially correct."
                else:
                    grade = 0
                    notes = "Incorrect or insufficient grounding."

            reviews.append({
                "query_id": qid,
                "answer_grade": grade,
                "citation_support": citation_support,
                "notes": notes
            })
        else:
            # Unanswerable question
            abstention_correct = (status == "abstain")
            reviews.append({
                "query_id": qid,
                "abstention_correct": abstention_correct,
                "citation_support": citation_support,
                "notes": "Correctly abstained" if abstention_correct else "Failed to abstain"
            })

    # Save outputs
    with open(out_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)
    with open(out_dir / "cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    with open(out_dir / "reviews.json", "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"Completed split {split_name}. Files saved to {out_dir}")

def main():
    with open("data/converted/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    agent = AgenticRAG(
        retriever_type="hybrid",
        model="google/gemini-2.5-flash",
        max_turns=3,
        context_budget=2000,
        top_k=4
    )

    # Run Dev evaluation (17 questions)
    run_agent_eval(agent, questions, "dev", "starter/results/agent-dev")

    # Run Test evaluation (7 questions)
    run_agent_eval(agent, questions, "test", "starter/results/agent-test")

if __name__ == "__main__":
    main()
