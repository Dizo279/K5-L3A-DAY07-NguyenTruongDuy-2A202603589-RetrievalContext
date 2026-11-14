"""Agent implementation for Day 7 Agentic RAG.
Provides multi-turn tool calling, context budgeting, and evidence citations.
Uses OpenRouter (google/gemini-2.5-flash) for LLM decision making.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval import DenseRetriever, BM25Retriever, HybridRetriever

load_dotenv()

SYSTEM_PROMPT = """You are an expert AI research assistant analyzing the World Bank Commodity Markets Outlook report.
Your task is to answer user questions accurately based ONLY on the evidence retrieved from the documents.

Rules:
1. You have tools to search documents:
   - `search_vector`: Dense semantic search in vector database.
   - `search_keyword`: Lexical BM25 keyword search.
   - `search_hybrid`: Combined hybrid search (Dense + BM25 with RRF).
2. You can perform multiple search turns if the first search does not find adequate evidence.
3. Once you have enough evidence, provide a final grounded answer citing the chunk ID in brackets, e.g. [cmo-october-2026-p9-fixed-0001].
4. CRITICAL: If the document does not contain sufficient evidence to answer the question, or if the topic is not covered in the report, explicitly state that there is insufficient evidence to answer (Abstain). Do NOT hallucinate or assume facts outside the retrieved text.
5. In your answer, ONLY cite chunk IDs that are actually present in the context.
"""

class AgenticRAG:
    def __init__(
        self,
        retriever_type: str = "hybrid",
        model: str = "google/gemini-2.5-flash",
        max_turns: int = 3,
        context_budget: int = 2000,
        top_k: int = 4
    ):
        self.retriever_type = retriever_type
        self.model = model
        self.max_turns = max_turns
        self.context_budget = context_budget
        self.top_k = top_k

        # Load chunks for BM25
        with open("data/converted/chunks_a.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.dense = DenseRetriever(strategy="A")
        self.bm25 = BM25Retriever(chunks=self.chunks)
        self.hybrid = HybridRetriever(dense_retriever=self.dense, bm25_retriever=self.bm25)

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_hybrid",
                    "description": "Perform hybrid semantic and keyword retrieval over the report chunks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query string."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_vector",
                    "description": "Perform dense semantic vector search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The semantic query string."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_keyword",
                    "description": "Perform BM25 keyword search.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The exact keywords."}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = args.get("query", "")
        if name == "search_vector":
            return self.dense.search(query, top_k=self.top_k)
        elif name == "search_keyword":
            return self.bm25.search(query, top_k=self.top_k)
        else: # search_hybrid
            return self.hybrid.search(query, top_k=self.top_k)

    def run(self, question: str, filter_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        start_time = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        context_chunks: Dict[str, Dict[str, Any]] = {}
        events: List[Dict[str, Any]] = []
        turn = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        while turn < self.max_turns:
            turn += 1
            t_call_start = time.time()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=800,
                    temperature=0.0
                )
            except Exception as e:
                events.append({"event": "error", "message": str(e), "turn": turn})
                return {
                    "question": question,
                    "status": "error",
                    "answer": f"Error calling model: {e}",
                    "citations": [],
                    "events": events,
                    "latency": time.time() - start_time
                }

            usage = response.usage
            if usage:
                total_prompt_tokens += usage.prompt_tokens or 0
                total_completion_tokens += usage.completion_tokens or 0

            choice = response.choices[0]
            msg = choice.message
            tool_calls = msg.tool_calls

            if not tool_calls:
                # Agent decided to answer
                answer_text = msg.content or ""
                # extract citations like [cmo-october-2026-p9-fixed-0001]
                cited_ids = re.findall(r'\[([a-zA-Z0-9_\-]+)\]', answer_text)
                valid_citations = [cid for cid in cited_ids if cid in context_chunks]

                is_abstain = any(p in answer_text.lower() for p in [
                    "insufficient evidence", "not mentioned", "not provided", "does not contain",
                    "no information", "cannot be answered", "insufficient information"
                ])
                status = "abstain" if is_abstain else "answered"

                events.append({"event": "answer", "turn": turn, "status": status})
                return {
                    "question": question,
                    "status": status,
                    "answer": answer_text,
                    "citations": valid_citations,
                    "all_cited": cited_ids,
                    "context_chunks": list(context_chunks.values()),
                    "events": events,
                    "turns": turn,
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                    "latency": time.time() - start_time
                }

            # Handle tool calls
            messages.append(msg)
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except Exception:
                    fn_args = {"query": tc.function.arguments}

                t_exec_start = time.time()
                hits = self._execute_tool(fn_name, fn_args)
                t_exec = time.time() - t_exec_start

                # Add to context_chunks keeping budget
                newly_added = 0
                for hit in hits:
                    cid = hit["chunk_id"]
                    if cid not in context_chunks:
                        # check budget
                        curr_len = sum(len(c["text"]) for c in context_chunks.values())
                        if curr_len + len(hit["text"]) <= self.context_budget:
                            context_chunks[cid] = hit
                            newly_added += 1

                events.append({
                    "event": "search",
                    "tool": fn_name,
                    "query": fn_args.get("query"),
                    "hits_count": len(hits),
                    "newly_added": newly_added,
                    "latency": t_exec,
                    "turn": turn
                })

                # Format tool output for assistant
                tool_output_str = "\n".join([f"[{h['chunk_id']}] {h['text']}" for h in hits])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn_name,
                    "content": tool_output_str
                })

        # Max turns reached, force final answer
        messages.append({
            "role": "user",
            "content": "Please synthesize your final answer now based on the retrieved context, citing chunk IDs [chunk_id]. If evidence is missing, state insufficient evidence."
        })
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.0
        )
        answer_text = resp.choices[0].message.content or ""
        cited_ids = re.findall(r'\[([a-zA-Z0-9_\-]+)\]', answer_text)
        valid_citations = [cid for cid in cited_ids if cid in context_chunks]
        is_abstain = any(p in answer_text.lower() for p in [
            "insufficient evidence", "not mentioned", "not provided", "does not contain",
            "no information", "cannot be answered", "insufficient information"
        ])

        return {
            "question": question,
            "status": "abstain" if is_abstain else "answered",
            "answer": answer_text,
            "citations": valid_citations,
            "all_cited": cited_ids,
            "context_chunks": list(context_chunks.values()),
            "events": events,
            "turns": turn,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "latency": time.time() - start_time
        }
