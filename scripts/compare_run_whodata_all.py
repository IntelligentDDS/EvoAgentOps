# compare_run_whodata_all.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import threading

from evoagentops.judge import Judge
from evoagentops.config import Config
from evoagentops.util import init_logger, logger, set_call_context, acall_embedding, acall_llm, RetryableError
from evoagentops.prompt import get_rerank_system_prompt, get_keyword_system_prompt, format_principles_for_prompt
from compare_run_whodata import convert_who_to_our_format, extract_prediction_from_judge
from a_judge_principle import PrincipleRetriever, RetrievalCache, get_config_from_model
from typing import List
from pydantic import BaseModel, Field
import numpy as np
from collections import defaultdict

# Thread-safe lock for writing to shared result file
_result_file_lock = threading.Lock()


class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(description="3-5 search keywords")


class RerankOutput(BaseModel):
    sorted_indices: List[int] = Field(description="Sorted indices")
    reason: str = Field(description="Brief rationale")


def append_result_to_file(result_file: Path, record: dict):
    """Thread-safe append record to jsonl file"""
    with _result_file_lock:
        with open(result_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_all_cases(data_dir: Path, dataset: str) -> list:
    """Get all JSON files for a dataset"""
    dataset_dir = data_dir / dataset
    if not dataset_dir.exists():
        return []
    return sorted([f for f in dataset_dir.glob("*.json") if f.is_file()])


def load_completed_results(output_base: Path, model: str, config_name: str, dataset: str) -> list:
    """Load all completed results"""
    results = []
    output_dir = output_base / model / config_name / dataset
    if not output_dir.exists():
        return results
    for case_folder in output_dir.iterdir():
        if case_folder.is_dir():
            output_file = case_folder / "output.json"
            if output_file.exists():
                try:
                    with open(output_file, encoding="utf-8") as f:
                        results.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Load failed {output_file}: {e}")
    return results


class PrincipleRetriever:
    """Retrieve top-k principles from principlebank using embedding similarity"""

    def __init__(
        self,
        principlebank_path: str,
        config: Config,
        functions: list = None,
        min_sim: float = 0.3,
        use_rerank: bool = False,
        top_e: int = 20,  # embedding retrieval
        top_l: int = 20,  # keyword retrieval
        retrieval_cache: RetrievalCache = None,
        rerank_config: Config = None,
    ):
        self.config = config
        self.rerank_config = rerank_config or config
        self.functions = functions or ["judge"]
        self.principles = {"global": [], "agent": defaultdict(list)}
        self._load_principlebank(principlebank_path)
        self.min_sim = min_sim
        self.use_rerank = use_rerank
        self.top_e = top_e
        self.top_l = top_l
        self.retrieval_cache = retrieval_cache

    def _load_principlebank(self, path: str):
        """Load and organize principles by level and agent_name"""
        if not Path(path).exists():
            logger.warning(f"Principlebank not found: {path}")
            return
        empty_emb_count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    p = json.loads(line)
                    if not p.get("embedding"):
                        empty_emb_count += 1
                        continue  # Skip principles without embedding
                    entry = {
                        "title": p.get("title", ""),
                        "content": p.get("content", ""),
                        "embedding": p.get("embedding", []),
                        "function": p.get("function", ""),
                    }
                    if p.get("type", "global") == "global":
                        self.principles["global"].append(entry)
                    else:
                        self.principles["agent"][p.get("agent_name", "")].append(entry)
                except:
                    continue
        if empty_emb_count:
            logger.warning(f"Skipped {empty_emb_count} principles without embedding")
        logger.info(
            f"Loaded principlebank: {len(self.principles['global'])} global, "
            f"{sum(len(v) for v in self.principles['agent'].values())} agent"
        )

    async def _rank_by_similarity(self, query_emb: list, principles: list) -> tuple:
        """Returns (results, scored_list_with_sim)"""
        query = np.array(query_emb)
        scored = []
        for p in principles:
            if emb := p.get("embedding"):
                doc = np.array(emb)
                sim = float(np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc) + 1e-8))
                if sim >= self.min_sim:
                    scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        limit = self.top_e
        top_scored = scored[:limit]
        # Return results and log data (without embedding for size)
        results = [p for _, p in top_scored]
        log_data = [{"title": p["title"], "content": p["content"], "sim": round(s, 4)} for s, p in top_scored]
        return results, log_data

    async def _rerank_by_llm(self, task_text: str, candidates: list) -> tuple:
        """LLM sorts ALL candidates by relevance. Returns (sorted_candidates, rerank_log)"""

        n = len(candidates)
        if n == 0:
            return [], {"sorted_indices": [], "reason": "empty"}
        task_text = task_text[:1000] if len(task_text) > 1000 else task_text
        cand_lines = [f"[{i}] {p['title']}: {p['content']}" for i, p in enumerate(candidates)]
        messages = [
            {"role": "system", "content": get_rerank_system_prompt("judge")},
            {
                "role": "user",
                "content": f"<task>{task_text}</task>\n<candidates>\n{chr(10).join(cand_lines)}\n</candidates>\nSort all {n} indices.",
            },
        ]
        err_msg = ""
        try:
            resp = await acall_llm(messages, self.rerank_config, output_schema=RerankOutput)
            result = json.loads(resp)
            indices = result.get("sorted_indices", [])
            reason = result.get("reason", "")
            # Dedupe, filter valid, append missing
            seen = set()
            valid = [i for i in indices if 0 <= i < n and i not in seen and not seen.add(i)]
            for i in range(n):
                if i not in seen:
                    valid.append(i)
            sorted_cands = [candidates[i] for i in valid]
            rerank_log = {"sorted_indices": valid, "reason": reason}
            logger.info(f"Rerank: n={n}, sorted={valid[:5]}..., reason={reason[:80]}")
            return sorted_cands, rerank_log
        except RetryableError:
            err_msg = "retryable"
            logger.warning("Rerank retryable, keep original order")
        except Exception as e:
            err_msg = str(e)[:50]
            logger.warning(f"Rerank failed: {e}, keep original order")
        return candidates, {"sorted_indices": list(range(n)), "reason": f"fallback:{err_msg}"}

    async def _expand_keywords(self, task_text: str) -> list:
        """LLM generates search keywords for judge_principle retrieval"""

        messages = [
            {"role": "system", "content": get_keyword_system_prompt("judge")},
            {
                "role": "user",
                "content": f"<task>{task_text[:500]}</task>",
            },
        ]
        try:
            resp = await acall_llm(messages, self.rerank_config, output_schema=KeywordsOutput)
            result = json.loads(resp)
            keywords = result.get("keywords", [])
            logger.info(f"Expand keywords: {keywords}")
            return keywords
        except Exception as e:
            logger.warning(f"Expand keywords failed: {e}")
            return []

    def _search_by_keywords(self, keywords: list, principles: list) -> tuple:
        """Full phrase (weight=2) + split words (weight=1), dedupe terms."""
        if not keywords:
            return [], []
        # Build match terms with weights, dedupe by keeping max weight
        term_weights = {}  # term -> weight
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if len(kw_lower) >= 2:
                term_weights[kw_lower] = max(term_weights.get(kw_lower, 0), 2)
            for w in kw_lower.split():
                if len(w) >= 3 and w != kw_lower:
                    term_weights[w] = max(term_weights.get(w, 0), 1)
        if not term_weights:
            return [], []
        scored = []
        for p in principles:
            text = f"{p.get('title','')} {p.get('content','')}".lower()
            score = sum(wt for t, wt in term_weights.items() if t in text)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        top_scored = scored[: self.top_l]
        results = [p for _, p in top_scored]
        log_data = [{"title": p["title"], "content": p["content"], "score": s} for s, p in top_scored]
        return results, log_data

    async def retrieve_once(self, task, original_case_id: str, agent_steps: list) -> dict:
        """Retrieve and cache full sorted results. Returns cached data."""
        # Check cache first
        if self.retrieval_cache and (cached := self.retrieval_cache.get(original_case_id)):
            logger.info(f"Cache hit: {original_case_id}")
            return cached

        task_text = task if isinstance(task, str) else json.dumps(task, ensure_ascii=False)
        query_emb = await acall_embedding(task_text, self.config)
        # Include None to match agent_name: null in principlebank
        agent_names = list({s.get("agent_name") for s in agent_steps})
        global_filtered = [p for p in self.principles["global"] if p.get("function") in self.functions]

        # Global level: hybrid retrieval
        emb_cands, _ = await self._rank_by_similarity(query_emb, global_filtered)
        keywords = await self._expand_keywords(task_text) if self.use_rerank else []
        kw_cands, _ = self._search_by_keywords(keywords, global_filtered)

        seen = {p["title"] for p in emb_cands}
        merged = emb_cands + [p for p in kw_cands if p["title"] not in seen]

        # Rerank to get full sorted list
        if self.use_rerank and merged:
            global_sorted, rerank_log = await self._rerank_by_llm(task_text, merged)
        else:
            global_sorted = merged

        # Agent level
        agent_sorted = {}
        # All agents retrieve from the same principlebank (agent_name=null)
        source_name = None
        agent_filtered = [
            p for p in self.principles["agent"].get(source_name, []) if p.get("function") in self.functions
        ]
        # Retrieve once, share for all agents
        if agent_filtered and agent_names:
            emb_cands, _ = await self._rank_by_similarity(query_emb, agent_filtered)
            kw_cands, _ = self._search_by_keywords(keywords, agent_filtered)
            seen = {p["title"] for p in emb_cands}
            merged = emb_cands + [p for p in kw_cands if p["title"] not in seen]
            if self.use_rerank and merged:
                sorted_list, _ = await self._rerank_by_llm(task_text, merged)
            else:
                sorted_list = merged
            shared_result = [{"title": p["title"], "content": p["content"]} for p in sorted_list]
            for agent_name in agent_names:
                agent_sorted[agent_name] = shared_result

        # Build cache data (without embedding to save space)
        cache_data = {
            "original_case_id": original_case_id,
            "task_text": task_text[:500],
            "keywords": keywords,
            "global_sorted": [{"title": p["title"], "content": p["content"]} for p in global_sorted],
            "agent_sorted": agent_sorted,
            "config": {"top_e": self.top_e, "top_l": self.top_l, "use_rerank": self.use_rerank},
        }

        # Save to cache
        if self.retrieval_cache:
            self.retrieval_cache.save(original_case_id, cache_data)

        agent_counts = {k: len(v) for k, v in agent_sorted.items()}
        logger.info(f"Retrieved: {original_case_id}, global={len(global_sorted)}, agents={agent_counts}")
        return cache_data


class WhoJudgeStore:
    """Store for Who&When judge results with resume support"""

    def __init__(self, output_dir: str, n: int):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n = n
        self._done, self._raw_done, self._raw_cache = self._load_done()
        logger.info(f"Store loaded: done={len(self._done)}, raw_cases={len(self._raw_done)}")

    def _load_done(self) -> tuple:
        done = set()
        raw_done = {}
        raw_cache = {}
        # Load final results
        for case_folder in self.output_dir.iterdir():
            if case_folder.is_dir():
                output_file = case_folder / "output.json"
                if output_file.exists():
                    try:
                        with open(output_file, encoding="utf-8") as f:
                            data = json.load(f)
                            if data.get("case_id"):
                                done.add(data["case_id"])
                    except Exception as e:
                        logger.warning(f"Load output failed {output_file}: {e}")
        # Load raw results
        raw_file = self.output_dir / "judge_raw.jsonl"
        if raw_file.exists():
            with open(raw_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        cid, idx = data.get("case_id"), data.get("idx")
                        if cid is not None and idx is not None:
                            raw_done.setdefault(cid, set()).add(idx)
                            raw_cache.setdefault(cid, {})[idx] = data
                    except Exception as e:
                        logger.warning(f"Parse raw line failed: {e}")
        return done, raw_done, raw_cache

    def is_done(self, case_id: str) -> bool:
        return case_id in self._done

    def get_missing_indices(self, case_id: str) -> list:
        done_indices = self._raw_done.get(case_id, set())
        return [i for i in range(self.n) if i not in done_indices]

    def save_raw(self, case_id: str, idx: int, result: dict):
        record = {"case_id": case_id, "idx": idx, "result": result}
        with open(self.output_dir / "judge_raw.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._raw_done.setdefault(case_id, set()).add(idx)
        self._raw_cache.setdefault(case_id, {})[idx] = record

    def load_raw_results(self, case_id: str) -> list:
        if case_id not in self._raw_cache:
            return []
        return [self._raw_cache[case_id][i]["result"] for i in sorted(self._raw_cache[case_id].keys())]

    def mark_done(self, case_id: str):
        self._done.add(case_id)


async def retrieve_all_cases(
    dataset: str,
    model: str,
    data_dir: Path,
    output_base: Path,
    principlebank_dir: Path,
    use_rerank: bool = True,
) -> dict:
    """Pre-retrieve principles for all cases in dataset. Returns {case_id: cache_data}"""
    cases = get_all_cases(data_dir, dataset)
    if not cases:
        return {}

    # Setup retriever
    base_url, api_key, model_name = get_config_from_model(model)
    principlebank = principlebank_dir / model / "principlebank" / "principlebank.jsonl"
    if not principlebank.exists():
        principlebank = principlebank_dir / "principlebank.jsonl"
    if not principlebank.exists():
        logger.warning(f"Principlebank not found: {principlebank}")
        return {}

    cache_dir = output_base / model / "retrieval_cache" / dataset
    cache_file = cache_dir / "retrieval.jsonl"
    retrieval_cache = RetrievalCache(str(cache_file))

    config = Config(output_dir=str(cache_dir))
    if base_url:
        config.openai_base_url = base_url
    if api_key:
        config.openai_api_key = api_key
    if model_name:
        config.openai_model = model_name

    retriever = PrincipleRetriever(
        str(principlebank),
        config,
        use_rerank=use_rerank,
        top_e=20,
        top_l=20,
        retrieval_cache=retrieval_cache,
        rerank_config=config,
    )

    # Retrieve for each case
    all_cache = {}
    pending = [p for p in cases if not retrieval_cache.get(p.stem)]
    logger.info(f"[RETRIEVE] {model}/{dataset}: total={len(cases)}, pending={len(pending)}")

    for json_path in tqdm(pending, desc=f"Retrieve {dataset[:15]}", ncols=100, disable=len(pending) == 0):
        case_id = json_path.stem
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            task_text = data.get("question", "")
            # Extract real agent names from history
            is_handcrafted = "Hand-Crafted" in str(json_path)
            index_key = "role" if is_handcrafted else "name"
            history = data.get("history", [])
            agent_names = list({entry.get(index_key) for entry in history if entry.get(index_key)})
            agent_steps = [{"agent_name": name} for name in agent_names]
            cache_data = await retriever.retrieve_once(task_text, case_id, agent_steps)
            all_cache[case_id] = cache_data
        except Exception as e:
            logger.error(f"Retrieve failed: {case_id}: {e}")

    # Load all cached
    for json_path in cases:
        case_id = json_path.stem
        if case_id not in all_cache and (cached := retrieval_cache.get(case_id)):
            all_cache[case_id] = cached

    logger.info(f"[RETRIEVE] {model}/{dataset}: {len(all_cache)} cases retrieved/cached")
    return all_cache


async def process_dataset(
    dataset: str,
    model: str,
    n: int,
    top_k: int,
    data_dir: Path,
    output_base: Path,
    semaphore: asyncio.Semaphore,
    result_file: Path,
    retrieval_cache: dict = None,
    global_level: bool = True,
    agent_level: bool = True,
) -> dict:
    """Process all cases for one (model, config, dataset)"""
    use_principles = top_k > 0
    config_name = f"n{n}_topk{top_k}" if use_principles else f"n{n}"

    cases = get_all_cases(data_dir, dataset)
    if not cases:
        logger.warning(f"[{model}/{config_name}/{dataset}] No cases found")
        return {"total": 0, "pending": 0, "success": 0, "failed": 0}

    # Setup config
    base_url, api_key, model_name = get_config_from_model(model)

    output_dir_base = output_base / model / config_name / dataset
    output_dir_base.mkdir(parents=True, exist_ok=True)

    # Setup store with resume support
    store = WhoJudgeStore(str(output_dir_base), n)

    # Filter pending cases
    pending = [p for p in cases if not store.is_done(p.stem)]
    done_count = len(cases) - len(pending)
    logger.info(f"[{model}/{config_name}/{dataset}] Total={len(cases)}, Done={done_count}, Pending={len(pending)}")

    if not pending:
        results = load_completed_results(output_base, model, config_name, dataset)
        agent_correct = sum(1 for r in results if r.get("is_agent_correct"))
        step_correct = sum(1 for r in results if r.get("is_step_correct"))
        return {
            "total": len(cases),
            "pending": 0,
            "success": len(results),
            "failed": 0,
            "agent_acc": agent_correct / len(results) * 100 if results else 0,
            "step_acc": step_correct / len(results) * 100 if results else 0,
        }

    async def run_one(json_path: Path) -> dict:
        async with semaphore:
            case_id = json_path.stem
            output_dir = output_dir_base / case_id
            output_dir.mkdir(parents=True, exist_ok=True)

            config = Config(output_dir=str(output_dir))
            if base_url:
                config.openai_base_url = base_url
            if api_key:
                config.openai_api_key = api_key
            if model_name:
                config.openai_model = model_name

            try:
                # Load and convert
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
                data["is_handcrafted"] = "Hand-Crafted" in str(json_path)

                agent_steps, agent_dependency, agent_settings, label = convert_who_to_our_format(data)
                principles_str = ""
                if use_principles and retrieval_cache and case_id in retrieval_cache:
                    # Use "both" to include agent-level (null agent_name) principles
                    principles_str = format_principles_for_prompt(retrieval_cache[case_id], top_k, "both")

                # Save converted data
                with open(output_dir / "data.json", "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "original": data,
                            "converted_steps": agent_steps,
                            "label": label,
                            "principles_used": bool(principles_str),
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

                # Run Judge
                judge = Judge(agent_steps, agent_dependency, agent_settings, config)
                set_call_context(case_id=case_id, stage="judge")
                # Get missing indices for resume
                missing = store.get_missing_indices(case_id)
                if missing:

                    async def judge_one(idx: int) -> tuple:
                        r = await judge.judge_once(
                            task_compare_withlabel="fail",
                            global_level=global_level,
                            agent_level=agent_level,
                            principles_str=principles_str,
                        )
                        store.save_raw(case_id, idx, r)
                        return idx, r

                    results = await asyncio.gather(*[judge_one(i) for i in missing], return_exceptions=True)
                    ok = sum(1 for r in results if not isinstance(r, Exception))
                    if ok < len(missing):
                        failed_idx = [i for i, r in zip(missing, results) if isinstance(r, Exception)]
                        logger.warning(f"Judge partial: {case_id}, ok={ok}/{len(missing)}, failed={failed_idx}")

                # Load all raw results
                all_results = store.load_raw_results(case_id)
                if not all_results:
                    logger.warning(f"No valid results: {case_id}")
                    return {"success": False, "case_id": case_id, "error": "no_results"}

                result = all_results[0] if all_results else {}

                # Extract prediction
                pred_agent, pred_step_0idx = extract_prediction_from_judge(result, agent_steps)

                # Evaluate
                label_agent = label["mistake_agent"]
                label_step_0idx = label["mistake_step_0idx"]
                is_agent_correct = (label_agent in pred_agent) if pred_agent else False
                is_step_correct = label_step_0idx == pred_step_0idx

                output = {
                    "case_id": case_id,
                    "model": model,
                    "config": config_name,
                    "predicted_agent": pred_agent,
                    "predicted_step": pred_step_0idx,
                    "label_agent": label_agent,
                    "label_step": label_step_0idx,
                    "is_agent_correct": is_agent_correct,
                    "is_step_correct": is_step_correct,
                }

                output_full = {**output, "judge_result": result, "all_results_count": len(all_results)}
                with open(output_dir / "output.json", "w", encoding="utf-8") as f:
                    json.dump(output_full, f, ensure_ascii=False, indent=2)

                store.mark_done(case_id)
                # Append to unified result file
                output["dataset"] = dataset
                append_result_to_file(result_file, output)

                return {"success": True, "case_id": case_id, "result": output}

            except Exception as e:
                logger.error(f"Failed: {json_path.stem}: {e}")
                return {"success": False, "case_id": json_path.stem, "error": str(e)}

    # Run with progress bar
    tasks = [run_one(p) for p in pending]
    results = []
    with tqdm(total=len(pending), desc=f"{model[:10]}/{config_name}/{dataset[:15]}", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
            if result["success"]:
                r = result["result"]
                pbar.set_postfix_str(f"a={r['is_agent_correct']},s={r['is_step_correct']}")

    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    # Load all for accuracy
    all_results = load_completed_results(output_base, model, config_name, dataset)
    agent_correct = sum(1 for r in all_results if r.get("is_agent_correct"))
    step_correct = sum(1 for r in all_results if r.get("is_step_correct"))

    stats = {
        "total": len(cases),
        "pending": len(pending),
        "success": success_count,
        "failed": failed_count,
        "agent_acc": agent_correct / len(all_results) * 100 if all_results else 0,
        "step_acc": step_correct / len(all_results) * 100 if all_results else 0,
    }

    logger.info(
        f"[{model}/{config_name}/{dataset}] Done: success={success_count}, failed={failed_count}, "
        f"agent_acc={stats['agent_acc']:.2f}%, step_acc={stats['step_acc']:.2f}%"
    )

    return stats


async def run_all(
    models: list,
    datasets: list,
    n_list: list,
    top_k_list: list,
    data_dir: Path,
    output_base: Path,
    principlebank_dir: Path,
    max_concurrency: int = 5,
    global_level: bool = True,
    agent_level: bool = True,
    use_rerank: bool = True,
):
    """Run all combinations"""
    total_configs = len(n_list) * len(top_k_list)
    total_tasks = len(models) * len(datasets) * total_configs
    logger.info(
        f"{'='*60}\n[BATCH] {len(models)} models x {len(datasets)} datasets x {total_configs} configs = {total_tasks}\n{'='*60}"
    )

    model_semaphores = {model: asyncio.Semaphore(max_concurrency) for model in models}
    all_stats = []
    result_file = output_base / "all_results.jsonl"

    # Clear result file to avoid duplicates from previous runs
    if result_file.exists():
        existing = {}
        with open(result_file, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    key = (r.get("model"), r.get("config"), r.get("dataset"), r.get("case_id"))
                    existing[key] = r
                except Exception as e:
                    logger.warning(f"Parse result line failed: {e}")
        with open(result_file, "w", encoding="utf-8") as f:
            for r in existing.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        logger.info(f"Deduped result file: {len(existing)} records")
    # Phase 1: Pre-retrieve for all (model, dataset) pairs where top_k > 0
    retrieval_data = {}
    if any(k > 0 for k in top_k_list):
        logger.info(f"{'='*60}\n[PHASE 1] Pre-retrieving principles\n{'='*60}")

        async def retrieve_one(model, dataset):
            try:
                cache = await retrieve_all_cases(dataset, model, data_dir, output_base, principlebank_dir, use_rerank)
                return (model, dataset), cache
            except Exception as e:
                logger.error(f"[RETRIEVE] {model}/{dataset} failed: {e}")
                return (model, dataset), {}

        retrieve_tasks = [retrieve_one(m, d) for m in models for d in datasets]
        logger.info(f"[PHASE 1] Parallel retrieving {len(retrieve_tasks)} (model, dataset) pairs")
        retrieve_results = await asyncio.gather(*retrieve_tasks)
        retrieval_data = dict(retrieve_results)

    # Phase 2: Judge with different configs
    logger.info(f"{'='*60}\n[PHASE 2] Running Judge\n{'='*60}")

    async def run_one_config(dataset, model, n, top_k, retrieval_cache):
        stats = await process_dataset(
            dataset,
            model,
            n,
            top_k,
            data_dir,
            output_base,
            model_semaphores[model],
            result_file,
            retrieval_cache,
            global_level,
            agent_level,
        )
        stats.update({"model": model, "dataset": dataset, "n": n, "top_k": top_k})
        return stats

    # Build tasks with pre-fetched retrieval data
    tasks = []
    for dataset in datasets:
        for n in n_list:
            for top_k in top_k_list:
                for model in models:
                    cache = retrieval_data.get((model, dataset), {}) if top_k > 0 else None
                    tasks.append(run_one_config(dataset, model, n, top_k, cache))
    with tqdm(total=len(tasks), desc="Overall", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            stats = await coro
            all_stats.append(stats)
            pbar.update(1)
            pbar.set_postfix_str(f"{stats['model'][:8]}/{stats['dataset'][:10]} a={stats['agent_acc']:.1f}%")

    # Summary
    logger.info(f"\n{'='*60}\n[SUMMARY]")
    for model in models:
        model_stats = [s for s in all_stats if s["model"] == model]
        total_cases = sum(s["total"] for s in model_stats)
        if total_cases > 0:
            avg_agent_acc = sum(s["agent_acc"] * s["total"] for s in model_stats) / total_cases
            avg_step_acc = sum(s["step_acc"] * s["total"] for s in model_stats) / total_cases
            logger.info(f"[{model}] Cases={total_cases}, Agent_Acc={avg_agent_acc:.2f}%, Step_Acc={avg_step_acc:.2f}%")

    # Save summary
    summary_file = output_base / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSummary saved to: {summary_file}, Results saved to: {result_file}")


# Defaults
DEFAULT_MODELS = [
    "doubao-seed-1-8-251228",
    "deepseek-v3-2-251201",
    "gpt-5.2-chat",
    "claude-sonnet-4.5",
    "kimi-k2-250905",
]
DEFAULT_DATASETS = ["Algorithm-Generated", "Hand-Crafted"]
DEFAULT_N = [1]
DEFAULT_TOP_K = [0, 5, 10] + [3, 8, 15, 20]


def main():
    parser = argparse.ArgumentParser(description="Batch run Judge on Who&When dataset")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--top_k", type=int, nargs="+", default=DEFAULT_TOP_K)
    parser.add_argument("--data_dir", type=str, default="../datasets/Who&When")
    parser.add_argument("--output_dir", type=str, default="../results/run_whodata_results")
    parser.add_argument("--principlebank_dir", type=str, default="../results/run_whodata_train_bank")
    parser.add_argument("--max_concurrency", type=int, default=10)
    parser.add_argument("--global_level", type=bool, default=True)
    parser.add_argument("--agent_level", type=bool, default=True)
    parser.add_argument("--use_rerank", type=bool, default=True)
    args = parser.parse_args()

    load_dotenv()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    init_logger(str(output_base / "run.log"), level="INFO")

    asyncio.run(
        run_all(
            models=args.models,
            datasets=args.datasets,
            n_list=args.n,
            top_k_list=args.top_k,
            data_dir=Path(args.data_dir),
            output_base=output_base,
            principlebank_dir=Path(args.principlebank_dir),
            max_concurrency=args.max_concurrency,
            global_level=args.global_level,
            agent_level=args.agent_level,
            use_rerank=args.use_rerank,
        )
    )


if __name__ == "__main__":
    main()
