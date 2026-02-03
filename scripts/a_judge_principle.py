# main.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
from evoagentops.judge import Judge
from evoagentops.config import Config
from evoagentops.util import init_logger, logger, acall_embedding, acall_llm, RetryableError
import numpy as np
from pydantic import BaseModel, Field
from typing import List
from evoagentops.prompt import get_rerank_system_prompt, get_keyword_system_prompt, format_principles_for_prompt

FJ_SEP = "__fj__"

# === Environment variable keys for single mode ===
ENV_MODEL = "JUDGE_MODEL"
ENV_DATASET = "JUDGE_DATASET"
ENV_N = "JUDGE_N"
ENV_TOP_K = "JUDGE_TOP_K"
ENV_USE_PRINCIPLES = "JUDGE_USE_PRINCIPLES"

MODEL_PREFIX_MAP = {
    "deepseek": "DEEPSEEK",
    "gpt": "GPT",
    "seed": "SEED",
    "doubao": "SEED",
    "kimi": "KIMI",
    "claude": "CLAUDE",
}


def get_model_config(prefix: str) -> tuple:
    """Get (base_url, api_key, model) for a given env prefix"""
    return (
        os.getenv(f"{prefix}_OPENAI_BASE_URL"),
        os.getenv(f"{prefix}_OPENAI_API_KEY"),
        os.getenv(f"{prefix}_OPENAI_MODEL"),
    )


def get_config_from_model(model_name: str) -> tuple:
    """Get (base_url, api_key, model) for LLM based on model_name keyword match"""
    model_lower = model_name.lower() if model_name else ""
    for keyword, prefix in MODEL_PREFIX_MAP.items():
        if keyword in model_lower:
            return get_model_config(prefix)
    return get_model_config("SEED")


class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(description="3-5 search keywords for fault detection principles")


class RerankOutput(BaseModel):
    sorted_indices: List[int] = Field(description="ALL candidate indices sorted by relevance, most relevant first")
    reason: str = Field(description="Brief ranking rationale")


class RetrievalCache:
    """Cache retrieval results for reuse across different top_k values"""

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        logger.info(f"RetrievalCache loaded: {len(self._cache)} cases from {cache_file}")

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if cid := data.get("original_case_id"):
                            cache[cid] = data
                    except:
                        pass
        return cache

    def get(self, case_id: str) -> dict | None:
        return self._cache.get(case_id)

    def save(self, case_id: str, data: dict):
        data["original_case_id"] = case_id
        self._cache[case_id] = data
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


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
        agent_names = list({s["agent_name"] for s in agent_steps if s.get("agent_name")})
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
        for agent_name in agent_names:
            agent_filtered = [
                p for p in self.principles["agent"].get(agent_name, []) if p.get("function") in self.functions
            ]
            if not agent_filtered:
                continue
            emb_cands, _ = await self._rank_by_similarity(query_emb, agent_filtered)
            kw_cands, _ = self._search_by_keywords(keywords, agent_filtered)
            seen = {p["title"] for p in emb_cands}
            merged = emb_cands + [p for p in kw_cands if p["title"] not in seen]
            if self.use_rerank and merged:
                sorted_list, _ = await self._rerank_by_llm(task_text, merged)
            else:
                sorted_list = merged
            agent_sorted[agent_name] = [{"title": p["title"], "content": p["content"]} for p in sorted_list]

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

        logger.info(f"Retrieved: {original_case_id}, global={len(global_sorted)}, agents={list(agent_sorted.keys())}")
        return cache_data


async def retrieve_all_cases(dataset_name: str, model_name: str, args) -> dict:
    """Retrieve principles for all cases in dataset. Returns {orig_id: cache_data}"""
    parsed_dir = Path(args.parsed_dir) / dataset_name
    fault_dir = Path(args.fault_dir) / dataset_name / "test" / "fail"
    principlebank = Path(args.principlebank_dir) / model_name / dataset_name / "principlebank" / "principlebank.jsonl"

    if not fault_dir.exists():
        logger.warning(f"Fault dir not found: {fault_dir}, skip")
        return {}

    # Initialize cache (shared across all top_k)
    cache_dir = Path(args.output_dir) / model_name / "retrieval_cache" / dataset_name
    cache_file = cache_dir / "retrieval.jsonl"
    retrieval_cache = RetrievalCache(str(cache_file))

    # Config for retriever (only need embedding/rerank config, not top_k)
    config = Config(output_dir=str(cache_dir))
    init_logger(str(cache_dir / "run.log"), level="INFO")
    for attr in ["openai_base_url", "openai_api_key", "openai_model", "llm_max_concurrency"]:
        if (val := getattr(args, attr, None)) is not None:
            setattr(config, attr, val)

    # Separate config for keyword/rerank LLM based on model_name
    rerank_base_url, rerank_api_key, rerank_model = get_config_from_model(model_name)
    rerank_config = Config(output_dir=str(cache_dir))
    if rerank_base_url:
        rerank_config.openai_base_url = rerank_base_url
    if rerank_api_key:
        rerank_config.openai_api_key = rerank_api_key
    if rerank_model:
        rerank_config.openai_model = rerank_model
    if args.llm_max_concurrency:
        rerank_config.llm_max_concurrency = args.llm_max_concurrency
    logger.info(f"Rerank config: model={rerank_config.openai_model}, base_url={rerank_config.openai_base_url}")

    retriever = PrincipleRetriever(
        str(principlebank),
        config,
        use_rerank=getattr(args, "use_rerank", True),
        top_e=getattr(args, "top_e", 20),
        top_l=getattr(args, "top_l", 20),
        retrieval_cache=retrieval_cache,
        rerank_config=rerank_config,
    )

    # Group by original_case_id
    groups = defaultdict(list)
    for folder in sorted(fault_dir.iterdir()):
        if folder.is_dir():
            orig_id, _, _ = parse_folder_name(folder.name)
            groups[orig_id].append(folder)

    # Retrieve for each original case
    all_cache = {}
    for orig_id, folders in groups.items():
        with open(folders[0] / "agent_steps.json", encoding="utf-8") as f:
            first_steps = json.load(f)
        task = first_steps[0]["agent"]["input"]
        cache_data = await retriever.retrieve_once(task, orig_id, first_steps)
        all_cache[orig_id] = cache_data

    logger.info(f"[RETRIEVE] {dataset_name}: {len(all_cache)} cases retrieved/cached")
    return all_cache


class FaultJudgeStore:
    """Store for fault injection judge results with resume support"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._done, self._raw_done, self._raw_cache = self._load_done()
        logger.info(
            f"Store loaded: done={len(self._done)}, raw_cases={len(self._raw_done)}, raw_records={sum(len(v) for v in self._raw_cache.values())}"
        )

    def _load_done(self) -> tuple:
        done = set()
        raw_done = {}  # case_id -> set of idx
        raw_cache = {}  # case_id -> {idx: data}
        result_file = self.output_dir / "judge_results.jsonl"
        if result_file.exists():
            with open(result_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        done.add(json.loads(line).get("case_id", ""))
                    except:
                        pass
        # Load raw results for resume
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
                    except:
                        pass
        return done, raw_done, raw_cache

    def is_done(self, case_id: str) -> bool:
        return case_id in self._done

    def get_missing_indices(self, case_id: str, n: int) -> list:
        """Get indices that need to be judged"""
        done_indices = self._raw_done.get(case_id, set())
        return [i for i in range(n) if i not in done_indices]

    def save_raw(self, case_id: str, idx: int, result: dict):
        """Save single judge result for resume support"""
        record = {"case_id": case_id, "idx": idx, **result}
        with open(self.output_dir / "judge_raw.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._raw_done.setdefault(case_id, set()).add(idx)
        self._raw_cache.setdefault(case_id, {})[idx] = record

    def load_raw_results(self, case_id: str) -> tuple:
        """Load raw results for aggregation"""
        global_results, agent_results_dict = [], {}
        if case_id not in self._raw_cache:
            return global_results, agent_results_dict
        for idx, data in self._raw_cache[case_id].items():
            if gr := data.get("global_result"):
                global_results.append(gr)
            for ar in data.get("agent_results", []):
                key = f"{ar['agent_name']}_{ar['start_step']}"
                agent_results_dict.setdefault(key, []).append(ar)
        return global_results, agent_results_dict

    def save(self, case_id: str, result: dict, label: dict):
        record = {"case_id": case_id, "label": label, **result}
        with open(self.output_dir / "judge_results.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._done.add(case_id)
        # Also append to unified file
        unified = self.output_dir.parent.parent.parent / "all_results.jsonl"
        record["config"] = self.output_dir.parent.name  # e.g., n1_topk5
        record["dataset"] = self.output_dir.name
        with open(unified, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def aggregate(results: list) -> dict:
        """Aggregate multiple judge results using median score"""
        if len(results) <= 1:
            return results[0] if results else {}

        all_metrics = {m["metric"] for r in results for m in r.get("judge_result", [])}
        if not all_metrics:
            return results[0]

        aggregated = []
        hits = {i: 0 for i in range(len(results))}

        for metric in all_metrics:
            scored = [
                (len(m.get("reasons", [])), i, m)  # Use reasons count as severity proxy
                for i, r in enumerate(results)
                for m in r.get("judge_result", [])
                if m["metric"] == metric
            ]
            if scored:
                scored.sort()
                _, idx, data = scored[len(scored) // 2]
                aggregated.append(data)
                hits[idx] += 1

        best = results[max(hits, key=hits.get)]
        return {
            "statement_action": best.get("statement_action", []),
            "judge_result": aggregated,
            "fault_root_cause": best.get("fault_root_cause", []),
            "is_success": best.get("is_success", False),
        }


def parse_folder_name(name: str) -> tuple:
    """Parse: {original_case_id}__fj__{fault_type}__fj__{id}"""
    parts = name.split(FJ_SEP)
    return (parts[0], parts[1], int(parts[2])) if len(parts) == 3 else (name, "", 0)


async def process_dataset(
    dataset_name: str,
    model_name: str,
    n: int,
    top_k: int,
    args,
    retrieval_cache: dict = None,  # {orig_case_id: cache_data}
) -> int:
    """Process a single dataset"""
    # Build paths for this dataset
    config_name = f"n{n}_topk{top_k}" if args.use_principles else f"n{n}"
    parsed_dir = Path(args.parsed_dir) / dataset_name
    fault_dir = Path(args.fault_dir) / dataset_name / "test" / "fail"
    output_dir = Path(args.output_dir) / model_name / config_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(output_dir=str(output_dir))
    # Use model-specific config for Judge
    judge_base_url, judge_api_key, judge_model = get_config_from_model(model_name)
    if judge_base_url:
        config.openai_base_url = judge_base_url
    if judge_api_key:
        config.openai_api_key = judge_api_key
    if judge_model:
        config.openai_model = judge_model
    if args.llm_max_concurrency:
        config.llm_max_concurrency = args.llm_max_concurrency
    init_logger(str(output_dir / "run.log"), level="INFO")
    logger.info(f"Judge config: model={config.openai_model}")

    # Check if fault_dir exists
    if not fault_dir.exists():
        logger.warning(f"Fault dir not found: {fault_dir}, skip")
        return 0

    # Load shared data
    with open(parsed_dir / "agent_dependency.json", encoding="utf-8") as f:
        agent_dependency = json.load(f)
    with open(parsed_dir / "agent_settings.json", encoding="utf-8") as f:
        agent_settings = json.load(f)

    store = FaultJudgeStore(str(output_dir))

    # Group by original_case_id
    groups = defaultdict(list)
    for folder in sorted(fault_dir.iterdir()):
        if folder.is_dir():
            orig_id, _, _ = parse_folder_name(folder.name)
            groups[orig_id].append(folder)
    logger.info(f"Found {len(groups)} original cases, {sum(len(v) for v in groups.values())} variants")

    # Process
    count = 0
    for orig_id, folders in groups.items():
        # Format principles from pre-retrieved cache
        principles_str = ""
        if args.use_principles and retrieval_cache and orig_id in retrieval_cache:
            principles_str = format_principles_for_prompt(
                retrieval_cache[orig_id], top_k, getattr(args, "principle_scope", "both")
            )

        for folder in folders:
            case_id = folder.name
            if store.is_done(case_id):
                logger.info(f"Skip (done): {case_id}")
                continue
            try:
                with open(folder / "agent_steps.json", encoding="utf-8") as f:
                    steps = json.load(f)
                with open(folder / "label.json", encoding="utf-8") as f:
                    label = json.load(f)
                judge = Judge(steps, agent_dependency, agent_settings, config)
                # Get missing indices for resume
                missing = store.get_missing_indices(case_id, n)
                if missing:
                    logger.info(f"Judge start: {case_id}, todo={missing}, total={n}")

                    # Concurrent judge for missing indices (capture case_id explicitly)
                    async def judge_one(idx: int, cid: str = case_id) -> tuple:
                        r = await judge.judge_once(
                            task_compare_withlabel="fail",
                            global_level=args.global_level,
                            agent_level=args.agent_level,
                            principles_str=principles_str,
                        )
                        # Save raw immediately for resume
                        store.save_raw(cid, idx, r)
                        return idx, r

                    results = await asyncio.gather(*[judge_one(i) for i in missing], return_exceptions=True)
                    # Log results
                    ok = sum(1 for r in results if not isinstance(r, Exception))
                    fail = [i for i, r in zip(missing, results) if isinstance(r, Exception)]
                    logger.info(
                        f"Judge done: {case_id}, ok={ok}/{len(missing)}" + (f", failed_idx={fail}" if fail else "")
                    )
                # Load all raw results for aggregation
                global_results, agent_results_dict = store.load_raw_results(case_id)
                # Check if we have enough valid results
                if not global_results and not agent_results_dict:
                    logger.warning(f"Skip save (no valid results): {case_id}")
                    continue
                raw_cnt = len(store._raw_done.get(case_id, set()))
                if raw_cnt < n:
                    logger.warning(f"Partial results: {case_id}, raw={raw_cnt}/{n}, global={len(global_results)}")
                # Aggregate with metadata preserved
                agent_results = []
                for key, v in agent_results_dict.items():
                    if v:
                        agg = FaultJudgeStore.aggregate(v)
                        agg["agent_name"] = v[0].get("agent_name")
                        agg["start_step"] = v[0].get("start_step")
                        agg["end_step"] = v[0].get("end_step")
                        agent_results.append(agg)
                result = {
                    "global_result": FaultJudgeStore.aggregate(global_results) if global_results else {},
                    "agent_results": agent_results,
                }
                store.save(case_id, result, label)
                count += 1
                logger.info(
                    f"Done: {case_id}, global_ok={result.get('global_result', {}).get('is_success')}, agent_cnt={len(result.get('agent_results', []))}"
                )
            except Exception as e:
                logger.error(f"Failed: {case_id}: {e}")
    return count


async def run_single_task(args) -> int:
    """Run single (model, dataset, n, top_k) config from env vars"""
    model_name = os.getenv(ENV_MODEL)
    dataset_name = os.getenv(ENV_DATASET)
    n = int(os.getenv(ENV_N, "1"))
    top_k = int(os.getenv(ENV_TOP_K, "0"))
    use_principles = os.getenv(ENV_USE_PRINCIPLES, "false").lower() == "true"
    principle_scope = os.getenv("JUDGE_PRINCIPLE_SCOPE", "both")

    if not model_name or not dataset_name:
        logger.error(f"Missing env: {ENV_MODEL}={model_name}, {ENV_DATASET}={dataset_name}")
        return 1

    config_name = f"n{n}_topk{top_k}" if use_principles else f"n{n}"
    logger.info(f"[SINGLE] {model_name}/{config_name}/{dataset_name}")

    # Retrieve if needed
    retrieval_data = None
    if use_principles:
        try:
            retrieval_data = await retrieve_all_cases(dataset_name, model_name, args)
        except Exception as e:
            logger.error(f"[RETRIEVE] failed: {e}")
            retrieval_data = {}

    # Process
    args.use_principles = use_principles
    args.principle_scope = principle_scope
    try:
        count = await process_dataset(dataset_name, model_name, n, top_k, args, retrieval_data)
        logger.info(f"[SINGLE] Done: {count} cases")
        return 0
    except Exception as e:
        logger.error(f"[SINGLE] Failed: {e}")
        return 1


# Set to empty list [] to run all, or specify subset to filter
# RUN_DATASETS = ["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]
RUN_MODELS = ["gpt-5.2-chat", "claude-sonnet-4.5", "kimi-k2-250905", "doubao-seed-1-8-251228", "deepseek-v3-2-251201"]


async def main():
    parser = argparse.ArgumentParser(description="Judge fault injection cases")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch", help="batch or single (from env)")
    parser.add_argument("--n", type=int, nargs="+", default=[1], help="Judge times per case list")
    parser.add_argument("--use_principles", default=True, help="Use principle retrieval")
    parser.add_argument("--top_k", type=int, nargs="+", default=[5, 10], help="Top-k principles list")
    parser.add_argument("--use_rerank", default=True, help="Enable LLM rerank for principles")
    parser.add_argument("--top_e", type=int, default=20, help="Embedding retrieval count")
    parser.add_argument("--top_l", type=int, default=20, help="Keyword retrieval count")
    parser.add_argument("--parsed_dir", default="../datasets/parsed")
    parser.add_argument("--fault_dir", default="../datasets/fault_injected")
    parser.add_argument("--principlebank_dir", default="../results/train_bank")
    parser.add_argument("--output_dir", default="../results/principle_judge")
    # "veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"
    parser.add_argument(
        "--datasets", nargs="+", default=["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]
    )
    parser.add_argument("--global_level", default=True)
    parser.add_argument("--agent_level", default=True)
    parser.add_argument("--principle_scope", choices=["global", "agent", "both"], default="both",
                        help="Which principles to use: global/agent/both")
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_model", default=None)
    parser.add_argument("--llm_max_concurrency", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()

    # Single mode: run from environment variables
    if args.mode == "single":
        exit_code = await run_single_task(args)
        return exit_code

    # Get model list
    if args.openai_model:
        model_list = [args.openai_model]
    elif RUN_MODELS:
        model_list = RUN_MODELS
    else:
        model_list = [os.getenv("OPENAI_MODEL")]
    if not model_list or not model_list[0]:
        logger.error("No model specified: set --openai_model or OPENAI_MODEL env")
        return
    # Filter datasets
    datasets = [d for d in args.datasets]
    if not datasets:
        logger.error(f"No datasets to process after filtering")
        return

    logger.info(f"Main start: models={model_list}, datasets={datasets}")

    for model_name in model_list:
        logger.info(f"{'='*50}\n[MODEL] {model_name}\n{'='*50}")
        # Phase 1: Retrieve all cases (once per dataset per model)
        retrieval_data = {}
        for dataset_name in datasets:
            try:
                retrieval_data[dataset_name] = await retrieve_all_cases(dataset_name, model_name, args)
            except Exception as e:
                logger.error(f"[RETRIEVE] {dataset_name} failed: {e}")
                retrieval_data[dataset_name] = {}
        # Phase 2: Judge with different top_k (reuse retrieval)
        for dataset_name in datasets:
            for n in args.n:
                configs = [(False, 0)] + [(True, k) for k in args.top_k]
                for use_principles, top_k in configs:
                    args.use_principles = use_principles
                    config_name = f"n{n}_topk{top_k}" if use_principles else f"n{n}"
                    logger.info(f"[CONFIG] {config_name} starting")
                    try:
                        count = await process_dataset(
                            dataset_name,
                            model_name,
                            n,
                            top_k,
                            args,
                            retrieval_data.get(dataset_name) if use_principles else None,
                        )
                        logger.info(f"[{model_name}/{config_name}/{dataset_name}] {count} cases processed")
                    except Exception as e:
                        logger.error(f"[{model_name}/{config_name}/{dataset_name}] failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
