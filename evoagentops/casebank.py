# casebank.py
from .judge import Judge
from .config import Config
from .util import (
    logger,
    acall_embedding,
    acall_embedding_batch,
    acall_llm,
    call_embedding,
    RetryableError,
    set_call_context,
)
import os
import json
import asyncio
from pathlib import Path
import numpy as np
from openai import OpenAI
from datetime import datetime
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
import uuid
from pydantic import BaseModel, Field
from .prompt import PRINCIPLE_DEFS, format_principle_definition, format_principle_example


class VectorDB:
    """Vector database"""

    def __init__(self, vdb_file: str, casebank_file: str, config: Config):
        self.config = config
        self.vdb_file = vdb_file
        self.casebank_file = casebank_file

    async def aget_embedding(self, text: str) -> List[float]:
        """Get text vectors"""
        return await acall_embedding(text, self.config)

    async def build(self):
        """Build vector database from casebank - concurrent processing, write one by one"""
        set_call_context(stage="vdb_build", case_id=None, idx=None)
        logger.info(f"start build vector database: {self.vdb_file}")

        # read existing vectors
        existing_vectors = set()
        if Path(self.vdb_file).exists():
            with open(self.vdb_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        existing_vectors.add(record["case_id"])
                    except:
                        pass
            logger.info(f"existing vectors: {len(existing_vectors)}")

        # collect tasks to vectorize
        tasks = []
        task_metadata = []  # record case info for each task

        with open(self.casebank_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                data = json.loads(line)

                # handle global case
                gc = data["global_case"]
                if gc["case_id"] not in existing_vectors:
                    text = gc["task"] + " " + " ".join(gc.get("statement_action", []))
                    tasks.append(self.aget_embedding(text))
                    task_metadata.append({"case_id": gc["case_id"], "type": "global"})

                # handle agent cases
                for ac in data["agent_cases"]:
                    if ac["case_id"] not in existing_vectors:
                        text = ac["task"] + " " + " ".join(ac.get("statement_action", []))
                        tasks.append(self.aget_embedding(text))
                        task_metadata.append({"case_id": ac["case_id"], "type": "agent"})

        logger.info(f"tasks to vectorize: {len(tasks)}")

        # Concurrent get embeddings with index tracking
        async def embed_with_idx(idx, coro):
            try:
                return idx, await coro, None
            except Exception as e:
                return idx, None, e

        indexed_tasks = [embed_with_idx(i, t) for i, t in enumerate(tasks)]
        processed_count = 0
        failed_count = 0
        with open(self.vdb_file, "a", encoding="utf-8") as out_f:
            for coro in asyncio.as_completed(indexed_tasks):
                idx, embedding, err = await coro
                if err:
                    failed_count += 1
                    logger.error(f"Embedding failed: idx={idx}, case={task_metadata[idx]['case_id']}, err={err}")
                    continue
                if embedding:
                    metadata = task_metadata[idx]  # Use correct index
                    record = {**metadata, "embedding": embedding}
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    processed_count += 1
        logger.info(f"VectorDB build done: success={processed_count}, failed={failed_count}, total={len(tasks)}")

    async def search(self, query_text: str, case_type: str, topk: int = 3) -> List[Dict]:
        """Search similar cases and return principles"""
        set_call_context(stage="vdb_search", case_id=None, idx=None)
        query_emb = np.array(await self.aget_embedding(query_text))
        similarities = []
        # calculate cosine similarity
        with open(self.vdb_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["type"] == case_type:
                    emb = np.array(record["embedding"])
                    sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-9)
                    similarities.append((sim, record["case_id"]))
        # sort by topk
        similarities.sort(reverse=True)
        top_cases = similarities[:topk]
        similarity_map = {case_id: sim for sim, case_id in top_cases}
        top_ids = set(similarity_map.keys())
        # extract full case info
        results = []
        with open(self.casebank_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if case_type == "global":
                    gc = data["global_case"]
                    if gc["case_id"] in top_ids:
                        results.append({"similarity": similarity_map[gc["case_id"]], **gc})
                else:
                    for ac in data["agent_cases"]:
                        if ac["case_id"] in top_ids:
                            results.append({"similarity": similarity_map[ac["case_id"]], **ac})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"search completed: type={case_type}, topk={len(results)}, query='{query_text[:30]}...'")
        return results

    def add(self, case_id: str, case_type: str, text: str):
        """Add case to vector database"""
        # Note: This is sync method, use call_embedding from util
        record = {"case_id": case_id, "type": case_type, "embedding": call_embedding(text, self.config)}
        with open(self.vdb_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def delete(self, case_id: str):
        """Delete case from vector database"""
        records = []
        with open(self.vdb_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["case_id"] != case_id:
                    records.append(record)
        with open(self.vdb_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def update(self, case_id: str, text: str):
        """Update case embedding in vector database"""
        records = []
        with open(self.vdb_file, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["case_id"] == case_id:
                    record["embedding"] = call_embedding(text, self.config)
                records.append(record)
        with open(self.vdb_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class PrincipleOperation(BaseModel):
    """Principle Operation definition"""

    # keep/modify/merge - for batch consolidation
    operation: Literal["keep", "modify", "merge"] = Field(
        description="keep: unique, no change; modify: refine wording; merge: combine 2+ overlapping"
    )
    principle_id: Optional[str] = Field(default=None, description="Target ID. Required for keep/modify.")
    title: Optional[str] = Field(default=None, description="Concise phrase: [Action] + [Scenario]. No specific values.")
    content: Optional[str] = Field(
        default=None, description="When [trigger], [action] by [method], avoiding [pitfall]."
    )
    merge_ids: Optional[List[str]] = Field(default=None, description="List of principle IDs to merge (2+ IDs required)")
    reason: Optional[str] = Field(default=None, description="Brief reason for modify/merge.")


class PrincipleOperations(BaseModel):
    operations: List[PrincipleOperation]


@dataclass
class PrincipleEntry:
    principle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    function: str = ""  # execute or judge
    type: str = ""  # global or agent
    agent_name: Optional[str] = None
    title: str = ""
    content: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_cases: List[str] = field(default_factory=list)
    source_principle_ids: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)


# ============ PrincipleBank ============
class PrincipleBank:
    def __init__(self, principlebank_dir: str, casebank_file: str, config: Config):
        self.principlebank_dir = Path(principlebank_dir)
        self.casebank_file = Path(casebank_file)
        self.config = config
        # can custom paras
        self.topk = 5
        self.threshold = 0.7  # Dynamic, computed per batch
        self.convergence_eps = 0.15  # Stop when reduction < 10%
        self.max_merge_rounds = 3

        # Use unified definitions from prompt.py
        self.principle_definitions = PRINCIPLE_DEFS

        # file paths - only keep two files
        self.principlebank_raw_file = self.principlebank_dir / "principlebank" / "principlebank_raw.jsonl"
        self.principlebank_file = self.principlebank_dir / "principlebank" / "principlebank.jsonl"
        # cache states
        self.processed_cases = set()  # casebank -> raw
        self.processed_raw_ids = set()  # raw -> merged
        self.failed_cases = set()  # Permanently failed cases

    def refresh_state(self):
        """Refresh cache states"""
        self.processed_cases.clear()
        self.processed_raw_ids.clear()
        self.failed_cases.clear()

        # Read raw file to get processed cases
        if self.principlebank_raw_file.exists():
            with open(self.principlebank_raw_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        src = p.get("source_cases", [])
                        if p.get("_failed"):
                            self.failed_cases.update(src)
                        else:
                            self.processed_cases.update(src)
                    except:
                        pass
        # Read merged file to get processed raw principle_ids
        if self.principlebank_file.exists():
            with open(self.principlebank_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        p = json.loads(line)
                        self.processed_raw_ids.update(p.get("source_principles", []))
                    except:
                        pass
        # Read merge progress file for interrupted merge recovery
        merge_progress_file = self.principlebank_dir / "principlebank" / "_merge_progress.json"
        if merge_progress_file.exists():
            try:
                with open(merge_progress_file, "r", encoding="utf-8") as f:
                    progress = json.load(f)
                    self.processed_raw_ids.update(progress.get("processed_raw_ids", []))
                logger.info(f"Recovered merge progress: {len(progress.get('processed_raw_ids', []))} raw_ids")
            except Exception as e:
                logger.warning(f"Failed to read merge progress: {e}")
        logger.info(
            f"refresh_state: processed={len(self.processed_cases)}, raw_ids={len(self.processed_raw_ids)}, failed={len(self.failed_cases)}"
        )

    def _get_embedding_text(self, principle: Dict) -> str:
        return f"# Title:\n{principle['title']}\n# Content:\n{principle['content']}"

    async def _extract_raw_principles(self, case: Dict) -> List[Dict]:
        """Extract raw principles from a single case"""
        gc_id = case["global_case"]["case_id"]
        set_call_context(case_id=gc_id, stage="extract_raw")
        logger.info(f"_extract_raw start: {gc_id}")
        # Collect all principle data first (without embedding)
        principle_data = []  # [(function, type, agent_name, title, content, source_case)]
        for function in ["execute", "judge"]:
            # Global principles (now at top level)
            global_principle = case.get("global_principle", {})
            func_principles = global_principle.get(f"{function}_principle", [])
            for p in func_principles:
                if not p.get("title") or not p.get("content"):
                    continue
                principle_data.append((function, "global", None, p["title"], p["content"], gc_id))
            # Agent-level unified principles (case["agent_principle"])
            agent_principle = case.get("agent_principle", {})
            agent_func_principles = agent_principle.get(f"{function}_principle", [])
            for p in agent_func_principles:
                if not p.get("title") or not p.get("content"):
                    continue
                principle_data.append((function, "agent", p.get("agent_name"), p["title"], p["content"], gc_id))

        if not principle_data:
            logger.info(f"_extract_raw: case={gc_id}, extracted=0 principles")
            return []

        # Batch embedding
        texts = [f"# Title:\n{d[3]}\n# Content:\n{d[4]}" for d in principle_data]
        try:
            embeddings = await acall_embedding_batch(texts, self.config)
        except RetryableError:
            raise
        except Exception as e:
            logger.error(f"Batch embedding failed: case={gc_id}, texts={len(texts)}, err={e}")
            raise
        # Build principles with embeddings
        principles = []
        now = datetime.now().isoformat()
        for i, (func, ptype, agent_name, title, content, source_case) in enumerate(principle_data):
            pid = hashlib.md5(f"{func}_{ptype}_{title}_{content[:50]}".encode()).hexdigest()
            principles.append(
                {
                    "principle_id": pid,
                    "function": func,
                    "type": ptype,
                    "agent_name": agent_name,
                    "title": title,
                    "content": content,
                    "created_at": now,
                    "updated_at": now,
                    "source_cases": [source_case],
                    "source_principles": [pid],
                    "embedding": embeddings[i],
                }
            )

        exec_n = sum(1 for p in principles if p["function"] == "execute")
        judge_n = sum(1 for p in principles if p["function"] == "judge")
        logger.info(f"Extract raw: {gc_id}, total={len(principles)} (exec={exec_n}, judge={judge_n})")
        return principles

    def _get_scope_key(self, p: Dict) -> str:
        """Get scope key for fusion: function + type + agent_name"""
        return f"{p['function']}_{p['type']}_{p.get('agent_name') or ''}"

    def _cluster_by_similarity(self, principles: List[Dict], threshold: float = None) -> List[List[Dict]]:
        """Cluster principles using Union-Find with adaptive threshold"""
        n = len(principles)
        if n <= 1:
            return [principles] if principles else []
        valid = [p for p in principles if p.get("embedding")]
        invalid = [p for p in principles if not p.get("embedding")]
        if invalid:
            logger.warning(f"Cluster: {len(invalid)}/{len(principles)} empty embeddings, as singletons")
        if len(valid) <= 1:
            return [[p] for p in valid] + [[p] for p in invalid]
        n = len(valid)

        # Compute all pairwise similarities first
        embeddings = [np.array(p["embedding"]) for p in valid]
        sim_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9
                )
                sim_pairs.append((sim, i, j))

        # Adaptive threshold: use median + 0.5*IQR, clamped to [0.4, 0.85]
        MIN_SAMPLES_FOR_ADAPTIVE = 10  # Need enough samples for stable statistics
        FALLBACK_THRESHOLD = 0.6
        if threshold is None and sim_pairs and len(sim_pairs) >= MIN_SAMPLES_FOR_ADAPTIVE:
            sims = sorted([s[0] for s in sim_pairs])
            q1, median, q3 = np.percentile(sims, [25, 50, 75])
            iqr = q3 - q1
            # Stability check: skip adaptive if distribution is too skewed
            skewness = (median - q1) / (iqr + 1e-9) if iqr > 0.05 else 0
            if abs(skewness) > 2:  # Highly skewed
                threshold = FALLBACK_THRESHOLD
                logger.info(f"Adaptive skip (skewed): skew={skewness:.2f}, using fallback={threshold}")
            else:
                threshold = float(np.clip(median + 0.5 * iqr, 0.4, 0.85))
                logger.debug(f"Adaptive threshold: median={median:.3f}, iqr={iqr:.3f}, threshold={threshold:.3f}")
        elif threshold is None:
            threshold = FALLBACK_THRESHOLD
            logger.debug(f"Using fallback threshold={threshold} (samples={len(sim_pairs) if sim_pairs else 0})")

        # Union-Find with path compression and cached members
        parent = list(range(n))
        rank = [0] * n
        members = {i: {i} for i in range(n)}  # Cache: root -> member set

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def can_merge(x, y, sim_matrix):
            """Complete-linkage: all pairs must exceed threshold"""
            px, py = find(x), find(y)
            if px == py:
                return True
            # Safety check: ensure both roots still have members
            if px not in members or py not in members:
                return False
            for mx in members[px]:
                for my in members[py]:
                    key = (min(mx, my), max(mx, my))
                    if sim_matrix.get(key, 0) <= threshold:
                        return False
            return True

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Merge smaller into larger (union by rank)
                if rank[px] < rank[py]:
                    parent[px] = py
                    members[py] |= members.pop(px)
                elif rank[px] > rank[py]:
                    parent[py] = px
                    members[px] |= members.pop(py)
                else:
                    parent[py] = px
                    members[px] |= members.pop(py)
                    rank[px] += 1

        # Build similarity matrix for complete-linkage check
        sim_matrix = {(min(i, j), max(i, j)): sim for sim, i, j in sim_pairs}

        # Union based on threshold
        for sim, i, j in sorted(sim_pairs, reverse=True):
            if sim > threshold and can_merge(i, j, sim_matrix):
                union(i, j)

        # Group by root
        groups: Dict[int, List[Dict]] = {}
        for root, member_set in members.items():
            groups[root] = [valid[i] for i in member_set]

        logger.debug(f"Cluster: n={n}, threshold={threshold:.3f}, groups={len(groups)}")
        # Add invalid principles as singletons (will be kept as-is)
        result = list(groups.values())
        for p in invalid:
            result.append([p])
        return result

    async def _merge_principles_batch(
        self, new_principles: List[Dict], existing: List[Dict], on_scope_done=None
    ) -> List[Dict]:
        """Batch merge using Union-Find clustering + concurrent LLM calls"""
        if not new_principles:
            return existing

        # Group new principles by scope
        scope_groups: Dict[str, List[Dict]] = {}
        for p in new_principles:
            scope = self._get_scope_key(p)
            scope_groups.setdefault(scope, []).append(p)

        result = list(existing)
        total_processed = 0

        max_batch_size = 15  # Max principles per LLM call
        has_retryable = False
        for scope, new_in_scope in scope_groups.items():
            same_scope = [p for p in result if self._get_scope_key(p) == scope]
            other_scope = [p for p in result if self._get_scope_key(p) != scope]
            # Iterative merge until convergence (max rounds)
            current = new_in_scope + same_scope
            prev_len = len(current)
            for round_idx in range(self.max_merge_rounds):
                if len(current) <= 1:
                    break
                len_before = len(current)
                # Cluster by similarity
                clusters = self._cluster_by_similarity(current)
                # Skip if no clustering happened (all singletons)
                if len(clusters) == len(current):
                    logger.info(f"Merge {scope} r{round_idx+1}: no clusters formed, stopping")
                    break
                # Split large clusters to fit LLM context
                batches = []
                for c in clusters:
                    for i in range(0, len(c), max_batch_size):
                        batches.append(c[i : i + max_batch_size])
                # Concurrent batch processing with partial success handling
                batch_results = await asyncio.gather(
                    *[self._process_merge_batch(b, scope) for b in batches], return_exceptions=True
                )
                merged = []
                for i, r in enumerate(batch_results):
                    if isinstance(r, RetryableError):
                        has_retryable = True
                        # Keep original principles for retryable batches
                        merged.extend(batches[i])
                        logger.warning(
                            f"Merge batch {i+1}/{len(batches)} retryable, keeping {len(batches[i])} principles"
                        )
                    elif isinstance(r, Exception):
                        # Keep original for other errors
                        merged.extend(batches[i])
                        logger.error(
                            f"Merge batch {i+1}/{len(batches)} failed: {r}, keeping {len(batches[i])} principles"
                        )
                    else:
                        merged.extend(r)
                # Delta convergence check
                reduction = (len_before - len(merged)) / len_before if len_before > 0 else 0
                logger.info(
                    f"Merge {scope} r{round_idx+1}: {len_before}->{len(merged)} (reduction={reduction:.1%}, threshold={self.convergence_eps:.1%})"
                )
                # Stop if: (1) reduction below threshold OR (2) no change for 2 consecutive rounds
                if reduction < self.convergence_eps or (round_idx > 0 and len(merged) == prev_len):
                    logger.info(f"Merge {scope}: converged at round {round_idx+1}")
                    current = merged
                    break
                prev_len = len(merged)
                current = merged
            total_processed += len(new_in_scope)

            merged_scope = current
            result = other_scope + merged_scope
            if on_scope_done:
                on_scope_done(result)
        logger.info(
            f"_merge_batch: processed {total_processed} principles, result={len(result)}, retryable={has_retryable}"
        )
        return result

    async def _process_merge_batch(self, principles: List[Dict], scope: str) -> List[Dict]:
        """Process a batch of principles with LLM to decide merge operations"""
        if len(principles) <= 1:
            return principles

        # Build short ID mapping
        id_prefix = "P"  # Configurable prefix for LLM short IDs
        id_to_short = {p["principle_id"]: f"{id_prefix}{i}" for i, p in enumerate(principles)}
        short_to_id = {v: k for k, v in id_to_short.items()}

        def simplify(p):
            return {"id": id_to_short[p["principle_id"]], "title": p["title"], "content": p["content"]}

        # Extract function type from scope for targeted prompt
        func_type = scope.split("_")[0]  # execute or judge
        pdef = PRINCIPLE_DEFS.get(func_type, PRINCIPLE_DEFS["execute"])
        boundary_str = (
            f"\n  MUST include: {', '.join(pdef['includes'])}.\n  MUST NOT include: {', '.join(pdef['excludes'])}."
        )
        example_str = format_principle_example(func_type)
        prompt = f"""<system>You are a principle consolidation expert.</system>
<context>
Scope: {scope}
{pdef['name']}: {pdef['purpose']}.
Covers: {pdef['covers']}{boundary_str}
Input: {len(principles)} principles
</context>
<instructions>
Goal: Remove redundancy while preserving ALL unique knowledge.
Success criteria:
1) No information loss - preserve every unique insight
2) No over-merging - only combine >70% semantic overlap
3) Output valid JSON
Steps:
1) For each principle, identify trigger + action + scenario
2) Find pairs with same trigger AND similar action
3) Decide: keep (unique), modify (clarify), merge (combine overlapping)
</instructions>

<output_format>
Return ONLY valid JSON:
{{"operations": [
  {{"operation": "keep", "principle_id": "P0"}},
  {{"operation": "modify", "principle_id": "P0", "reason": "...", "title": "...", "content": "..."}},
  {{"operation": "merge", "merge_ids": ["P0", "P1"], "reason": "...", "title": "...", "content": "..."}}
]}}
</output_format>

<format>
title: {pdef.get('title_pattern', '[Action] + [Scenario]')}
content: {pdef.get('content_pattern', pdef['format'])}
Example: {example_str}
</format>

<example>
Input: [
  {{"id":"{id_prefix}0","title":"Validate API response status","content":"Check HTTP status code before parsing response body."}},
  {{"id":"{id_prefix}1","title":"Check API return code","content":"Verify status is 2xx before processing API response."}}
]
Analysis: P0 and P1 both address HTTP status validation with same trigger (API call) and action (check status). High overlap.
Output: {{"operations":[{{"operation":"merge","merge_ids":["{id_prefix}0","{id_prefix}1"],"reason":"Both validate HTTP status before processing, same trigger and action","title":"Validate API response before processing","content":"When calling external API, verify HTTP status is 2xx before parsing response body, avoiding assumption of success without explicit status check."}}]}}
</example>

<input>
{json.dumps([simplify(p) for p in principles], ensure_ascii=False)}
</input>

Think step by step, then return ONLY valid JSON: {{"operations":[...]}}"""

        try:
            response = await acall_llm(
                [{"role": "user", "content": prompt}], self.config, output_schema=PrincipleOperations
            )
            result = json.loads(response)
            # Handle both {"operations": [...]} and direct [...] formats
            if isinstance(result, list):
                operations = result
            else:
                operations = result.get("operations", [])
            # Filter valid operations (must be dict with 'operation' key)
            operations = [op for op in operations if isinstance(op, dict) and "operation" in op]
            logger.info(
                f"_process_batch: scope={scope}, input={len(principles)}, ops={[op.get('operation') for op in operations]}"
            )
        except RetryableError:
            raise  # Propagate - will retry on restart
        except Exception as e:
            logger.error(f"_process_batch LLM failed: {e}, keeping all")
            return principles

        # Apply operations
        p_dict = {p["principle_id"]: p.copy() for p in principles}
        merged_away = set()  # IDs that were merged into others
        needs_embed = []  # Principles that need embedding update

        for op in operations:
            op_type = op.get("operation")

            if op_type == "keep":
                pass  # No action needed

            elif op_type == "modify":
                pid = short_to_id.get(op.get("principle_id"), op.get("principle_id"))
                if pid in p_dict:
                    p_dict[pid]["title"] = op.get("title") or p_dict[pid]["title"]
                    p_dict[pid]["content"] = op.get("content") or p_dict[pid]["content"]
                    p_dict[pid]["updated_at"] = datetime.now().isoformat()
                    needs_embed.append(p_dict[pid])
            elif op_type == "merge":
                merge_ids = [short_to_id.get(mid, mid) for mid in op.get("merge_ids", [])]
                valid_ids = [mid for mid in merge_ids if mid in p_dict and mid not in merged_away]
                if len(valid_ids) < 2:
                    continue

                # Collect source info from all merged principles
                source_cases = set()
                source_principles = set()
                base_p = p_dict[valid_ids[0]]
                for mid in valid_ids:
                    source_cases.update(p_dict[mid].get("source_cases", []))
                    source_principles.update(p_dict[mid].get("source_principles", []))
                    source_principles.add(mid)
                    if mid != valid_ids[0]:
                        merged_away.add(mid)

                # Create merged principle with new ID
                new_pid = hashlib.md5(f"{op['title']}{op['content'][:50]}".encode()).hexdigest()
                merged_p = {
                    "principle_id": new_pid,
                    "function": base_p["function"],
                    "type": base_p["type"],
                    "agent_name": base_p.get("agent_name"),
                    "title": op["title"],
                    "content": op["content"],
                    "created_at": base_p.get("created_at", datetime.now().isoformat()),
                    "updated_at": datetime.now().isoformat(),
                    "source_cases": list(source_cases),
                    "source_principles": list(source_principles),
                    "embedding": [],  # Will update later
                }
                # Replace old entry with new merged principle
                del p_dict[valid_ids[0]]
                p_dict[new_pid] = merged_p
                needs_embed.append(merged_p)

        # Return non-merged principles
        result_principles = [p for pid, p in p_dict.items() if pid not in merged_away]

        # Update embeddings for modified/merged principles
        if needs_embed:
            texts = [self._get_embedding_text(p) for p in needs_embed]
            try:
                embeddings = await acall_embedding_batch(texts, self.config)
                for p, emb in zip(needs_embed, embeddings):
                    p["embedding"] = emb
                logger.info(f"_process_batch: updated {len(needs_embed)} embeddings")
            except Exception as e:
                affected_ids = [p["principle_id"][:8] for p in needs_embed]
                logger.error(
                    f"_process_batch embedding failed: {e}, affected={affected_ids}, will be skipped in retrieval"
                )

        return result_principles

    async def sync_bank(self):
        """Sync casebank -> raw -> merged principlebank"""
        self.principlebank_dir.mkdir(parents=True, exist_ok=True)
        (self.principlebank_dir / "principlebank").mkdir(parents=True, exist_ok=True)
        self.refresh_state()

        # Count total and pending
        with open(self.casebank_file, "r", encoding="utf-8") as f:
            total_cases = sum(1 for _ in f)
        processed_count = len(self.processed_cases)
        logger.info(f"sync_bank Step1 start: total={total_cases}, processed={processed_count}")

        # Step 1: casebank -> principlebank_raw.jsonl (concurrent extraction + streaming write)
        pending_cases = []
        with open(self.casebank_file, "r", encoding="utf-8") as f:
            for line in f:
                case = json.loads(line)
                case_id = case["global_case"]["case_id"]
                # Skip both processed and permanently failed
                if case_id not in self.processed_cases and case_id not in self.failed_cases:
                    pending_cases.append(case)

        logger.info(f"Step1: pending={len(pending_cases)}, skip_failed={len(self.failed_cases)}")
        step1_count = 0
        failed_count = 0

        async def extract_with_case(case):
            """Extract principles with case info for tracking"""
            case_id = case["global_case"]["case_id"]
            try:
                principles = await self._extract_raw_principles(case)
                return case_id, principles, None
            except Exception as e:
                return case_id, None, e

        # Concurrent extraction + streaming write (semaphore controls actual concurrency)
        tasks = [extract_with_case(c) for c in pending_cases]
        with open(self.principlebank_raw_file, "a", encoding="utf-8") as out_f:
            for coro in asyncio.as_completed(tasks):
                case_id, principles, err = await coro
                if isinstance(err, RetryableError):
                    # Don't write marker - will retry on restart
                    logger.warning(f"Step1 retryable: case={case_id}, will retry on restart")
                elif err:
                    # Write failure marker with _failed flag
                    out_f.write(json.dumps({"source_cases": [case_id], "_failed": True}, ensure_ascii=False) + "\n")
                    out_f.flush()
                    failed_count += 1
                    logger.error(f"Step1 failed permanently: case={case_id}, err={err}")
                elif not principles:
                    out_f.write(json.dumps({"source_cases": [case_id]}, ensure_ascii=False) + "\n")
                    out_f.flush()
                    step1_count += 1
                    logger.info(f"Step1 [{step1_count}/{len(pending_cases)}]: {case_id} (empty)")
                else:
                    for p in principles:
                        out_f.write(json.dumps(p, ensure_ascii=False) + "\n")
                    out_f.flush()
                    step1_count += 1
                    # Log with execute/judge breakdown
                    exec_n = sum(1 for p in principles if p["function"] == "execute")
                    judge_n = sum(1 for p in principles if p["function"] == "judge")
                    logger.info(
                        f"Step1 [{step1_count}/{len(pending_cases)}]: {case_id}, exec={exec_n}, judge={judge_n}"
                    )
        logger.info(f"sync_bank Step1 done: success={step1_count}, failed={failed_count}")

        # Step 2: principlebank_raw -> principlebank.jsonl (merge with immediate save)
        merged = []
        if self.principlebank_file.exists():
            with open(self.principlebank_file, "r", encoding="utf-8") as f:
                merged = [json.loads(line) for line in f]

        # Collect pending raw principles (merge duplicates by principle_id)
        pending_raw_dict = {}
        raw_total = 0
        if self.principlebank_raw_file.exists():
            with open(self.principlebank_raw_file, "r", encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    pid = p.get("principle_id")
                    if not pid:  # Skip empty markers (failed/empty cases)
                        continue
                    raw_total += 1
                    if pid in self.processed_raw_ids:
                        continue
                    if pid in pending_raw_dict:
                        # Merge source_cases from duplicate
                        pending_raw_dict[pid]["source_cases"] = list(
                            set(pending_raw_dict[pid].get("source_cases", [])) | set(p.get("source_cases", []))
                        )
                    else:
                        pending_raw_dict[pid] = p
        pending_raw = list(pending_raw_dict.values())
        pending_raw_ids = set(pending_raw_dict.keys())
        logger.info(
            f"sync_bank Step2 start: existing={len(merged)}, pending={len(pending_raw)}, raw_total={raw_total}, processed_raw={len(self.processed_raw_ids)}"
        )
        if pending_raw:
            set_call_context(stage="merge", case_id=None, idx=None)
            # Track processed raw_ids for resume support
            merge_progress_file = self.principlebank_dir / "principlebank" / "_merge_progress.json"

            def save_merged(m):
                # Save merged principles
                with open(self.principlebank_file, "w", encoding="utf-8") as f:
                    for p in m:
                        f.write(json.dumps(p, ensure_ascii=False) + "\n")
                # Save progress: mark all pending_raw_ids as processed
                with open(merge_progress_file, "w", encoding="utf-8") as f:
                    json.dump({"processed_raw_ids": list(self.processed_raw_ids | pending_raw_ids)}, f)

            try:
                merged = await self._merge_principles_batch(pending_raw, merged, on_scope_done=save_merged)
            except RetryableError:
                logger.warning(f"Step2 merge retryable, will retry on restart")
                return  # Don't clean progress file - resume on restart
            except Exception as e:
                logger.error(f"Step2 merge failed: {e}")
                # Permanent failure - save current progress and continue
                save_merged(merged)
                logger.warning(f"Step2 partial save: {len(merged)} principles preserved")
                return
            # Update processed_raw_ids after successful merge
            self.processed_raw_ids.update(pending_raw_ids)
            # Clean up progress file
            if merge_progress_file.exists():
                merge_progress_file.unlink()
            logger.info(f"Step2 done: {len(pending_raw)} pending -> {len(merged)} total principles")

        logger.info(f"Sync completed: {len(merged)} principles in bank")

    def _retrieve_similar(
        self, query_embedding: np.ndarray, candidates: List[Dict], topk: int = None, threshold: float = None
    ) -> List[Dict]:
        """Retrieve similar principles by cosine similarity"""
        topk = topk or self.topk
        threshold = threshold if threshold is not None else (self.threshold or 0.5)
        similarities = []
        skipped = 0
        for c in candidates:
            if not c.get("embedding"):
                skipped += 1
                continue
            emb = np.array(c["embedding"])
            norm_q, norm_e = np.linalg.norm(query_embedding), np.linalg.norm(emb)
            if norm_q < 1e-9 or norm_e < 1e-9:
                skipped += 1
                continue
            sim = np.dot(query_embedding, emb) / (norm_q * norm_e)
            if sim > threshold:
                similarities.append((sim, c))
        similarities.sort(reverse=True, key=lambda x: x[0])
        if skipped > 0:
            logger.debug(f"_retrieve_similar: skipped {skipped}/{len(candidates)} empty embeddings")
        return [item[1] for item in similarities[:topk]]

    async def search(
        self,
        query_text: str,
        function: str,
        type_: str = None,
        agent_name: str = None,
        topk: int = 3,
        threshold: float = None,
    ) -> List[Dict]:
        """Search principles with optional filters"""
        query_emb = np.array(await acall_embedding(query_text, self.config))
        candidates = self.filter_principles(function, type_, agent_name)
        results = self._retrieve_similar(query_emb, candidates, topk, threshold or 0.5)
        logger.info(f"Search: function={function}, type={type_}, agent={agent_name}, found={len(results)}")
        return results

    def filter_principles(self, function: str = None, type_: str = None, agent_name: str = None) -> List[Dict]:
        """Filter principles by function/type/agent_name"""
        if not self.principlebank_file.exists():
            return []
        results = []
        with open(self.principlebank_file, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if function and p.get("function") != function:
                    continue
                if type_ and p.get("type") != type_:
                    continue
                if agent_name is not None and p.get("agent_name") != agent_name:
                    continue
                results.append(p)
        return results

    async def update_embeddings(self):
        """Update embeddings for existing principles (batch mode)"""
        set_call_context(stage="update_embed", case_id=None, idx=None)
        for file_path in [self.principlebank_raw_file, self.principlebank_file]:
            if not file_path.exists():
                continue

            # Load all principles
            principles = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    if p.get("principle_id"):  # Skip empty markers
                        principles.append(p)
            if not principles:
                logger.info(f"Skip empty file: {file_path.name}")
                continue
            logger.info(f"Updating embeddings for {len(principles)} principles in {file_path.name}")

            # Batch update embeddings
            texts = [self._get_embedding_text(p) for p in principles]
            try:
                embeddings = await acall_embedding_batch(texts, self.config)
                now = datetime.now().isoformat()
                for p, emb in zip(principles, embeddings):
                    p["embedding"] = emb
                    p["updated_at"] = now
            except Exception as e:
                logger.error(f"Batch embedding failed for {file_path.name}: {e}")
                continue

            # Rewrite file (preserve empty markers)
            all_records = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if not r.get("principle_id"):  # Keep empty markers
                        all_records.append(r)
            # Add updated principles
            all_records.extend(principles)
            with open(file_path, "w", encoding="utf-8") as f:
                for p in all_records:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

            logger.info(f"Saved {len(principles)} principles to {file_path.name}")
