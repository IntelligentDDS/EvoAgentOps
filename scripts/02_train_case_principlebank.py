# main.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import json
from pathlib import Path
from evoagentops.judge import Judge
from evoagentops.config import Config
from evoagentops.util import init_logger, logger, RetryableError, set_call_context
from evoagentops.casebank import PrincipleBank
from dotenv import load_dotenv


class Store:
    """Incremental save with resume support for both global and agent level"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._raw_cache, self._done_global, self._done_agent, self._failed = self._load_progress()
        logger.info(
            f"Store loaded: dir={output_dir}, raw={len(self._raw_cache)}, done_global={len(self._done_global)}, done_agent={len(self._done_agent)}, failed={len(self._failed)}"
        )

    async def _mark_failed(self, case_id: str):
        """Mark case as permanently failed. Async-safe with lock."""
        if case_id in self._failed:
            return
        async with self._lock:
            if case_id in self._failed:
                return
            self._failed.add(case_id)
            with open(self.output_dir / "casebank_raw.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"case_id": case_id, "_failed": True}, ensure_ascii=False) + "\n")
        logger.info(f"Marked failed: {case_id}")

    def _load_progress(self) -> tuple:
        """Load progress from files"""
        raw_cache = {}  # case_id -> data
        done_global = {}
        done_agent = {}
        failed = set()  # Permanently failed cases

        raw_file = self.output_dir / "casebank_raw.jsonl"
        if raw_file.exists():
            with open(raw_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        case_id = data.get("case_id")
                        if not case_id:
                            continue
                        if data.get("_failed"):
                            failed.add(case_id)
                            continue
                        raw_cache[case_id] = data  # last wins
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skip malformed line in casebank_raw: {e}")
                        continue

        final_file = self.output_dir / "casebank.jsonl"
        if final_file.exists():
            with open(final_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        gc = data.get("global_case", {})
                        case_id = gc.get("case_id", "")
                        if not case_id:
                            continue
                        if "judge_result" in gc:
                            done_global[case_id] = data
                        # Check if agent_principle exists (even empty means done)
                        if "agent_principle" in data:
                            done_agent[case_id] = data
                    except:
                        pass

        return raw_cache, done_global, done_agent, failed

    def is_raw_done(self, case_id: str, global_level: bool, agent_level: bool) -> bool:
        """Check if raw judge already done"""
        if case_id in self._failed:
            return True  # Skip permanently failed
        data = self._raw_cache.get(case_id)
        if not data:
            return False
        if global_level and not data.get("global_case", {}).get("judge_result"):
            return False
        if agent_level and not data.get("agent_cases"):
            return False
        return True

    def save_raw(self, case_id: str, judge_output: dict, task, steps: list, global_level: bool, agent_level: bool):
        """Save judge result to casebank_raw.jsonl"""
        global_result = judge_output.get("global_result", {})
        agent_results = judge_output.get("agent_results", [])

        raw = {
            "case_id": case_id,
            "global_case": (
                {
                    "case_id": case_id,
                    "task": task,
                    "start_step": 1,
                    "end_step": len(steps),
                    "statement_action": global_result.get("statement_action", []),
                    "success": global_result.get("is_success", False),
                    "judge_result": global_result.get("judge_result", []),
                    "fault_sorted": global_result.get("fault_sorted", []),
                }
                if global_level
                else {}
            ),
            "agent_cases": (
                [
                    {
                        "case_id": f"{case_id}_{ar['agent_name']}_{ar['start_step']}",
                        "task": task,
                        "start_step": ar["start_step"],
                        "end_step": ar["end_step"],
                        "agent_name": ar["agent_name"],
                        "statement_action": ar.get("statement_action", []),
                        "success": ar.get("is_success", False),
                        "judge_result": ar.get("judge_result", []),
                        "fault_sorted": ar.get("fault_sorted", []),
                    }
                    for ar in agent_results
                ]
                if agent_level
                else []
            ),
            "global_level": global_level,
            "agent_level": agent_level,
        }

        self._raw_cache[case_id] = raw
        with open(self.output_dir / "casebank_raw.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
        logger.info(f"Saved raw: {case_id} (global={global_level}, agents={len(agent_results) if agent_level else 0})")

    def load_raw_result(self, case_id: str) -> dict:
        """Load raw judge result for a case"""
        data = self._raw_cache.get(case_id, {})
        gc = data.get("global_case", {})
        global_result = (
            {
                "statement_action": gc.get("statement_action", []),
                "judge_result": gc.get("judge_result", []),
                "fault_sorted": gc.get("fault_sorted", []),
                "is_success": gc.get("success", False),
            }
            if gc and gc.get("judge_result")
            else None
        )
        agent_results = {}
        for ac in data.get("agent_cases", []):
            if not ac.get("judge_result"):
                continue
            key = f"{ac['agent_name']}_{ac['start_step']}"
            agent_results[key] = {
                "agent_name": ac["agent_name"],
                "start_step": ac["start_step"],
                "end_step": ac["end_step"],
                "statement_action": ac.get("statement_action", []),
                "judge_result": ac.get("judge_result", []),
                "fault_sorted": ac.get("fault_sorted", []),
                "is_success": ac.get("success", False),
            }
        return {"global_result": global_result, "agent_results": agent_results}

    async def save_final(
        self,
        case_id: str,
        global_result: dict,
        agent_results: list,
        global_principle: dict,
        agent_principle: dict,
        task,
        steps: list,
        global_level: bool,
        agent_level: bool,
    ):
        """Save final result with merge support"""
        async with self._lock:
            final_file = self.output_dir / "casebank.jsonl"

            # Load existing data for this case
            existing = self._done_global.get(case_id) or self._done_agent.get(case_id) or {}
            existing_gc = existing.get("global_case", {})
            existing_acs = {f"{ac['agent_name']}_{ac['start_step']}": ac for ac in existing.get("agent_cases", [])}

            # Build new global_case
            if global_level:
                new_gc = {
                    "case_id": case_id,
                    "task": task,
                    "start_step": 1,
                    "end_step": len(steps),
                    "statement_action": global_result.get("statement_action", []),
                    "success": global_result.get("is_success", False),
                    "judge_result": global_result.get("judge_result", []),
                    "fault_sorted": global_result.get("fault_sorted", []),
                }
            else:
                # Keep existing or create minimal placeholder
                new_gc = existing_gc if "judge_result" in existing_gc else {"case_id": case_id, "task": task}

            # Build new agent_cases (merge)
            if agent_level:
                for aa in agent_results:
                    key = f"{aa['agent_name']}_{aa['start_step']}"
                    existing_acs[key] = {
                        "case_id": f"{case_id}_{key}",
                        "task": task,
                        "start_step": aa["start_step"],
                        "end_step": aa["end_step"],
                        "agent_name": aa["agent_name"],
                        "statement_action": aa.get("statement_action", []),
                        "success": aa.get("is_success", False),
                        "judge_result": aa.get("judge_result", []),
                        "fault_sorted": aa.get("fault_sorted", []),
                    }
            final = {
                "case_id": case_id,
                "global_case": new_gc,
                "agent_cases": sorted(existing_acs.values(), key=lambda x: x["start_step"]),
                "global_principle": global_principle if global_level else {},
                "agent_principle": agent_principle if agent_level else {},  # Case-level unified
            }

            # Atomic write: write to temp file, then rename
            import tempfile

            lines = []
            if final_file.exists():
                with open(final_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            if json.loads(line).get("global_case", {}).get("case_id") != case_id:
                                lines.append(line)
                        except:
                            lines.append(line)
            lines.append(json.dumps(final, ensure_ascii=False) + "\n")
            # Write to temp then atomic rename
            tmp_fd, tmp_path = tempfile.mkstemp(dir=self.output_dir, suffix=".tmp")
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    f.writelines(lines)
                os.replace(tmp_path, final_file)  # Atomic on POSIX
            except:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise

            # Update done tracking
            if global_level:
                self._done_global[case_id] = final
            if agent_level:
                self._done_agent[case_id] = final

            logger.info(f"Saved final: {case_id} (global={global_level}, agents={len(existing_acs)})")


async def judge_case(
    case_id: str,
    agent_steps: list,
    agent_dependency: dict,
    agent_settings: dict,
    task_status: str,
    task: str,
    store: Store,
    config: Config,
    global_level: bool = True,
    agent_level: bool = True,
) -> bool:
    """Run judge once and save result"""
    if store.is_raw_done(case_id, global_level, agent_level):
        skip_reason = "failed" if case_id in store._failed else "done"
        logger.info(f"Judge skip ({skip_reason}): {case_id}")
        return True
    logger.info(f"Judge start: {case_id}, global={global_level}, agent={agent_level}")

    judge = Judge(agent_steps, agent_dependency, agent_settings, config)
    set_call_context(case_id=case_id, stage="judge")
    try:
        result = await judge.judge_once(
            task_compare_withlabel=task_status, global_level=global_level, agent_level=agent_level
        )
    except RetryableError as e:
        logger.warning(f"Judge retryable (will retry): {case_id}, err={e}")
        return False
    except Exception as e:
        # Permanent failure - mark to skip on restart
        logger.error(f"Judge failed permanently: {case_id}, err={e}")
        await store._mark_failed(case_id)
        return True
    has_global = bool(result.get("global_result"))
    has_agent = bool(result.get("agent_results"))
    if has_global or has_agent:
        store.save_raw(case_id, result, task, agent_steps, global_level and has_global, agent_level and has_agent)
        g_faults = len(result.get("global_result", {}).get("fault_sorted", []))
        a_faults = sum(len(ar.get("fault_sorted", [])) for ar in result.get("agent_results", []))
        logger.info(f"Judge done: {case_id}, faults=(global={g_faults}, agent={a_faults})")
    else:
        logger.warning(f"Judge {case_id}: empty result")
    return True


async def extract_case(
    case_id: str,
    agent_steps: list,
    agent_dependency: dict,
    agent_settings: dict,
    task_status: str,
    task: str,
    store: Store,
    config: Config,
    global_level: bool = True,
    agent_level: bool = True,
):
    """Extract principles from existing raw results (with resume support)"""
    # Skip permanently failed cases
    if case_id in store._failed:
        logger.info(f"Extract skip (failed): {case_id}")
        return
    # Check if raw judge exists
    if not store.is_raw_done(case_id, global_level, agent_level):
        logger.warning(f"Extract skip (no raw): {case_id}")
        return
    do_global = global_level and case_id not in store._done_global
    do_agent = agent_level and case_id not in store._done_agent

    if not do_global and not do_agent:
        logger.info(f"Extract skip (already done): {case_id}")
        return

    raw = store.load_raw_result(case_id)
    global_result, agent_results = raw["global_result"], raw["agent_results"]

    if do_global and not global_result:
        logger.warning(f"No valid global results: {case_id}, skipping global")
        do_global = False
    if do_agent and not agent_results:
        logger.warning(f"No valid agent results: {case_id}, skipping agent")
        do_agent = False

    # If nothing to do, still mark as done to avoid retry
    if not do_global and not do_agent:
        await store.save_final(
            case_id,
            {},
            [],
            {"execute_principle": [], "judge_principle": []},
            {"execute_principle": [], "judge_principle": []},
            task,
            agent_steps,
            global_level,
            agent_level,
        )
        logger.warning(f"Extract marked done (no valid results): {case_id}")
        return

    if not agent_steps:
        logger.warning(f"Extract skip (empty steps): {case_id}")
        return
    judge = Judge(agent_steps, agent_dependency, agent_settings, config)
    set_call_context(case_id=case_id, stage="extract")
    logger.info(f"Extract start: {case_id}, global={do_global}, agent={do_agent}({len(agent_results)} segs)")

    # Build input for extract_principle (wrap single result in list for API compatibility)
    global_judge_results = [global_result] if do_global and global_result else []
    agent_judge_results = {k: [v] for k, v in agent_results.items()} if do_agent else {}

    try:
        principles = await judge.extract_principle(
            task_compare_withlabel=task_status,
            global_level=do_global,
            agent_level=do_agent,
            global_judge_results=global_judge_results,
            agent_judge_results=agent_judge_results,
        )
    except RetryableError:
        logger.warning(f"Extract retryable (will retry on restart): {case_id}")
        return  # Don't save - will retry on restart
    except Exception as e:
        logger.error(f"Extract principle failed: {case_id}, {e}")
        # Save empty to mark as done (prevent infinite retry on permanent failure)
        principles = {
            "global_principle": {"execute_principle": [], "judge_principle": []},
            "agent_principle": {"execute_principle": [], "judge_principle": []},
        }

    await store.save_final(
        case_id,
        global_result or {},
        list(agent_results.values()),
        principles.get("global_principle", {"execute_principle": [], "judge_principle": []}),
        principles.get("agent_principle", {"execute_principle": [], "judge_principle": []}),
        task,
        agent_steps,
        do_global,
        do_agent,
    )
    gp = principles.get("global_principle", {})
    ap = principles.get("agent_principle", {})
    logger.info(
        f"Extract done: {case_id}, global_exec={len(gp.get('execute_principle', []))}, global_judge={len(gp.get('judge_principle', []))}, agent_exec={len(ap.get('execute_principle', []))}, agent_judge={len(ap.get('judge_principle', []))}"
    )


async def process_split(
    split_dir: Path,
    agent_dependency: dict,
    agent_settings: dict,
    store: Store,
    config: Config,
    global_level: bool,
    agent_level: bool,
):
    """Process all cases in a split (train/test)"""
    # Collect all cases
    all_cases = []
    # "success", "fail", "unknown"
    for status in ["success", "fail", "unknown"]:
        status_dir = split_dir / status
        if not status_dir.exists():
            continue
        for case_dir in sorted(status_dir.iterdir()):
            if case_dir.is_dir():
                all_cases.append((case_dir, status))

    logger.info(f"Split start: {split_dir}, cases={len(all_cases)}, global={global_level}, agent={agent_level}")

    count = 0
    # Sequential processing - one case at a time (each case has internal concurrency)
    for i, (case_dir, task_status) in enumerate(all_cases, 1):
        case_id = case_dir.name
        try:
            with open(case_dir / "agent_steps.json", encoding="utf-8") as f:
                agent_steps = json.load(f)
            task = agent_steps[0]["agent"]["input"]

        except Exception as e:
            logger.error(f"Case load failed: {case_id}, err={e}")
            continue

        await judge_case(
            case_id,
            agent_steps,
            agent_dependency,
            agent_settings,
            task_status,
            task,
            store,
            config,
            global_level,
            agent_level,
        )
        await extract_case(
            case_id,
            agent_steps,
            agent_dependency,
            agent_settings,
            task_status,
            task,
            store,
            config,
            global_level,
            agent_level,
        )
        count += 1
        logger.info(f"Case done [{i}/{len(all_cases)}]: {case_id}")
    logger.info(f"Split done: {count}/{len(all_cases)} cases processed")
    return count


async def process_dataset(
    dataset_name: str,
    parsed_dir: Path,
    output_dir: Path,
    args,
):
    """Process a single dataset (can be run in parallel)"""
    dataset_dir = parsed_dir / dataset_name
    dataset_output = output_dir / dataset_name
    dataset_output.mkdir(parents=True, exist_ok=True)

    config = Config(output_dir=str(dataset_output))
    for attr in ["openai_base_url", "openai_api_key", "openai_model", "llm_max_concurrency"]:
        if (val := getattr(args, attr, None)) is not None:
            setattr(config, attr, val)

    init_logger(f"{dataset_output}/run.log")
    logger.info(
        f"===== Dataset: {dataset_name} | global={args.global_level}, agent={args.agent_level}, model={args.openai_model} ====="
    )

    with open(dataset_dir / "agent_dependency.json", encoding="utf-8") as f:
        agent_dependency = json.load(f)
    with open(dataset_dir / "agent_settings.json", encoding="utf-8") as f:
        agent_settings = json.load(f)

    store = Store(str(dataset_output))
    set_call_context(dataset=dataset_name)
    split_dir = dataset_dir / "train"
    if not split_dir.exists():
        return dataset_name, 0

    count = await process_split(
        split_dir, agent_dependency, agent_settings, store, config, args.global_level, args.agent_level
    )
    logger.info(f"{split_dir}: {count} cases processed")

    # Build PrincipleBank after all done
    casebank_file = dataset_output / "casebank.jsonl"
    if casebank_file.exists():
        logger.info(f"PrincipleBank sync start: {dataset_name}, casebank={casebank_file}")
        pb = PrincipleBank(str(dataset_output), str(casebank_file), config)
        await pb.sync_bank()
        logger.info(f"PrincipleBank built: {dataset_name}")
    else:
        logger.warning(f"PrincipleBank skip (no casebank): {dataset_name}")
    return dataset_name, count


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_dir", default="../datasets/parsed")
    parser.add_argument("--output_dir", default="../results/train_bank")
    # "veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"
    parser.add_argument("--datasets", nargs="+", default=["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"])
    parser.add_argument("--global_level", type=bool, default=True)
    parser.add_argument("--agent_level", type=bool, default=True)
    # LLM config
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_model", default=None)
    parser.add_argument("--llm_max_concurrency", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    logger.info(f"Main start: datasets={args.datasets}, parsed_dir={args.parsed_dir}, output_dir={args.output_dir}")

    model_name = args.openai_model or os.getenv("OPENAI_MODEL", "default")
    parsed_dir, output_dir = Path(args.parsed_dir), Path(args.output_dir) / model_name
    # Process datasets sequentially
    for dataset_name in args.datasets:
        try:
            _, count = await process_dataset(dataset_name, parsed_dir, output_dir, args)
            logger.info(f"Dataset {dataset_name}: {count} cases processed")
        except Exception as e:
            logger.error(f"Dataset {dataset_name} failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
