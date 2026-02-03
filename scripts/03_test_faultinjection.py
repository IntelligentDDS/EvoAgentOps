# main.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import asyncio
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
from dotenv import load_dotenv
from evoagentops.config import Config
from evoagentops.fault_injection import FaultInjector, FaultType
from evoagentops.util import init_logger, logger
import shutil
import copy

FJ_SEP = "__fj__"  # Separator for folder name: {case_id}__fj__{fault_type}__fj__{id}


@dataclass
class InjectionPlan:
    """Injection allocation plan"""

    total_count: int
    fault_types: List[str]
    cases: List[Path]
    # fault_type -> case_path -> list of injection ids
    allocation: Dict[str, Dict[str, List[int]]] = field(default_factory=dict)
    # Already completed injections: (case_id, fault_type, id)
    completed: Set[Tuple[str, str, int]] = field(default_factory=set)

    def calculate_allocation(self):
        """Calculate stratified sampling allocation"""
        n_faults = len(self.fault_types)
        n_cases = len(self.cases)

        if n_faults == 0 or n_cases == 0:
            logger.error("No fault types or cases available")
            return

        # Step 1: Distribute total count across fault types (stratified by fault type)
        fault_quotas = {}
        base_per_fault = self.total_count // n_faults
        remainder = self.total_count % n_faults

        for idx, fault in enumerate(self.fault_types):
            fault_quotas[fault] = base_per_fault + (1 if idx < remainder else 0)

        # Step 2: For each fault type, distribute across cases (round-robin)
        self.allocation = {fault: defaultdict(list) for fault in self.fault_types}

        for fault_idx, fault in enumerate(self.fault_types):
            quota = fault_quotas[fault]
            start_case_idx = fault_idx % n_cases  # Round-robin start

            for i in range(quota):
                case_idx = (start_case_idx + i) % n_cases
                case_path = self.cases[case_idx]
                case_id = case_path.name

                # Determine local injection id for this (case, fault) pair
                existing_ids = self.allocation[fault][case_id]
                local_id = len(existing_ids)

                self.allocation[fault][case_id].append(local_id)

        # Log allocation summary
        logger.info(f"Allocation plan: {self.total_count} injections across {n_faults} fault types and {n_cases} cases")
        for fault in self.fault_types:
            fault_total = sum(len(ids) for ids in self.allocation[fault].values())
            logger.info(f"  {fault}: {fault_total} injections")

    def get_pending_tasks(self) -> List[Tuple[str, Path, int]]:
        """Get list of pending injection tasks: (fault_type, case_path, id)"""
        pending = []
        case_map = {c.name: c for c in self.cases}

        for fault in self.fault_types:
            for case_id, ids in self.allocation[fault].items():
                case_path = case_map.get(case_id)
                if not case_path:
                    continue
                for local_id in ids:
                    if (case_id, fault, local_id) not in self.completed:
                        pending.append((fault, case_path, local_id))

        return pending


class FaultInjectionRunner:
    """Batch fault injection runner with resume support"""

    def __init__(
        self,
        parsed_dir: str,
        output_base_dir: str,
        dataset_name: str,
        total_count: int,
        fault_types: List[str] = None,
        config: Config = None,
    ):
        self.parsed_dir = Path(parsed_dir)
        self.output_base_dir = Path(output_base_dir)
        self.dataset_name = dataset_name
        self.total_count = total_count
        self.config = config or Config()

        # Setup paths
        self.dataset_dir = self.parsed_dir / dataset_name
        self.output_dir = self.output_base_dir / dataset_name / "test" / "fail"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get fault types
        if fault_types:
            self.fault_types = fault_types
        else:
            # Get all fault types from FAULT_TYPES dict
            self.fault_types = FaultType.all_keys()

        # Load agent settings
        self.agent_dependency = self._load_json(self.dataset_dir / "agent_dependency.json")
        self.agent_settings = self._load_json(self.dataset_dir / "agent_settings.json")

        # Initialize plan
        self.plan = None

    def _load_json(self, path: Path) -> dict:
        """Load JSON file"""
        if not path.exists():
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _get_success_cases(self) -> List[Path]:
        """Get all success case directories from test/success"""
        success_dir = self.dataset_dir / "test" / "success"
        cases = [d for d in sorted(success_dir.iterdir()) if d.is_dir()]
        logger.info(f"Found {len(cases)} success cases in {success_dir}")

        # Filter: prefer step > 10, else use max step case
        def _get_steps(p: Path) -> int:
            f = p / "agent_steps.json"
            if not f.exists():
                return 0
            try:
                with open(f) as fp:
                    return len(json.load(fp))
            except:
                return 0

        case_steps = [(c, _get_steps(c)) for c in cases]
        filtered = [c for c, s in case_steps if s > 10]

        if filtered:
            logger.info(f"Selected {len(filtered)} cases with step > 10")
            return filtered
        elif case_steps:
            max_case = max(case_steps, key=lambda x: x[1])
            logger.info(f"No case with step > 10, using max step case: {max_case[0].name} (steps={max_case[1]})")
            return [max_case[0]]
        return []

    def _scan_completed(self) -> Set[Tuple[str, str, int]]:
        """Scan output directory for completed injections"""
        completed = set()

        if not self.output_dir.exists():
            return completed

        for item in self.output_dir.iterdir():
            if not item.is_dir():
                continue

            parts = item.name.split(FJ_SEP)
            if len(parts) == 3:
                case_id, fault_type, id_str = parts
                # Convert underscore back to space
                fault_type = fault_type.replace("_", " ")
                try:
                    # Validate label.json
                    label_file = item / "label.json"
                    if label_file.exists():
                        with open(label_file, encoding="utf-8") as f:
                            label = json.load(f)

                        orig = label.get("original_correct_answer", "").strip()
                        wrong = label.get("wrong_final_answer", "").strip()
                        actual_changed = orig != wrong
                        # Verify flag matches reality
                        claimed_changed = label.get("is_modify_to_final_step", False)
                        if actual_changed != claimed_changed:
                            logger.warning(f"Removing invalid (flag mismatch): {item.name}")
                            shutil.rmtree(item)
                            continue
                    else:
                        # No label.json, remove and redo
                        logger.warning(f"Removing invalid injection (no label.json): {item.name}")
                        shutil.rmtree(item)
                        continue
                    completed.add((case_id, fault_type, int(id_str)))
                except ValueError:
                    continue

        logger.info(f"Found {len(completed)} completed injections")
        return completed

    def _get_output_folder_name(self, case_id: str, fault_type: str, injection_id: int) -> str:
        # Convert space to underscore for folder name
        fault_type_safe = fault_type.replace(" ", "_")
        return f"{case_id}{FJ_SEP}{fault_type_safe}{FJ_SEP}{injection_id}"

    async def _inject_single(
        self,
        fault_type: str,
        case_path: Path,
        injection_id: int,
        max_retries: int = 100,
        parallel_per_round: int = 10,
    ) -> Tuple[str, bool, str]:
        """Inject single fault into a case"""
        case_id = case_path.name
        folder_name = self._get_output_folder_name(case_id, fault_type, injection_id)
        output_path = self.output_dir / folder_name

        # Load files once
        steps_file = case_path / "agent_steps.json"
        if not steps_file.exists():
            return folder_name, False, "agent_steps.json not found"
        try:
            with open(steps_file, encoding="utf-8") as f:
                agent_steps = json.load(f)
        except Exception as e:
            return folder_name, False, f"Load error: {e}"
        case_dependency = self._load_json(case_path / "agent_dependency.json") or self.agent_dependency
        case_settings = self._load_json(case_path / "agent_settings.json") or self.agent_settings

        # Single attempt helper
        async def _try_once(force_last: bool, require_final: bool = True):
            try:
                injector = FaultInjector(copy.deepcopy(agent_steps), case_dependency, case_settings, self.config)
                result = await injector.inject(
                    fault_type, force_last_steps=force_last, require_final_change=require_final
                )
                if result and result.get("agent_steps"):
                    label = result["label"]
                    # Verify is_modify_to_final_step matches actual change
                    orig = label.get("original_correct_answer", "").strip()
                    wrong = label.get("wrong_final_answer", "").strip()
                    actual_changed = orig != wrong
                    if label.get("is_modify_to_final_step") != actual_changed:
                        logger.warning(f"is_modify_to_final_step mismatch, reject")
                        return None
                    # Mode check: require_final requires change, !require_final requires no change
                    if require_final and not actual_changed:
                        logger.warning(f"require_final=True but final unchanged, retry")
                        return None
                    if not require_final and actual_changed:
                        logger.warning(f"require_final=False but final changed, retry")
                        return None
                    return result
            except Exception:
                pass
            return None

        # Parallel retry: round 0 normal, round 1+ force last steps
        total_rounds = (max_retries + parallel_per_round - 1) // parallel_per_round
        for round_idx in range(total_rounds):
            force_last = round_idx > 0
            if force_last:
                logger.info(f"🔄 Round {round_idx+1}: last-steps mode: {folder_name}")
            # Run parallel attempts, return first valid result
            results = await asyncio.gather(*[_try_once(force_last) for _ in range(parallel_per_round)])
            for result in results:
                if result:
                    label = result["label"]
                    output_path.mkdir(parents=True, exist_ok=True)
                    with open(output_path / "agent_steps.json", "w", encoding="utf-8") as f:
                        json.dump(result["agent_steps"], f, ensure_ascii=False, indent=2)
                    with open(output_path / "agent_dependency.json", "w", encoding="utf-8") as f:
                        json.dump(case_dependency, f, ensure_ascii=False, indent=2)
                    with open(output_path / "agent_settings.json", "w", encoding="utf-8") as f:
                        json.dump(case_settings, f, ensure_ascii=False, indent=2)
                    with open(output_path / "label.json", "w", encoding="utf-8") as f:
                        json.dump(label, f, ensure_ascii=False, indent=2)
                    logger.info(
                        f"✅ Injected: {folder_name} (root_step={label.get('root_cause_step')}, round={round_idx+1})"
                    )
                    return folder_name, True, ""
            logger.warning(f"⚠️ Round {round_idx+1}/{total_rounds} all failed: {folder_name}")
        logger.error(f"❌ Failed after {total_rounds} rounds: {folder_name}")
        return folder_name, False, "All attempts failed"

    async def run(self):
        """Run batch fault injection"""
        # Get success cases
        cases = self._get_success_cases()
        if not cases:
            logger.error("No success cases found")
            return

        # Scan completed injections
        completed = self._scan_completed()

        # Create injection plan
        self.plan = InjectionPlan(
            total_count=self.total_count,
            fault_types=self.fault_types,
            cases=cases,
            completed=completed,
        )
        self.plan.calculate_allocation()

        # Get pending tasks
        pending = self.plan.get_pending_tasks()
        logger.info(f"Pending injections: {len(pending)} (completed: {len(completed)})")

        if not pending:
            logger.info("All injections completed!")
            return

        tasks = [
            self._inject_single(fault_type, case_path, injection_id) for fault_type, case_path, injection_id in pending
        ]

        total = len(tasks)
        success_count = 0

        for coro in asyncio.as_completed(tasks):
            folder_name, success, error = await coro
            if success:
                success_count += 1
        # Summary
        fail_count = total - success_count
        logger.info(f"===== Done: success={success_count}, fail={fail_count}, total={total} =====")


async def main():
    parser = argparse.ArgumentParser(description="Batch fault injection for test success cases")
    parser.add_argument("--total", type=int, default=26, help="Total number of fault injections to perform")
    parser.add_argument("--parsed_dir", default="../datasets/parsed", help="Path to parsed datasets directory")
    parser.add_argument(
        "--output_dir", default="../datasets/fault_injected", help="Output directory for injected datasets"
    )
    # ["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"],
        help="Dataset names to process",
    )
    parser.add_argument(
        "--fault_types",
        nargs="*",
        default=None,
        help="Specific fault types to inject (default: all types)",
    )
    # LLM config
    parser.add_argument("--openai_base_url", default=None)
    parser.add_argument("--openai_api_key", default=None)
    parser.add_argument("--openai_model", default=None)
    parser.add_argument("--llm_max_concurrency", default=None)

    args = parser.parse_args()
    load_dotenv()

    for dataset_name in args.datasets:
        output_dir = Path(args.output_dir) / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        config = Config(output_dir=str(output_dir))
        for attr in ["openai_base_url", "openai_api_key", "openai_model", "llm_max_concurrency"]:
            if (val := getattr(args, attr, None)) is not None:
                setattr(config, attr, val)

        init_logger(str(output_dir / "run.log"))
        logger.info(f"===== Dataset: {dataset_name} =====")
        logger.info(f"Total injections: {args.total}")

        runner = FaultInjectionRunner(
            parsed_dir=args.parsed_dir,
            output_base_dir=args.output_dir,
            dataset_name=dataset_name,
            total_count=args.total,
            fault_types=args.fault_types,
            config=config,
        )

        await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
