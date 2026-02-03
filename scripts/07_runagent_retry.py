import os, sys, json, shutil, subprocess, time, argparse, threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Agent Configs ===
AGENT_CONFIGS = {
    "autogen_math": {
        "dir": "AutoGen_GSM8K",
        "prompt_map": {
            "MathSolverA": "MATHSOLVERA_SYSTEM",
            "MathSolverB": "MATHSOLVERB_SYSTEM",
            "MathSolverC": "MATHSOLVERC_SYSTEM",
            "MathSolverD": "MATHSOLVERD_SYSTEM",
        },
        "result_prefix": "RESULT:",
        "query_key": "question",
        "test_dir": "datasets/raw_split/autogen_math/test",
        "output_base": "results/test_mas/autogen_math",
        "bank_path": "results/train_bank/autogen_math",
    },
    "veadk_gaia": {
        "dir": "CogKernelPro",
        "prompt_map": {
            "ck_plan_agent": "_CK_PLAN_SYS",
            "ck_action_agent": "_CK_ACTION_SYS",
            "ck_end_agent": "_CK_END_SYS",
            "file_plan_agent": "_FILE_PLAN_SYS",
            "file_action_agent": "_FILE_ACTION_SYS",
            "file_end_agent": "_FILE_END_SYS",
            "web_plan_agent": "_WEB_PLAN_SYS",
            "web_action_agent": "_WEB_ACTION_SYS",
            "web_end_agent": "_WEB_END_SYS",
        },
        "result_prefix": "CK_RESULT:",
        "query_key": "question",
        "metadata_file": "_test/validation/metadata_filtered.jsonl",
        "test_dir": "datasets/raw_split/veadk_gaia/test",
        "output_base": "results/test_mas/veadk_gaia",
        "bank_path": "results/train_bank/veadk_gaia",
    },
    "langgraph_sql": {
        "dir": "LangGraph_Spider",
        "prompt_map": {
            "write_query": "WRITE_QUERY_SYSTEM",
            "check_query": "CHECK_QUERY_SYSTEM",
            "rewrite_query": "REWRITE_QUERY_SYSTEM",
        },
        "result_prefix": "RESULT:",
        "query_key": "question",
        "task_file": "data/test.json",
        "test_dir": "datasets/raw_split/langgraph_sql/test",
        "output_base": "results/test_mas/langgraph_sql",
        "bank_path": "results/train_bank/langgraph_sql",
    },
    "agno_rca": {
        "dir": "OpenRCA",
        "prompt_map": {
            "reasoning_agent": "reasoning_agent_system",
            "execution_agent": "execution_agent_system",
        },
        "result_prefix": "RESULT:",
        "query_key": "instruction",
        "csv_file": "dataset/Market/cloudbed-1/query.csv",
        "test_dir": "datasets/raw_split/agno_rca/test",
        "output_base": "results/test_mas/agno_rca",
        "bank_path": "results/train_bank/agno_rca",
    },
    "adk_swe": {
        "dir": "SWE",
        "prompt_map": {
            "swe_agent": "SYSTEM_PROMPT",
        },
        "result_prefix": "RESULT:",
        "query_key": "problem_statement",
        "test_dir": "datasets/raw_split/adk_swe/test",
        "output_base": "results/test_mas/adk_swe",
        "bank_path": "results/train_bank/adk_swe",
    },
}


def get_project_root():
    return Path(__file__).resolve().parent.parent


def extract_task_id(folder_name: str) -> str:
    """Extract task_id by removing timestamp suffix"""
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 14:
        return parts[0]
    return folder_name


def find_existing_workdir(workdir_root: Path, task_id: str) -> Path | None:
    """Find existing workdir for task_id"""
    if not workdir_root.exists():
        return None
    for d in workdir_root.iterdir():
        if d.is_dir() and extract_task_id(d.name) == task_id:
            return d
    return None


def is_completed(workdir: Path) -> bool:
    """Check if workdir has output.json (completed)"""
    return (workdir / "output.json").exists()


def cleanup_workdir(workdir: str, agent_name: str):
    """Remove temp files from workdir"""
    if agent_name == "adk_swe":
        p = os.path.join(workdir, "outputs")
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    elif agent_name == "langgraph_sql":
        for f in ["schema.sql", "vehicle_rent.sqlite"]:
            p = os.path.join(workdir, f)
            if os.path.isfile(p):
                os.remove(p)
    elif agent_name == "agno_rca":
        p = os.path.join(workdir, "tmp")
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)


def load_agent_tasks(agent_name: str, agent_dir: Path) -> tuple[dict, str | None]:
    """Load task definitions for an agent"""
    extra_dir = None
    if agent_name == "autogen_math":
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test", cache_dir=str(agent_dir / ".cache"))
        tasks = {
            f"gsm8k-{i:04d}": {"task_id": f"gsm8k-{i:04d}", "question": ex["question"], "answer": ex["answer"]}
            for i, ex in enumerate(ds)
        }
    elif agent_name == "veadk_gaia":
        mf = agent_dir / AGENT_CONFIGS[agent_name]["metadata_file"]
        tasks = {}
        for line in open(mf):
            if line.strip():
                item = json.loads(line)
                if tid := item.get("task_id"):
                    tasks[tid] = item
        extra_dir = str(mf.parent)
    elif agent_name == "langgraph_sql":
        tf = agent_dir / AGENT_CONFIGS[agent_name]["task_file"]
        samples = json.load(open(tf))
        tasks = {f"spider-{i:04d}": {"task_id": f"spider-{i:04d}", **s} for i, s in enumerate(samples)}
        extra_dir = str(tf.parent)
    elif agent_name == "agno_rca":
        import pandas as pd

        cf = agent_dir / AGENT_CONFIGS[agent_name]["csv_file"]
        df = pd.read_csv(cf)
        tasks = {
            f"{row['task_index']}-{i:04d}": {
                "task_id": f"{row['task_index']}-{i:04d}",
                "instruction": row["instruction"],
                "scoring_points": row["scoring_points"],
            }
            for i, row in df.iterrows()
        }
    elif agent_name == "adk_swe":
        from datasets import load_dataset

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", cache_dir=str(agent_dir / ".cache"))
        tasks = {
            ex["instance_id"]: {
                "task_id": ex["instance_id"],
                "problem_statement": ex["problem_statement"],
                "patch": ex["patch"],
            }
            for ex in ds
        }
    else:
        tasks = {}
    return tasks, extra_dir


def prepare_workdir(task: dict, workdir: str, agent_name: str, extra_dir: str = None):
    """Prepare workdir with qa.json and extra files"""
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "monitor_data"), exist_ok=True)
    # Write qa.json
    with open(os.path.join(workdir, "qa.json"), "w") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)
    info = task.get("info", task)
    # Copy extra files for specific agents
    if agent_name == "veadk_gaia" and extra_dir:
        if fn := info.get("file_name"):
            src = os.path.join(extra_dir, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(workdir, os.path.basename(fn)))
    elif agent_name == "langgraph_sql" and extra_dir:
        if db_id := task.get("db_id"):
            for sub in ["database", "test_database", ""]:
                db_dir = os.path.join(extra_dir, sub, db_id) if sub else os.path.join(extra_dir, db_id)
                db_path = os.path.join(db_dir, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    shutil.copy(db_path, os.path.join(workdir, f"{db_id}.sqlite"))
                    schema = os.path.join(db_dir, "schema.sql")
                    if os.path.exists(schema):
                        shutil.copy(schema, os.path.join(workdir, "schema.sql"))
                    break


def run_task(task: dict, agent_name: str, agent_dir: str, workdir: str, extra_dir: str, result_prefix: str) -> int:
    """Run a single task, return corr (1=success, 0=fail)"""
    main_script = os.path.join(agent_dir, "main.py")
    backup_prompt = os.path.join(agent_dir, "prompt.py.bak")
    source_prompt = os.path.join(agent_dir, "prompt.py")

    prepare_workdir(task, workdir, agent_name, extra_dir)

    env = os.environ.copy()
    if agent_name in ("agno_rca", "adk_swe") and "OPENAI_MODEL_NAME" not in env:
        env["OPENAI_MODEL_NAME"] = f"openai/{os.getenv('OPENAI_MODEL', 'doubao-seed-1.6-250615')}"

    result = 0

    for attempt in range(1, 4):  # max 3 attempts
        proc = None
        try:
            # Restore original prompt (no principle injection)
            if os.path.exists(backup_prompt):
                shutil.copy(backup_prompt, source_prompt)
            shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))

            cmd = [sys.executable, main_script, "--workdir", workdir]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=agent_dir
            )

            timer = threading.Timer(3600, proc.kill)  # 1h timeout
            timer.start()
            stdout_lines = []
            try:
                for line in proc.stdout:
                    print(f"  {line.rstrip()}", flush=True)
                    stdout_lines.append(line)
            finally:
                timer.cancel()

            proc.wait()
            if proc.returncode == 0:
                for line in stdout_lines:
                    if line.startswith(result_prefix):
                        try:
                            result = int(json.loads(line.split(result_prefix, 1)[1].strip()).get("_this_corr", 0))
                            break
                        except:
                            pass
                break
        except Exception as e:
            print(f"  Attempt {attempt} error: {e}", flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
    cleanup_workdir(workdir, agent_name)
    return result


# === Config: tasks to retry by agent ===
# 00d579ea-0889-4fd9-a771-2c8d79835c8d_20260111214747
RETRY_TASKS = {
    # "veadk_gaia": ['00d579ea-0889-4fd9-a771-2c8d79835c8d', '0383a3ee-47a7-41a4-b493-519bdefe0488', '04a04a9b-226c-43fd-b319-d5e89743676f', '0512426f-4d28-49f0-be77-06d05daec096', '05407167-39ec-4d3a-a234-73a9120c325d', '08cae58d-4084-4616-b6dd-dd6534e4825b', '0a3cd321-3e76-4622-911b-0fda2e5d6b1a', '0a65cb96-cb6e-4a6a-8aae-c1084f613456', '0b260a57-3f3a-4405-9f29-6d7a1012dbfb', '0bb3b44a-ede5-4db5-a520-4e844b0079c5', '0bdb7c40-671d-4ad1-9ce3-986b159c0ddc', '0e9e85b8-52b9-4de4-b402-5f635ab9631f', '0ff53813-3367-4f43-bcbd-3fd725c1bf4b', '114d5fd0-e2ae-4b6d-a65a-870da2d19c08', '16d825ff-1623-4176-a5b5-42e0f5c2b0ac', '17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc', '1dcc160f-c187-48c2-b68e-319bd4354f3d', '20194330-9976-4043-8632-f8485c6c71b2', '23dd907f-1261-4488-b21c-e9185af91d5e', '2a649bb1-795f-4a01-b3be-9a01868dae73', '2d83110e-a098-4ebb-9987-066c06fa42d0', '2dfc4c37-fec1-4518-84a7-10095d30ad75', '305ac316-eef6-4446-960a-92d80d542f82', '384d0dd8-e8a4-4cfe-963c-d37f256e7662', '3cef3a44-215e-4aed-8e3b-b1e3f08063b7', '3da89939-209c-4086-8520-7eb734e6b4ef', '42576abe-0deb-4869-8c63-225c2d75a95a', '42d4198c-5895-4f0a-b0c0-424a66465d83', '46719c30-f4c3-4cad-be07-d5cb21eee6bb', '48eb8242-1099-4c26-95d4-ef22b002457a', '4b6bb5f7-f634-410e-815d-e673ab7f8632', '4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2', '50f58759-7bd6-406f-9b0d-5692beb2a926', '56137764-b4e0-45b8-9c52-1866420c3df5', '5b2a14e8-6e59-479c-80e3-4696e8980152', '5d0080cb-90d7-4712-bc33-848150e917d3', '5f982798-16b9-4051-ab57-cfc7ebdb2a91', '624cbf11-6a41-4692-af9c-36b3e5ca3130', '6359a0b1-8f7b-499b-9336-840f9ab90688', '65afbc8a-89ca-4ad5-8d62-355bb401f61d', '65da0822-a48a-4a68-bbad-8ed1b835a834', '676e5e31-a554-4acc-9286-b60d90a92d26', '6b078778-0b90-464d-83f6-59511c811b01', '708b99c5-e4a7-49cb-a5cf-933c8d46470d', '71345b0a-9c7d-4b50-b2bf-937ec5879845', '72c06643-a2fa-4186-aa5c-9ec33ae9b445', '72e110e7-464c-453c-a309-90a95aed6538', '73c1b9fe-ee1d-4cf4-96ca-35c08f97b054', '7673d772-ef80-4f0f-a602-1bf4485c9b43', '7a4a336d-dcfa-45a0-b014-824c7619e8de', '7bd855d8-463d-4ed5-93ca-5fe35145f733', '7d4a7d1d-cac6-44a8-96e8-ea9584a70825', '8131e2c0-0083-4265-9ce7-78c2d568425d', '851e570a-e3de-4d84-bcfa-cc85578baa59', '853c8244-429e-46ca-89f2-addf40dfb2bd', '872bfbb1-9ccf-49f6-8c5f-aa22818ccd66', '8b3379c0-0981-4f5b-8407-6444610cb212', '8d46b8d6-b38a-47ff-ac74-cda14cf2d19b', '8f80e01c-1296-4371-9486-bb3d68651a60', '9318445f-fe6a-4e1b-acbf-c68228c9906a', '983bba7c-c092-455f-b6c9-7857003d48fc', '9d191bce-651d-4746-be2d-7ef8ecadb9c2', '9e1fc53b-46ff-49a1-9d05-9e6faac34cc5', 'a0c07678-e491-4bbc-8f0b-07405144218f', 'a7feb290-76bb-4cb7-8800-7edaf7954f2f', 'ad2b4d70-9314-4fe6-bfbe-894a45f6055f', 'b2c257e0-3ad7-4f05-b8e3-d9da973be36e', 'b415aba4-4b68-4fc6-9b89-2c812e55a3e1', 'b4cc024b-3f5e-480e-b96a-6656493255b5', 'b7f857e4-d8aa-4387-af2a-0e844df5b9d8', 'b816bfce-3d80-4913-a07d-69b752ce6377', 'bda648d7-d618-4883-88f4-3466eabd860e', 'c3a79cfe-8206-451f-aca8-3fec8ebe51d3', 'c526d8d6-5987-4da9-b24c-83466fa172f3', 'c61d22de-5f6c-4958-a7f6-5e9707bd3466', 'c8b7e059-c60d-472e-ad64-3b04ae1166dc', 'cabe07ed-9eca-40ea-8ead-410ef5e83f91', 'cca530fc-4052-43b2-b130-b30968d8aa44', 'cca70ce6-1952-45d2-acd4-80c903b0bc49', 'd0633230-7067-47a9-9dbf-ee11e0a2cdd6', 'd1af70ea-a9a4-421a-b9cc-94b5e02f1788', 'd5141ca5-e7a0-469f-bf3e-e773507c86e2', 'd8152ad6-e4d5-4c12-8bb7-8d57dc10c6de', 'da52d699-e8d2-4dc5-9191-a2199e0b6a9b', 'dc22a632-937f-4e6a-b72f-ba0ff3f5ff97', 'de9887f5-ead8-4727-876f-5a4078f8598c', 'ded28325-3447-4c56-860f-e497d6fb3577', 'df6561b2-7ee5-4540-baab-5095f742716a', 'e142056d-56ab-4352-b091-b56054bd1359', 'e1fc63a2-da7a-432f-be78-7c4a95598703', 'e29834fd-413a-455c-a33e-c3915b07401c', 'e2d69698-bc99-4e85-9880-67eaccd66e6c', 'e8cb5b03-41e0-4086-99e5-f6806cd97211', 'e961a717-6b25-4175-8a68-874d28190ee4', 'ebbc1f13-d24d-40df-9068-adcf735b4240', 'ec09fa32-d03f-4bf8-84b0-1f16922c3ae4', 'ecbc4f94-95a3-4cc7-b255-6741a458a625', 'ed58682d-bc52-4baa-9eb0-4eb81e1edacc', 'f0f46385-fc03-4599-b5d3-f56496c3e69f', 'f2feb6a4-363c-4c09-a804-0db564eafd68']+
    # ['023e9d44-96ae-4eed-b912-244ee8c3b994', '076c8171-9b3b-49b9-a477-244d2a532826', '08c0b6e9-1b43-4c2e-ae55-4e3fce2c2715', '08f3a05f-5947-4089-a4c4-d4bcfaa6b7a0', '11af4e1a-5f45-467d-9aeb-46f4bb0bf034', '14569e28-c88c-43e4-8c32-097d35b9a67d', '27d5d136-8563-469e-92bf-fd103c28b57c', '32102e3e-d12a-4209-9163-7b3a104efe5d', '33d8ea3b-6c6b-4ff1-803d-7e270dea8a57', '3627a8be-a77f-41bb-b807-7e1bd4c0ebdf', '366e2f2b-8632-4ef2-81eb-bc3877489217', '3f57289b-8c60-48be-bd80-01f8099ca449', '3ff6b7a9-a5bd-4412-ad92-0cd0d45c0fee', '4b650a35-8529-4695-89ed-8dc7a500a498', '4d0aa727-86b1-406b-9b33-f870dd14a4a5', '4fc2f1ae-8625-45b5-ab34-ad4433bc21f8', '50ad0280-0819-4bd9-b275-5de32d3b5bcb', '50ec8903-b81f-4257-9450-1085afd2c319', '5188369a-3bbe-43d8-8b94-11558f909a08', '544b7f0c-173a-4377-8d56-57b36eb26ddf', '54612da3-fd56-4941-80f4-5eb82330de25', '56db2318-640f-477a-a82f-bc93ad13e882', '5a0c1adf-205e-4841-a666-7c3ef95def9d', '5cfb274c-0207-4aa7-9575-6ac0bd95d9b2', '65638e28-7f37-4fa7-b7b9-8c19bb609879', '67e8878b-5cef-4375-804e-e6291fdbe78a', '6f37996b-2ac7-44b0-8e68-6d28256631b4', '7619a514-5fa8-43ef-9143-83b66a43d7a4', '7b5377b0-3f38-4103-8ad2-90fe89864c04', '7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f', '840bfca7-4f7b-481a-8794-c560c340185d', '87c610df-bef7-4932-b950-1d83ef4e282b', '8e867cd7-cff9-4e6c-867a-ff5ddc2550be', '935e2cff-ae78-4218-b3f5-115589b19dae', '9f41b083-683e-4dcf-9185-ccfeaa88fa45', 'a0068077-79f4-461a-adfe-75c1a4148545', 'a1e91b78-d3d8-4675-bb8d-62741b4b68a6', 'a26649c6-1cb2-470a-871e-6910c64c3e53', 'a56f1527-3abf-41d6-91f8-7296d6336c3f', 'ad37a656-079a-49f9-a493-7b739c9167d1', 'b9763138-c053-4832-9f55-86200cb1f99c', 'c365c1c7-a3db-4d5e-a9a1-66f56eae7865', 'c714ab3a-da30-4603-bacd-d008800188b9', 'cf106601-ab4f-4af9-b045-5295fe67b37d', 'd700d50d-c707-4dca-90dc-4528cddd0c80', 'db4fd70a-2d37-40ea-873f-9433dc5e301f', 'dc28cf18-6431-458b-83ef-64b3ce566c10', 'dd3c7503-f62a-4bd0-9f67-1b63b94194cc', 'e0c10771-d627-4fd7-9694-05348e54ee36', 'e4e91f1c-1dcd-439e-9fdd-cb976f5293fd', 'e9a2c537-8232-4c3f-85b0-b52de6bcba99', 'edd4d4f2-1a58-45c4-b038-67337af4e029', 'f46b4380-207e-4434-820b-f32ce04ae2a4'],
    "agno_rca":  ['task_1-0008', 'task_1-0026', 'task_1-0029', 'task_1-0060', 'task_1-0070', 'task_2-0002', 'task_2-0011', 'task_2-0014', 'task_2-0015', 'task_2-0022', 'task_2-0027', 'task_2-0035', 'task_2-0039', 'task_2-0045', 'task_2-0047', 'task_2-0048', 'task_2-0056', 'task_2-0057', 'task_2-0063', 'task_2-0065', 'task_3-0000', 'task_3-0001', 'task_3-0018', 'task_3-0031', 'task_3-0033', 'task_3-0038', 'task_3-0046', 'task_3-0052', 'task_3-0055', 'task_4-0016', 'task_4-0017', 'task_4-0032', 'task_4-0041', 'task_4-0044', 'task_4-0059', 'task_4-0068', 'task_4-0071', 'task_4-0072', 'task_4-0074', 'task_5-0009', 'task_5-0012', 'task_5-0050', 'task_5-0051', 'task_5-0054', 'task_5-0058', 'task_5-0069', 'task_5-0073', 'task_5-0075', 'task_5-0076', 'task_6-0005', 'task_6-0010', 'task_6-0013', 'task_6-0023', 'task_6-0037', 'task_6-0040', 'task_6-0043', 'task_6-0049', 'task_6-0053', 'task_6-0064', 'task_6-0067', 'task_7-0028', 'task_7-0030', 'task_7-0066', 'task_7-0077'],
}
#     "adk_swe": ['astropy__astropy-13033', 'astropy__astropy-13236', 'astropy__astropy-13398', 'astropy__astropy-13453', 'astropy__astropy-13579', 'astropy__astropy-13977', 'astropy__astropy-14096', 'astropy__astropy-14182', 'astropy__astropy-14365', 'astropy__astropy-14369', 'astropy__astropy-14508', 'astropy__astropy-14598', 'astropy__astropy-8707', 'astropy__astropy-8872', 'django__django-10554', 'django__django-10999', 'django__django-11087', 'django__django-11138', 'django__django-11141', 'django__django-11149', 'django__django-11206', 'django__django-11211', 'django__django-11239', 'django__django-11265', 'django__django-11292', 'django__django-11299', 'django__django-11333', 'django__django-11400', 'django__django-11433', 'django__django-11477', 'django__django-11490', 'django__django-11532', 'django__django-11555', 'django__django-11728', 'django__django-11734', 'django__django-11740', 'django__django-11749', 'django__django-11790', 'django__django-11820', 'django__django-11848', 'django__django-11885', 'django__django-11964', 'django__django-11999', 'django__django-12125', 'django__django-12193', 'django__django-12262', 'django__django-12273', 'django__django-12304', 'django__django-12308', 'django__django-12325', 'django__django-12406', 'django__django-12663', 'django__django-12708', 'django__django-12754', 'django__django-12774', 'django__django-12965', 'django__django-13012', 'django__django-13023', 'django__django-13028', 'django__django-13033', 'django__django-13112', 'django__django-13121', 'django__django-13128'] + ['astropy__astropy-12907', 'astropy__astropy-14309', 'astropy__astropy-14539', 'astropy__astropy-14995', 'astropy__astropy-7166', 'astropy__astropy-7336', 'astropy__astropy-7606', 'astropy__astropy-7671', 'django__django-10880', 'django__django-10914', 'django__django-10973', 'django__django-11066', 'django__django-11095', 'django__django-11099', 'django__django-11119', 'django__django-11133', 'django__django-11163', 'django__django-11179', 'django__django-11276', 'django__django-11451', 'django__django-11551', 'django__django-11603', 'django__django-11815', 'django__django-11880', 'django__django-11951', 'django__django-12039', 'django__django-12050', 'django__django-12143', 'django__django-12155', 'django__django-12209', 'django__django-12276', 'django__django-12419', 'django__django-12713', 'django__django-12741', 'django__django-12858', 'django__django-13089', 'django__django-13109'],
#     "agno_rca": ['task_1-0002', 'task_1-0010', 'task_1-0011', 'task_1-0015', 'task_1-0034', 'task_1-0041', 'task_1-0043', 'task_1-0052', 'task_1-0053', 'task_1-0062', 'task_2-0005', 'task_2-0012', 'task_2-0017', 'task_2-0028', 'task_2-0031', 'task_2-0032', 'task_2-0050', 'task_2-0055', 'task_2-0060', 'task_2-0069', 'task_3-0004', 'task_3-0024', 'task_3-0030', 'task_3-0037', 'task_3-0048', 'task_3-0057', 'task_3-0064', 'task_4-0009', 'task_4-0022', 'task_4-0035', 'task_4-0039', 'task_4-0042', 'task_4-0063', 'task_4-0065', 'task_5-0008', 'task_5-0013', 'task_5-0014', 'task_5-0016', 'task_5-0021', 'task_5-0023', 'task_5-0036', 'task_5-0040', 'task_5-0046', 'task_5-0049', 'task_6-0000', 'task_6-0003', 'task_6-0006', 'task_6-0007', 'task_6-0018', 'task_6-0019', 'task_6-0020', 'task_6-0029', 'task_6-0045', 'task_6-0051', 'task_6-0054', 'task_6-0066', 'task_7-0025', 'task_7-0026', 'task_7-0027', 'task_7-0033', 'task_7-0038', 'task_7-0044', 'task_7-0047', 'task_7-0056', 'task_7-0059', 'task_7-0061', 'task_7-0067']+
#     ['task_1-0001', 'task_1-0058', 'task_3-0068'],
#     "autogen_math": ['gsm8k-0064', 'gsm8k-0119', 'gsm8k-0146', 'gsm8k-0163', 'gsm8k-0201', 'gsm8k-0230', 'gsm8k-0241', 'gsm8k-0249', 'gsm8k-0255', 'gsm8k-0304', 'gsm8k-0306', 'gsm8k-0368', 'gsm8k-0380', 'gsm8k-0403', 'gsm8k-0454', 'gsm8k-0505', 'gsm8k-0515', 'gsm8k-0539', 'gsm8k-0590', 'gsm8k-0610', 'gsm8k-0611', 'gsm8k-0640', 'gsm8k-0641', 'gsm8k-0642', 'gsm8k-0649', 'gsm8k-0682', 'gsm8k-0711', 'gsm8k-0749', 'gsm8k-0751', 'gsm8k-0768', 'gsm8k-0780', 'gsm8k-0782', 'gsm8k-0796', 'gsm8k-0814', 'gsm8k-0819', 'gsm8k-0823', 'gsm8k-0829', 'gsm8k-0835', 'gsm8k-0852', 'gsm8k-0951', 'gsm8k-0952', 'gsm8k-0962', 'gsm8k-0984', 'gsm8k-0997', 'gsm8k-1001', 'gsm8k-1009', 'gsm8k-1012', 'gsm8k-1016', 'gsm8k-1019', 'gsm8k-1035', 'gsm8k-1042', 'gsm8k-1048', 'gsm8k-1050', 'gsm8k-1059', 'gsm8k-1067', 'gsm8k-1074', 'gsm8k-1076', 'gsm8k-1137', 'gsm8k-1161', 'gsm8k-1183', 'gsm8k-1206', 'gsm8k-1288', 'gsm8k-1309']+
#     ['gsm8k-0000', 'gsm8k-0001', 'gsm8k-0002', 'gsm8k-0003', 'gsm8k-0004', 'gsm8k-0005', 'gsm8k-0006', 'gsm8k-0007', 'gsm8k-0008', 'gsm8k-0009', 'gsm8k-0010', 'gsm8k-0011', 'gsm8k-0012', 'gsm8k-0013', 'gsm8k-0014', 'gsm8k-0015', 'gsm8k-0016', 'gsm8k-0017', 'gsm8k-0018', 'gsm8k-0019', 'gsm8k-0020', 'gsm8k-0021', 'gsm8k-0022', 'gsm8k-0023', 'gsm8k-0024', 'gsm8k-0025', 'gsm8k-0026', 'gsm8k-0027', 'gsm8k-0028', 'gsm8k-0029', 'gsm8k-0030', 'gsm8k-0031', 'gsm8k-0032', 'gsm8k-0033', 'gsm8k-0034', 'gsm8k-0035', 'gsm8k-0036', 'gsm8k-0037', 'gsm8k-0038', 'gsm8k-0039', 'gsm8k-0040', 'gsm8k-0041', 'gsm8k-0042', 'gsm8k-0043', 'gsm8k-0044', 'gsm8k-0045', 'gsm8k-0046', 'gsm8k-0047', 'gsm8k-0048', 'gsm8k-0049', 'gsm8k-0050', 'gsm8k-0051', 'gsm8k-0052', 'gsm8k-0053', 'gsm8k-0054', 'gsm8k-0055', 'gsm8k-0056', 'gsm8k-0057', 'gsm8k-0058', 'gsm8k-0059', 'gsm8k-0060', 'gsm8k-0061', 'gsm8k-0062', 'gsm8k-0063', 'gsm8k-0065', 'gsm8k-0066', 'gsm8k-0067', 'gsm8k-0068', 'gsm8k-0069', 'gsm8k-0070', 'gsm8k-0071', 'gsm8k-0072', 'gsm8k-0073', 'gsm8k-0074', 'gsm8k-0075', 'gsm8k-0076', 'gsm8k-0077', 'gsm8k-0078', 'gsm8k-0079', 'gsm8k-0080', 'gsm8k-0081', 'gsm8k-0082', 'gsm8k-0083', 'gsm8k-0084', 'gsm8k-0085', 'gsm8k-0086', 'gsm8k-0087', 'gsm8k-0088', 'gsm8k-0089', 'gsm8k-0090', 'gsm8k-0091', 'gsm8k-0092', 'gsm8k-0093', 'gsm8k-0094', 'gsm8k-0095', 'gsm8k-0096', 'gsm8k-0097', 'gsm8k-0098', 'gsm8k-0099', 'gsm8k-0100'],
#     "langgraph_sql":['spider-0026', 'spider-0027', 'spider-0034', 'spider-0040', 'spider-0041', 'spider-0043', 'spider-0045', 'spider-0052', 'spider-0053', 'spider-0056', 'spider-0057', 'spider-0063', 'spider-0065', 'spider-0069', 'spider-0074', 'spider-0075', 'spider-0077', 'spider-0100', 'spider-0105', 'spider-0110', 'spider-0111', 'spider-0125', 'spider-0129', 'spider-0133', 'spider-0135', 'spider-0137', 'spider-0140', 'spider-0141', 'spider-0144', 'spider-0148', 'spider-0157', 'spider-0159', 'spider-0160', 'spider-0173', 'spider-0174', 'spider-0180', 'spider-0207', 'spider-0208', 'spider-0213', 'spider-0214', 'spider-0217', 'spider-0218', 'spider-0227', 'spider-0228', 'spider-0259', 'spider-0267', 'spider-0268', 'spider-0277', 'spider-0278', 'spider-0287', 'spider-0293', 'spider-0294', 'spider-0296', 'spider-0299', 'spider-0300', 'spider-0338', 'spider-0340', 'spider-0342', 'spider-0345', 'spider-0346', 'spider-0359', 'spider-0360', 'spider-0363', 'spider-0365', 'spider-0366', 'spider-0368', 'spider-0371', 'spider-0372', 'spider-0377', 'spider-0378', 'spider-0379', 'spider-0380', 'spider-0383', 'spider-0387', 'spider-0388', 'spider-0394', 'spider-0403', 'spider-0404', 'spider-0405', 'spider-0407', 'spider-0408', 'spider-0423', 'spider-0424', 'spider-0431', 'spider-0432', 'spider-0433', 'spider-0434', 'spider-0437', 'spider-0438', 'spider-0441', 'spider-0442', 'spider-0443', 'spider-0449', 'spider-0450', 'spider-0453', 'spider-0454', 'spider-0456', 'spider-0457', 'spider-0458', 'spider-0461']+
#     ['spider-0000', 'spider-0001', 'spider-0002', 'spider-0003', 'spider-0004', 'spider-0005', 'spider-0006', 'spider-0007', 'spider-0008', 'spider-0009', 'spider-0010', 'spider-0011', 'spider-0012', 'spider-0013', 'spider-0014', 'spider-0015', 'spider-0016', 'spider-0017', 'spider-0018', 'spider-0019', 'spider-0020', 'spider-0021', 'spider-0022', 'spider-0023', 'spider-0024', 'spider-0025', 'spider-0028', 'spider-0029', 'spider-0030', 'spider-0031', 'spider-0032', 'spider-0033', 'spider-0035', 'spider-0036', 'spider-0037', 'spider-0038', 'spider-0039', 'spider-0042', 'spider-0044', 'spider-0046', 'spider-0047', 'spider-0048', 'spider-0049', 'spider-0050', 'spider-0051', 'spider-0054', 'spider-0055', 'spider-0058', 'spider-0059', 'spider-0060', 'spider-0061', 'spider-0062', 'spider-0064', 'spider-0066', 'spider-0067', 'spider-0068', 'spider-0070', 'spider-0071', 'spider-0072', 'spider-0073', 'spider-0076', 'spider-0078', 'spider-0079', 'spider-0080', 'spider-0081', 'spider-0082', 'spider-0083', 'spider-0084', 'spider-0085', 'spider-0086', 'spider-0087', 'spider-0088', 'spider-0089', 'spider-0090', 'spider-0091', 'spider-0092', 'spider-0093', 'spider-0094', 'spider-0095', 'spider-0096', 'spider-0097', 'spider-0098', 'spider-0099', 'spider-0101', 'spider-0102', 'spider-0103', 'spider-0104', 'spider-0106', 'spider-0107', 'spider-0108', 'spider-0109', 'spider-0112', 'spider-0113', 'spider-0114', 'spider-0115', 'spider-0116', 'spider-0117', 'spider-0118', 'spider-0119', 'spider-0120']
# ,
#     "veadk_gaia": ['00d579ea-0889-4fd9-a771-2c8d79835c8d', '0383a3ee-47a7-41a4-b493-519bdefe0488', '04a04a9b-226c-43fd-b319-d5e89743676f', '0512426f-4d28-49f0-be77-06d05daec096', '05407167-39ec-4d3a-a234-73a9120c325d', '08cae58d-4084-4616-b6dd-dd6534e4825b', '0a3cd321-3e76-4622-911b-0fda2e5d6b1a', '0a65cb96-cb6e-4a6a-8aae-c1084f613456', '0b260a57-3f3a-4405-9f29-6d7a1012dbfb', '0bb3b44a-ede5-4db5-a520-4e844b0079c5', '0bdb7c40-671d-4ad1-9ce3-986b159c0ddc', '0e9e85b8-52b9-4de4-b402-5f635ab9631f', '0ff53813-3367-4f43-bcbd-3fd725c1bf4b', '114d5fd0-e2ae-4b6d-a65a-870da2d19c08', '16d825ff-1623-4176-a5b5-42e0f5c2b0ac', '17b5a6a3-bc87-42e8-b0fb-6ab0781ef2cc', '1dcc160f-c187-48c2-b68e-319bd4354f3d', '20194330-9976-4043-8632-f8485c6c71b2', '23dd907f-1261-4488-b21c-e9185af91d5e', '2a649bb1-795f-4a01-b3be-9a01868dae73', '2d83110e-a098-4ebb-9987-066c06fa42d0', '2dfc4c37-fec1-4518-84a7-10095d30ad75', '305ac316-eef6-4446-960a-92d80d542f82', '384d0dd8-e8a4-4cfe-963c-d37f256e7662', '3cef3a44-215e-4aed-8e3b-b1e3f08063b7', '3da89939-209c-4086-8520-7eb734e6b4ef', '42576abe-0deb-4869-8c63-225c2d75a95a', '42d4198c-5895-4f0a-b0c0-424a66465d83', '46719c30-f4c3-4cad-be07-d5cb21eee6bb', '48eb8242-1099-4c26-95d4-ef22b002457a', '4b6bb5f7-f634-410e-815d-e673ab7f8632', '4d51c4bf-4b0e-4f3d-897b-3f6687a7d9f2', '50f58759-7bd6-406f-9b0d-5692beb2a926', '56137764-b4e0-45b8-9c52-1866420c3df5', '5b2a14e8-6e59-479c-80e3-4696e8980152', '5d0080cb-90d7-4712-bc33-848150e917d3', '5f982798-16b9-4051-ab57-cfc7ebdb2a91', '624cbf11-6a41-4692-af9c-36b3e5ca3130', '6359a0b1-8f7b-499b-9336-840f9ab90688', '65afbc8a-89ca-4ad5-8d62-355bb401f61d', '65da0822-a48a-4a68-bbad-8ed1b835a834', '676e5e31-a554-4acc-9286-b60d90a92d26', '6b078778-0b90-464d-83f6-59511c811b01', '708b99c5-e4a7-49cb-a5cf-933c8d46470d', '71345b0a-9c7d-4b50-b2bf-937ec5879845', '72c06643-a2fa-4186-aa5c-9ec33ae9b445', '72e110e7-464c-453c-a309-90a95aed6538', '73c1b9fe-ee1d-4cf4-96ca-35c08f97b054', '7673d772-ef80-4f0f-a602-1bf4485c9b43', '7a4a336d-dcfa-45a0-b014-824c7619e8de', '7bd855d8-463d-4ed5-93ca-5fe35145f733', '7d4a7d1d-cac6-44a8-96e8-ea9584a70825', '8131e2c0-0083-4265-9ce7-78c2d568425d', '851e570a-e3de-4d84-bcfa-cc85578baa59', '853c8244-429e-46ca-89f2-addf40dfb2bd', '872bfbb1-9ccf-49f6-8c5f-aa22818ccd66', '8b3379c0-0981-4f5b-8407-6444610cb212', '8d46b8d6-b38a-47ff-ac74-cda14cf2d19b', '8f80e01c-1296-4371-9486-bb3d68651a60', '9318445f-fe6a-4e1b-acbf-c68228c9906a', '983bba7c-c092-455f-b6c9-7857003d48fc', '9d191bce-651d-4746-be2d-7ef8ecadb9c2', '9e1fc53b-46ff-49a1-9d05-9e6faac34cc5', 'a0c07678-e491-4bbc-8f0b-07405144218f', 'a7feb290-76bb-4cb7-8800-7edaf7954f2f', 'ad2b4d70-9314-4fe6-bfbe-894a45f6055f', 'b2c257e0-3ad7-4f05-b8e3-d9da973be36e', 'b415aba4-4b68-4fc6-9b89-2c812e55a3e1', 'b4cc024b-3f5e-480e-b96a-6656493255b5', 'b7f857e4-d8aa-4387-af2a-0e844df5b9d8', 'b816bfce-3d80-4913-a07d-69b752ce6377', 'bda648d7-d618-4883-88f4-3466eabd860e', 'c3a79cfe-8206-451f-aca8-3fec8ebe51d3', 'c526d8d6-5987-4da9-b24c-83466fa172f3', 'c61d22de-5f6c-4958-a7f6-5e9707bd3466', 'c8b7e059-c60d-472e-ad64-3b04ae1166dc', 'cabe07ed-9eca-40ea-8ead-410ef5e83f91', 'cca530fc-4052-43b2-b130-b30968d8aa44', 'cca70ce6-1952-45d2-acd4-80c903b0bc49', 'd0633230-7067-47a9-9dbf-ee11e0a2cdd6', 'd1af70ea-a9a4-421a-b9cc-94b5e02f1788', 'd5141ca5-e7a0-469f-bf3e-e773507c86e2', 'd8152ad6-e4d5-4c12-8bb7-8d57dc10c6de', 'da52d699-e8d2-4dc5-9191-a2199e0b6a9b', 'dc22a632-937f-4e6a-b72f-ba0ff3f5ff97', 'de9887f5-ead8-4727-876f-5a4078f8598c', 'ded28325-3447-4c56-860f-e497d6fb3577', 'df6561b2-7ee5-4540-baab-5095f742716a', 'e142056d-56ab-4352-b091-b56054bd1359', 'e1fc63a2-da7a-432f-be78-7c4a95598703', 'e29834fd-413a-455c-a33e-c3915b07401c', 'e2d69698-bc99-4e85-9880-67eaccd66e6c', 'e8cb5b03-41e0-4086-99e5-f6806cd97211', 'e961a717-6b25-4175-8a68-874d28190ee4', 'ebbc1f13-d24d-40df-9068-adcf735b4240', 'ec09fa32-d03f-4bf8-84b0-1f16922c3ae4', 'ecbc4f94-95a3-4cc7-b255-6741a458a625', 'ed58682d-bc52-4baa-9eb0-4eb81e1edacc', 'f0f46385-fc03-4599-b5d3-f56496c3e69f', 'f2feb6a4-363c-4c09-a804-0db564eafd68']+
#     ['023e9d44-96ae-4eed-b912-244ee8c3b994', '076c8171-9b3b-49b9-a477-244d2a532826', '08c0b6e9-1b43-4c2e-ae55-4e3fce2c2715', '08f3a05f-5947-4089-a4c4-d4bcfaa6b7a0', '11af4e1a-5f45-467d-9aeb-46f4bb0bf034', '14569e28-c88c-43e4-8c32-097d35b9a67d', '27d5d136-8563-469e-92bf-fd103c28b57c', '32102e3e-d12a-4209-9163-7b3a104efe5d', '33d8ea3b-6c6b-4ff1-803d-7e270dea8a57', '3627a8be-a77f-41bb-b807-7e1bd4c0ebdf', '366e2f2b-8632-4ef2-81eb-bc3877489217', '3f57289b-8c60-48be-bd80-01f8099ca449', '3ff6b7a9-a5bd-4412-ad92-0cd0d45c0fee', '4b650a35-8529-4695-89ed-8dc7a500a498', '4d0aa727-86b1-406b-9b33-f870dd14a4a5', '4fc2f1ae-8625-45b5-ab34-ad4433bc21f8', '50ad0280-0819-4bd9-b275-5de32d3b5bcb', '50ec8903-b81f-4257-9450-1085afd2c319', '5188369a-3bbe-43d8-8b94-11558f909a08', '544b7f0c-173a-4377-8d56-57b36eb26ddf', '54612da3-fd56-4941-80f4-5eb82330de25', '56db2318-640f-477a-a82f-bc93ad13e882', '5a0c1adf-205e-4841-a666-7c3ef95def9d', '5cfb274c-0207-4aa7-9575-6ac0bd95d9b2', '65638e28-7f37-4fa7-b7b9-8c19bb609879', '67e8878b-5cef-4375-804e-e6291fdbe78a', '6f37996b-2ac7-44b0-8e68-6d28256631b4', '7619a514-5fa8-43ef-9143-83b66a43d7a4', '7b5377b0-3f38-4103-8ad2-90fe89864c04', '7cc4acfa-63fd-4acc-a1a1-e8e529e0a97f', '840bfca7-4f7b-481a-8794-c560c340185d', '87c610df-bef7-4932-b950-1d83ef4e282b', '8e867cd7-cff9-4e6c-867a-ff5ddc2550be', '935e2cff-ae78-4218-b3f5-115589b19dae', '9f41b083-683e-4dcf-9185-ccfeaa88fa45', 'a0068077-79f4-461a-adfe-75c1a4148545', 'a1e91b78-d3d8-4675-bb8d-62741b4b68a6', 'a26649c6-1cb2-470a-871e-6910c64c3e53', 'a56f1527-3abf-41d6-91f8-7296d6336c3f', 'ad37a656-079a-49f9-a493-7b739c9167d1', 'b9763138-c053-4832-9f55-86200cb1f99c', 'c365c1c7-a3db-4d5e-a9a1-66f56eae7865', 'c714ab3a-da30-4603-bacd-d008800188b9', 'cf106601-ab4f-4af9-b045-5295fe67b37d', 'd700d50d-c707-4dca-90dc-4528cddd0c80', 'db4fd70a-2d37-40ea-873f-9433dc5e301f', 'dc28cf18-6431-458b-83ef-64b3ce566c10', 'dd3c7503-f62a-4bd0-9f67-1b63b94194cc', 'e0c10771-d627-4fd7-9694-05348e54ee36', 'e4e91f1c-1dcd-439e-9fdd-cb976f5293fd', 'e9a2c537-8232-4c3f-85b0-b52de6bcba99', 'edd4d4f2-1a58-45c4-b038-67337af4e029', 'f46b4380-207e-4434-820b-f32ce04ae2a4'],
OUTPUT_BASE = "results/all_tasks_retry"


def run_single_task(
    agent_name: str, tid: str, task: dict, agent_dir: str, extra_dir: str, result_prefix: str, workdir_root: Path
) -> tuple:
    """Run single task, return (tid, status)"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    workdir = str(workdir_root / f"{tid}_{timestamp}")
    print(f"\n[RUN] {agent_name}:{tid}")
    corr = run_task(task, agent_name, agent_dir, workdir, extra_dir, result_prefix)
    status = "success" if corr == 1 else "fail"
    print(f"[DONE] {agent_name}:{tid} -> {status}")
    return tid, status


def run_agent_tasks(
    agent_name: str,
    task_list: list,
    tasks: dict,
    agent_dir: str,
    extra_dir: str,
    result_prefix: str,
    workdir_root: Path,
    executor: ThreadPoolExecutor,
) -> dict:
    """Run tasks for agent. veadk_gaia serial, others parallel via shared executor."""
    results = {}
    valid = []
    for _, tid in task_list:
        if tid in tasks:
            valid.append((tid, tasks[tid]))
        else:
            print(f"[SKIP] {agent_name}:{tid} - not in task source")
    serial_agents = ("veadk_gaia",)
    staggered_agents = ("adk_swe",)  # Parallel with delay between launches
    print(f"[AGENT] {agent_name} starting {len(valid)} tasks (parallel={agent_name not in serial_agents})")
    if agent_name in serial_agents:
        for tid, task in valid:
            _, status = run_single_task(agent_name, tid, task, agent_dir, extra_dir, result_prefix, workdir_root)
            results[tid] = status
    elif agent_name in staggered_agents:
        futs = {}
        for i, (tid, task) in enumerate(valid):
            if i > 0:
                time.sleep(5)  # 5s delay between launches
            f = executor.submit(
                run_single_task, agent_name, tid, task, agent_dir, extra_dir, result_prefix, workdir_root
            )
            futs[f] = tid
            print(f"[LAUNCH] {agent_name}:{tid} (#{i+1}/{len(valid)})")
        for f in as_completed(futs):
            tid, status = f.result()
            results[tid] = status
    else:
        futs = {
            executor.submit(
                run_single_task, agent_name, tid, task, agent_dir, extra_dir, result_prefix, workdir_root
            ): tid
            for tid, task in valid
        }
        for f in as_completed(futs):
            tid, status = f.result()
            results[tid] = status
    success = sum(1 for s in results.values() if s == "success")
    print(f"[AGENT] {agent_name} finished: {success}/{len(results)} success")
    return results


load_dotenv()
project_root = get_project_root()
output_base = project_root / OUTPUT_BASE

# Cleanup temp files in all existing workdirs
print("[INIT] Cleaning temp files...")
for agent_name in ["adk_swe", "langgraph_sql", "agno_rca"]:
    workdir_root = output_base / agent_name
    if workdir_root.exists():
        for d in workdir_root.iterdir():
            if d.is_dir():
                cleanup_workdir(str(d), agent_name)
print("[INIT] Cleanup done")

# Load task definitions
agent_data = {}  # agent -> (tasks_dict, agent_dir, extra_dir, result_prefix)
for agent_name in RETRY_TASKS:
    cfg = AGENT_CONFIGS[agent_name]
    agent_dir = project_root / "agent_system" / cfg["dir"]
    # Ensure prompt backup
    sp, bp = agent_dir / "prompt.py", agent_dir / "prompt.py.bak"
    if sp.exists() and not bp.exists():
        shutil.copy(sp, bp)
    tasks, extra_dir = load_agent_tasks(agent_name, agent_dir)
    agent_data[agent_name] = (tasks, str(agent_dir), extra_dir, cfg["result_prefix"])

# Build pending tasks (check existing workdirs)
agent_tasks = {}  # agent -> [(folder, tid), ...]
total_tasks, total_completed = 0, 0
for agent_name, folders in RETRY_TASKS.items():
    workdir_root = output_base / agent_name
    pending = []
    for f in folders:
        tid = extract_task_id(f)
        existing = find_existing_workdir(workdir_root, tid)
        if existing:
            if is_completed(existing):
                total_completed += 1
                continue  # Skip completed
            else:
                # Remove incomplete workdir
                print(f"[CLEAN] {agent_name}:{tid} - removing incomplete workdir")
                shutil.rmtree(existing, ignore_errors=True)
        pending.append((f, tid))
    agent_tasks[agent_name] = pending
    total_tasks += len(folders)
    print(f"[{agent_name}] Total={len(folders)}, Completed={len(folders)-len(pending)}, Pending={len(pending)}")

print(f"\nTotal: {total_tasks}, Completed: {total_completed}, Pending: {total_tasks - total_completed}")

MAX_WORKERS = 2
# Run agents in parallel, tasks serial within each agent
all_results = {}  # agent -> {tid: status}
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {}
    for agent_name, task_list in agent_tasks.items():
        if not task_list:  # Skip if no pending tasks
            continue
        tasks, agent_dir, extra_dir, result_prefix = agent_data[agent_name]
        workdir_root = output_base / agent_name
        workdir_root.mkdir(parents=True, exist_ok=True)
        f = executor.submit(
            run_agent_tasks, agent_name, task_list, tasks, agent_dir, extra_dir, result_prefix, workdir_root, executor
        )
        futures[f] = agent_name
    for f in as_completed(futures):
        agent_name = futures[f]
        all_results[agent_name] = f.result()


# Summary
print(f"\n{'='*60}")
total_success = sum(1 for r in all_results.values() for s in r.values() if s == "success")
total_fail = sum(1 for r in all_results.values() for s in r.values() if s == "fail")
for agent_name, results in all_results.items():
    s = sum(1 for v in results.values() if v == "success")
    print(f"[{agent_name}] Success: {s}/{len(results)}")
print(f"[TOTAL] Success: {total_success}, Fail: {total_fail}")
print(f"Output: {output_base}")
