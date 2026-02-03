# config.py
from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class Config:
    """Global config class - automatically read from env vars, support command line override"""

    # path config
    output_dir: str = field(default_factory=lambda: os.getenv("OUTPUT_DIR", os.path.join(os.getcwd(), "output")))

    # LLM config
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "your-api-key"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4"))
    app_name: str = "default_app"
    user_id: str = "default_user"
    session_id: str = "default_session_id"
    llm_max_concurrency: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_CONCURRENCY", "20")))

    # embedding config
    embedding_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")
    )
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", "ollama"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:567m-fp16"))
    embedding_max_concurrency: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "20")))

    # Judge config ranges
    reasons_range: Tuple[int, int] = (0, 10)
    fault_root_cause_range: Tuple[int, int] = (0, 10)
    execute_principle_range: Tuple[int, int] = (0, 10)
    judge_principle_range: Tuple[int, int] = (0, 10)

    def update_from_args(self, args):
        """Override config from command line args (only override non-None values)"""
        if hasattr(args, "output_dir") and args.output_dir:
            self.output_dir = args.output_dir
        if hasattr(args, "base_url") and args.base_url:
            self.openai_base_url = args.base_url
        if hasattr(args, "api_key") and args.api_key:
            self.openai_api_key = args.api_key
        if hasattr(args, "model") and args.model:
            self.openai_model = args.model
        if hasattr(args, "embedding_base_url") and args.embedding_base_url:
            self.embedding_base_url = args.embedding_base_url
        if hasattr(args, "embedding_api_key") and args.embedding_api_key:
            self.embedding_api_key = args.embedding_api_key
        if hasattr(args, "embedding_model") and args.embedding_model:
            self.embedding_model = args.embedding_model
        if hasattr(args, "reasons_range") and args.reasons_range:
            self.reasons_range = args.reasons_range
        if hasattr(args, "fault_root_cause_range") and args.fault_root_cause_range:
            self.fault_root_cause_range = args.fault_root_cause_range
        if hasattr(args, "execute_principle_range") and args.execute_principle_range:
            self.execute_principle_range = args.execute_principle_range
        if hasattr(args, "judge_principle_range") and args.judge_principle_range:
            self.judge_principle_range = args.judge_principle_range
        return self
