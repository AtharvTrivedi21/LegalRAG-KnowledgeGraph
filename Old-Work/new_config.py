# config.py (conceptual structure)
from dataclasses import dataclass
import os

@dataclass
class ModelConfig:
    config_name: str
    llm_model_name: str
    embedding_model_name: str

CONFIGS = {
    # "baseline_llama2_nomic": ModelConfig(
    #     config_name="baseline_llama2_nomic",
    #     llm_model_name="llama2",
    #     embedding_model_name="nomic-embed-text",
    # ),
    "llama3_nomic": ModelConfig(
        config_name="llama3_nomic",
        llm_model_name="llama3:8b",       # change size if you want
        embedding_model_name="nomic-embed-text",
    ),
    "qwen_nomic": ModelConfig(
        config_name="qwen_nomic",
        llm_model_name="qwen2.5:7b",      # this must match the name you pulled
        embedding_model_name="nomic-embed-text",
    ),
}

ACTIVE_CONFIG_NAME = os.getenv("ACTIVE_CONFIG_NAME", "baseline_llama2_nomic")
ACTIVE_CONFIG = CONFIGS[ACTIVE_CONFIG_NAME]