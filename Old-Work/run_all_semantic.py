# app/run_all_semantic.py

import os
import sys
import subprocess

# These MUST match keys in CONFIGS in config.py
# Edit this list if you want to run only some models.
CONFIG_NAMES = [
    # "baseline_llama2_nomic",
    "llama3_nomic",
    "qwen_nomic",
]


def main():
    # CSV will be created in the current working directory
    csv_path = os.path.join(os.getcwd(), "semantic_results.csv")

    # Start fresh each time
    if os.path.exists(csv_path):
        print(f"[SEMANTIC EVAL] Removing existing CSV: {csv_path}")
        os.remove(csv_path)

    # Resolve path to semantic_eval.py (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    semantic_script = os.path.join(script_dir, "semantic_eval.py")

    print("[SEMANTIC EVAL] Will evaluate configs:", CONFIG_NAMES)
    print(f"[SEMANTIC EVAL] Results will be saved to: {csv_path}")

    for config_name in CONFIG_NAMES:
        print(f"\n=== Running semantic eval for config: {config_name} ===")

        env = os.environ.copy()
        env["ACTIVE_CONFIG_NAME"] = config_name

        # Use the same Python interpreter (venv-safe)
        subprocess.run(
            [sys.executable, semantic_script],
            env=env,
            check=True,
        )

    print(f"\n[SEMANTIC EVAL] All configs finished. Combined results in: {csv_path}")


if __name__ == "__main__":
    main()