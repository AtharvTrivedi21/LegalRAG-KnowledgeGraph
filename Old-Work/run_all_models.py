import os
import sys
import subprocess

# These MUST match keys in CONFIGS in config.py
CONFIG_NAMES = [
    # "baseline_llama2_nomic",
    "llama3_nomic",
    "qwen_nomic",
]

CSV_PATH = "results_all_models.csv"


def main():
    # Optional: remove old CSV if you want a fresh run each time
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)

    # Resolve path to test_10_queries.py (same folder as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(script_dir, "new_test_10_q.py")

    for config_name in CONFIG_NAMES:
        print(f"\n=== Running eval for config: {config_name} ===")
        env = os.environ.copy()
        env["ACTIVE_CONFIG_NAME"] = config_name

        # Use the same Python interpreter that is running this script (venv-safe)
        subprocess.run(
            [sys.executable, test_script, "--csv", os.path.join(script_dir, "..", CSV_PATH)],
            env=env,
            check=True,
        )

    print(f"\nAll configs finished. Combined metrics in: {CSV_PATH}")


if __name__ == "__main__":
    main()