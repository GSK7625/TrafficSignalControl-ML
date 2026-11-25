"""In lịch test đã khóa. Chỉ chạy SUMO khi có --run-to THU_MUC_MOI."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import run_evaluation, sha256
from q_learning import load_q_table
from sumo_env import SumoEnv


TEST_SEEDS = (161101, 161102, 161103, 161104, 161105)
# Checkpoint 15, seed khám phá 7. Chọn trên validation 42 và 12345.
MODELS = {
    "cologne1": {
        "QL-Original": ("reports/ql_final_models/cologne1_original.json",
                        "1267e2e6b75f07894cf9f4252097fb719373c0240a0c89fd4229047a78017136"),
        "QL-Overdue": ("reports/ql_final_models/cologne1_overdue.json",
                      "6e9953dac311913d77d1dd5abf041a6830b6a150d323ac5566afbc4be93cd33e"),
    },
    "ingolstadt1": {
        "QL-Original": ("reports/stage7/training/ingolstadt1/exploration_7/ingolstadt1_q_table.json",
                        "24169b3ebe29d41548fc08daf4f96b4f9aa704a32343363534e4d0ae75a7573a"),
    },
}


def build_test_plan():
    runs = []
    for name, models in MODELS.items():
        env = SumoEnv(name)
        if env.end - env.begin != 3600:
            raise ValueError(f"Unexpected horizon for {name}")
        for path, digest in models.values():
            if sha256(ROOT / path) != digest:
                raise ValueError(f"Locked checkpoint differs: {path}")
            load_q_table(ROOT / path, env)
        for seed in TEST_SEEDS:
            for repeat in (1, 2):
                for label in ("FT", "MP", *models):
                    path, digest = models.get(label, (None, None))
                    runs.append(dict(map=name, controller=label, seed=seed, repeat=repeat,
                                     model=path, sha256=digest))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-to", type=Path, help="Run the full test into a new directory")
    output = parser.parse_args().run_to
    runs = build_test_plan()
    manifest = json.dumps(runs, indent=2)
    if output is None:
        print(manifest)
        return

    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(manifest, encoding="utf-8")
    metrics = ("requested", "valid_completed", "completion_rate", "throughput",
               "average_queue_length", "average_waiting_time", "average_travel_time",
               "average_time_loss", "undeparted", "unfinished", "teleports", "collisions",
               "longest_movement_no_green")
    with (output / "runs.csv").open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=(*runs[0], *metrics))
        writer.writeheader()
        for run in runs:
            name, label, seed = run["map"], run["controller"], run["seed"]
            directory = output / name / label / f"seed_{seed}" / f"repeat_{run['repeat']}"
            model = ROOT / run["model"] if run["model"] else None
            result = run_evaluation(name, "QL" if model else label, seed, directory, model,
                                    horizon=3600, expect_starvation_lambda=0 if model else None)
            if result["model_sha256_before"] != run["sha256"] or result["model_sha256_after"] != run["sha256"]:
                raise ValueError(f"Unexpected checkpoint during evaluation: {directory}")
            writer.writerow({**run, **{key: result[key] for key in metrics}})
            file.flush()


if __name__ == "__main__":
    main()
