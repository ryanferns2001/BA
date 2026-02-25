import os
import json
import glob
from statistics import mean
import matplotlib.pyplot as plt
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

CURRENT = Path(__file__).resolve()
METHODS_DIR = CURRENT.parent
BLACKBOARD_DIR = METHODS_DIR.parent.parent
ROOT = BLACKBOARD_DIR.parent
VC_SLAM_BASE_PATH = ROOT / "datacorpus" / "vcslam"

VC_SLAM_BASE = str(VC_SLAM_BASE_PATH)

PREFIX_STR = (
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
    "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n"
    "@prefix vcslam: <http://www.vcslam.tmdt.info/schema#> ."
)

from datacorpus.tools.top_k_eval import evaluate_top_k


# ======================================================
# BUILD INPUT JSON
# ======================================================

def build_input_json(attributes):
    mappings = {}

    for attr_name, data in attributes.items():
        state = data.get("state", {}) or {}

        # Paper-consistent: evaluate validated candidates (C_a)
        candidates = state.get("validated_candidates")
        if candidates is None:
            candidates = state.get("candidates", [])

        arr = []
        for idx, item in enumerate(candidates):
            cand_str = (item or {}).get("candidate")
            if not cand_str:
                continue
            arr.append({
                "candidate": cand_str,
                "score": len(candidates) - idx,   # same scoring style as baseline script
            })

        mappings[attr_name] = arr

    return {
        "prefix": PREFIX_STR,
        "mappings_candidates": mappings
    }

# ======================================================
# EVALUATE @k FOR SINGLE SID
# ======================================================

def evaluate_sid(root_folder, sid, ks=(1, 3, 5, 10)):
    sid_folder = os.path.join(root_folder, sid)
    mapping_file = os.path.join(sid_folder, f"{sid}_mapping_results.json")

    if not os.path.exists(mapping_file):
        logger.info(f"Skipping {sid} (no mapping_results.json)")
        return None

    with open(mapping_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    attributes = data.get("attributes", {})
    input_json = build_input_json(attributes)

    ref_file = os.path.join(VC_SLAM_BASE, sid, f"{sid}_mapped.json")
    if not os.path.exists(ref_file):
        logger.info(f"Reference model missing for SID {sid}")
        return None

    with open(ref_file, "r", encoding="utf-8") as f:
        reference_model = json.load(f)

    results_per_k = {}

    for k in ks:
        evaluation = evaluate_top_k(
            k=k,
            reference_model=reference_model,
            to_evaluate=input_json
        )
        results_per_k[k] = evaluation

        # Save debug JSON
        debug_dir = os.path.join(root_folder, "plots", "debug", sid)
        os.makedirs(debug_dir, exist_ok=True)

        debug_file = os.path.join(debug_dir, f"{sid}_debug_hits@{k}.json")
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump({
                "input_json": input_json,
                "evaluation": evaluation
            }, f, indent=4)

        logger.info(f"Saved debug → {debug_file}")

    return results_per_k


# ======================================================
# GLOBAL AVERAGES + CSV + PLOTS
# ======================================================

def create_plot(values, ylabel, title, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(values)), values, marker="o")
    plt.title(title)
    plt.xlabel("Samples (index)")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_global_averages(results_by_sid, root_folder):
    averages = {}

    for k in (1, 3, 5, 10):
        vals = []
        for sid, res in results_by_sid.items():
            ev = res.get(k, {})
            hits = ev.get("hits@" + str(k), 0)
            total = hits + ev.get("not_hits@" + str(k), 0) + ev.get("no_mappings_provided", 0)

            if total > 0:
                vals.append((hits / total) * 100)

        averages[k] = round(mean(vals), 2) if vals else 0.0

    # Save CSV
    out_dir = os.path.join(root_folder, "plots", "debug")
    os.makedirs(out_dir, exist_ok=True)

    csv_file = os.path.join(out_dir, "debug_global_averages.csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("metric,percentage\n")
        for k in (1, 3, 5, 10):
            f.write(f"hits@{k},{averages[k]}\n")

    logger.info(f"Global averages saved → {csv_file}")

    # Create plots
    for k in (1, 3, 5, 10):
        vals = []
        for sid, res in results_by_sid.items():
            ev = res.get(k, {})
            hits = ev.get("hits@" + str(k), 0)
            total = hits + ev.get("not_hits@" + str(k), 0) + ev.get("no_mappings_provided", 0)
            vals.append((hits / total) * 100 if total > 0 else 0)

        plot_path = os.path.join(out_dir, f"hits@{k}_plot.png")
        create_plot(
            values=vals,
            ylabel="Percentage",
            title=f"Hits@{k} across samples",
            save_path=plot_path
        )


# ======================================================
# MAIN ENTRY
# ======================================================

def run(root_folder):
    sid_folders = [
        os.path.basename(p)
        for p in glob.glob(os.path.join(root_folder, "*"))
        if os.path.isdir(p)
    ]

    results_by_sid = {}

    for sid in sid_folders:
        res = evaluate_sid(root_folder, sid)
        if res:
            results_by_sid[sid] = res

    compute_global_averages(results_by_sid, root_folder)

    logger.info("\nDONE — All debug JSONs + plots + global CSV created.")
