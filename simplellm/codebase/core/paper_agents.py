import os
import json
import copy
import random
import csv
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path
from dotenv import load_dotenv
from simplellm.codebase.agents.agents import LLMMapper
from datacorpus.tools.top_k_eval import evaluate_top_k
from simplellm.evaluation_methods import evaluations
import logging
import datetime
from simplellm.codebase.config.logging_config import setup_root_logger
from openai import OpenAI


logger = logging.getLogger(__name__)

AGENTS_CONFIG = [{"name": "GPT-4o-mini", "type": "openai", "model": "gpt-4o-mini" , "weight": 1.0}]

Possible_agents = [{"name": "GPT-4o-mini", "type": "openai", "model": "gpt-4o-mini", "weight": 1.0}]

setup_root_logger(log_level="INFO")


def load_environment(vcslam_path: str):
    CURRENT = Path(__file__).resolve()
    METHODS_DIR = CURRENT.parent
    BLACKBOARD_DIR = METHODS_DIR.parent.parent
    ROOT = BLACKBOARD_DIR.parent
    ENV_DIR = ROOT / "env"

    load_dotenv(os.path.join(ENV_DIR, ".env"))
    openai_key = os.getenv("OPENAIKEY")
    ollama_host = os.getenv("OLLAMAHOST")

    base_dir = Path(vcslam_path)
    ontology_path = base_dir / "ontology" / "ontology.ttl"

    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontology not found: {ontology_path}")

    return {
        "openai_key": openai_key,
        "ollama_host": ollama_host,
        "base_dir": base_dir,
        "ontology_path": ontology_path
    }


def load_sample(base_dir: Path, sid: str):
    folder = base_dir / sid
    json_path = folder / f"{sid}_samples.json"
    unmapped_path = folder / f"{sid}_unmapped.json"
    doc_path = folder / f"{sid}.txt"
    ref_path = folder / f"{sid}_mapped.json"

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    documentation = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    unmapped = json.loads(unmapped_path.read_text(encoding="utf-8")) if unmapped_path.exists() else {}
    ref_mapping = json.loads(ref_path.read_text(encoding="utf-8")) if ref_path.exists() else {}

    return json_data, documentation, unmapped, ref_mapping



def build_agents(env):
    agents = []
    for conf in AGENTS_CONFIG:
        params = {"model": conf["model"], "temperature": 0.0 , "weight": conf["weight"]}
        if conf["type"] == "openai":
            params["api_key"] = env["openai_key"]

        elif conf["type"] == "ollama":
            params["host"] = env["ollama_host"]

        agents.append(LLMMapper(conf["name"], conf["type"], params))
    return agents


def run_agents_on_sample(agents, ontology, json_data, documentation, unmapped, historical):
    results = {}
    logger.info("Starting agents")
    for agent in agents:
        logger.debug(f"Starting {agent.name}")
        result = agent.mapping(
            json_data=json_data,
            ontology=ontology,
            documentation=documentation,
            historical_mappings=historical,
            unmapped_attributes=unmapped,
            candidate_mapping_amount=10
        )
        results[agent.name] = result
    return results



def merge_all_candidates(agent_results):

    logger.debug("Merging all candidates")
    combined = None
    for r in agent_results.values():
        if combined is None:
            combined = {"prefix": r["prefix"], "mappings_candidates": {}}
        for k, cands in r["mappings_candidates"].items():
            combined["mappings_candidates"].setdefault(k, []).extend(cands)
    return combined


def select_best_candidates(combined):

    logger.debug("Selecting best candidates")
    best = {"prefix": combined["prefix"], "mappings_candidates": {}}
    for key, cands in combined["mappings_candidates"].items():
        if not cands:
            continue
        best_cand = sorted(cands, key=lambda x: x["score"], reverse=True)[0]
        best["mappings_candidates"][key] = [best_cand]
    return best


def majority_voting_weighted(agents, combined, json_data, documentation, historical):

    combined_context = copy.deepcopy(combined)
    voted = {
        "prefix": combined["prefix"],
        "mappings_candidates": {},
        "debug_info": {}
    }

    for key in sorted(combined["mappings_candidates"].keys()):
        candidates = combined["mappings_candidates"][key]
        if not candidates:
            continue

        votes = []

        for agent in agents:
            try:
                response = agent.select_best_candidate(
                    key=key,
                    json_data=json_data,
                    documentation=documentation,
                    historical=historical,
                    candidate_list=[c["candidate"] for c in candidates],
                    struct=json.dumps(combined_context, indent=2)
                )

                model = agent.llm["model"]
                logger.debug(f"🗳️ Votum von {model}: {response}")
            except Exception as e:
                logger.exception(f"Agent {agent.name} failed for key '{key}': {e}")
                continue

            if not response:
                continue

            # 🔹 Normalisieren – falls das LLM Text oder Dict liefert
            if isinstance(response, dict):
                parsed = response
            else:
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    parsed = {"candidate": str(response), "score": None}

            cand = parsed.get("candidate")
            llm_rank = parsed.get("score", None)  # optionaler LLM-"score"

            if not cand:
                continue

            votes.append({
                "candidate": cand.strip(),
                "agent_weight": agent.get_weight(),
                "llm_rank": llm_rank,
                "agent": agent.name
            })

        if not votes:
            continue


        aggregated = {}
        for v in votes:
            ttl = v["candidate"]
            aggregated[ttl] = aggregated.get(ttl, 0) + v["agent_weight"]

        chosen, best_score = max(aggregated.items(), key=lambda x: x[1])


        voted["mappings_candidates"][key] = [{
            "candidate": chosen,
            "score": best_score
        }]

        voted["debug_info"][key] = {
            "votes": votes,
            "aggregated_scores": aggregated,
            "chosen": chosen
        }

        combined_context["mappings_candidates"][key] = [{
            "candidate": chosen,
            "score": best_score
        }]

        logger.debug(f"✅ Final choice for '{key}': {chosen} (Score: {best_score})")

    return voted

def candidate_selection_count(agent_results, agent_weights, top_m=3):

    selected = {
        "prefix": next(iter(agent_results.values()))["prefix"],
        "mappings_candidates": {},
        "debug_info": {}
    }

    all_keys = set()
    for r in agent_results.values():
        all_keys.update(r["mappings_candidates"].keys())

    for key in sorted(all_keys):
        collected = []


        for agent_name, agent_output in agent_results.items():
            weight = agent_weights.get(agent_name, 1.0)
            candidates = agent_output["mappings_candidates"].get(key, [])[:top_m]

            for idx, c in enumerate(candidates):
                collected.append({
                    "agent": agent_name,
                    "candidate": c["candidate"].strip(),
                    "rank": idx + 1,
                    "weight": weight
                })

        if not collected:
            continue

        frequency = {}
        for c in collected:
            ttl = c["candidate"]
            frequency[ttl] = frequency.get(ttl, 0) + 1

        collected.sort(key=lambda x: (-frequency[x["candidate"]], x["rank"]))


        weighted_scores = {}
        for c in collected:
            ttl = c["candidate"]
            weighted_scores[ttl] = weighted_scores.get(ttl, 0) + c["weight"]


        chosen, best_score = max(weighted_scores.items(), key=lambda x: x[1])

        selected["mappings_candidates"][key] = [{
            "candidate": chosen,
            "score": best_score
        }]
        selected["debug_info"][key] = {
            "m_matrix": collected,
            "frequency": frequency,
            "weighted_scores": weighted_scores,
            "selected_candidate": chosen
        }

    return selected


def evaluate_and_export(sample_id, reference, agent_results, selected_semantic, selected_weighted, voted, output_csv, vote_and_selection):
    headers = [
        "sample",
        "Mappings_From",
        "hits@1", "hits@1_pct",
        "hits@3", "hits@3_pct",
        "hits@5", "hits@5_pct",
        "hits@10", "hits@10_pct",
        "not_hits@1", "not_hits@3", "not_hits@5", "not_hits@10",
        "fail_abs", "fail_%"
    ]

    rows = []
    json_export = {
        "sample_id": sample_id,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "evaluated_models": {}
    }

    def summarize(name, model_data):
        total = len(reference["mappings"])

        res1 = evaluate_top_k(1, reference, model_data)
        res3 = evaluate_top_k(3, reference, model_data)
        res5 = evaluate_top_k(5, reference, model_data)
        res10 = evaluate_top_k(10, reference, model_data)


        fail_abs = res1["no_mappings_provided"]
        fail_pct = round(fail_abs / total * 100, 2) if total > 0 else 0

        hits1_pct = round(res1["hits@1"] / total * 100, 2) if total > 0 else 0
        hits3_pct = round(res3["hits@3"] / total * 100, 2) if total > 0 else 0
        hits5_pct = round(res5["hits@5"] / total * 100, 2) if total > 0 else 0
        hits10_pct = round(res10["hits@10"] / total * 100, 2) if total > 0 else 0

        row = [
            sample_id,
            name,
            res1["hits@1"], f"{hits1_pct}%",
            res3["hits@3"], f"{hits3_pct}%",
            res5["hits@5"], f"{hits5_pct}%",
            res10["hits@10"], f"{hits10_pct}%",
            res1["not_hits@1"], res3["not_hits@3"], res5["not_hits@5"], res10["not_hits@10"],
            fail_abs, f"{fail_pct}%"
        ]


        json_export["evaluated_models"][name] = {
            "metrics": {
                "hits@1": res1["hits@1"],
                "hits@1_%": f"{hits1_pct}%",
                "hits@3": res3["hits@3"],
                "hits@3_%": f"{hits3_pct}%",
                "hits@5": res5["hits@5"],
                "hits@5_%": f"{hits5_pct}%",
                "hits@10": res10["hits@10"],
                "hits@10_%": f"{hits10_pct}%",
                "not_hits@1": res1["not_hits@1"],
                "not_hits@3": res3["not_hits@3"],
                "not_hits@5": res5["not_hits@5"],
                "not_hits@10": res10["not_hits@10"],
                "fail_abs": fail_abs,
                "fail_%": f"{fail_pct}%"
            },
            "mappings_candidates": model_data.get("mappings_candidates", {}),
            "prefix": model_data.get("prefix", "")
        }

        return row


    for agent_name, model_data in agent_results.items():
        rows.append(summarize(agent_name, model_data))





    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    json_path = Path(output_csv).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_export, jf, indent=4, ensure_ascii=False)

    logger.info(f"\nResults saved in {output_csv}")





def candidate_selection_semantic_top_m(agent_results, top_m=3, model_name="all-MiniLM-L6-v2"):

    embedder = SentenceTransformer(model_name)
    selected = {
        "prefix": next(iter(agent_results.values()))["prefix"],
        "mappings_candidates": {},
        "debug_info": {}
    }

    all_keys = set()
    for r in agent_results.values():
        all_keys.update(r["mappings_candidates"].keys())

    for key in sorted(all_keys):

        collected = []
        for agent_name, agent_output in agent_results.items():
            cands = agent_output["mappings_candidates"].get(key, [])
            top_cands = sorted(cands, key=lambda x: x["score"], reverse=True)[:top_m]
            for c in top_cands:
                collected.append({
                    "agent": agent_name,
                    "candidate": c["candidate"],
                    "orig_score": c.get("score", 1.0)
                })

        if not collected:
            continue

        if len(collected) == 1:
            selected["mappings_candidates"][key] = [{
                "candidate": collected[0]["candidate"],
                "score": collected[0]["orig_score"]
            }]
            selected["debug_info"][key] = {
                "considered_candidates": collected,
                "selected_candidate": collected[0]["candidate"]
            }
            continue

        sentences = [c["candidate"] for c in collected]
        embeddings = embedder.encode(sentences, convert_to_tensor=True, normalize_embeddings=True)
        sim_matrix = util.cos_sim(embeddings, embeddings)

        avg_sim = sim_matrix.mean(dim=1).tolist()
        best_score = max(avg_sim)
        best_indices = [i for i, s in enumerate(avg_sim) if s == best_score]
        best_idx = random.choice(best_indices)
        best_cand = collected[best_idx]["candidate"]

        selected["mappings_candidates"][key] = [{
            "candidate": best_cand,
            "score": best_score
        }]

        selected["debug_info"][key] = {
            "considered_candidates": collected,
            "avg_similarity_scores": dict(zip(sentences, avg_sim)),
            "selected_candidate": best_cand
        }

    return selected

def append_combined_results_to_exports(csv_path, json_path, selected_count, voted_llm, reference, sample_id):

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        header = reader[0]
        rows = reader[1:]

    def pct(x, total):
        return f"{round(x / total * 100, 2)}%" if total else "0%"

    total = len(reference.get("mappings", {}))
    models_to_add = {
        "MajorityVote_Weighted": voted_llm
    }

    top_m = [1,3,5,10]

    for m in top_m:
        models_to_add[f"CandidateSelectionCount{m}"] = selected_count[m]

    new_rows = []
    for name, model in models_to_add.items():
        res1 = evaluate_top_k(1, reference, model)
        res3 = evaluate_top_k(3, reference, model)
        res5 = evaluate_top_k(5, reference, model)
        res10 = evaluate_top_k(10, reference, model)

        new_rows.append([
            sample_id, name,
            res1["hits@1"], pct(res1["hits@1"], total),
            res3["hits@3"], pct(res3["hits@3"], total),
            res5["hits@5"], pct(res5["hits@5"], total),
            res10["hits@10"], pct(res10["hits@10"], total),
            res1["not_hits@1"], res3["not_hits@3"], res5["not_hits@5"], res10["not_hits@10"],
            res1["no_mappings_provided"], pct(res1["no_mappings_provided"], total)
        ])


    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)


    with open(json_path, "r", encoding="utf-8") as f:
        jdata = json.load(f)

    if "evaluated_models" not in jdata:
        jdata["evaluated_models"] = {}

    for name, model in models_to_add.items():
        jdata["evaluated_models"][name] = model

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(jdata, f, indent=4, ensure_ascii=False)




def calculate_agent_weights(base_output_dir: Path):

    scores = {}
    weights_factors = [0.4, 0.3, 0.2, 0.1]

    for csv_file in base_output_dir.rglob("*_mapping_results.csv"):
        with open(csv_file, "r", encoding="utf-8") as f:
            next(f)  # Header überspringen
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 12:
                    continue
                agent = parts[1]
                if agent in ("CandidateSelection", "MajorityVote"):
                    continue

                try:
                    h1 = float(parts[3].replace("%", "")) / 100
                    h3 = float(parts[5].replace("%", "")) / 100
                    h5 = float(parts[7].replace("%", "")) / 100
                    h10 = float(parts[9].replace("%", "")) / 100
                except Exception:
                    continue

                score = (
                    weights_factors[0]*h1 +
                    weights_factors[1]*h3 +
                    weights_factors[2]*h5 +
                    weights_factors[3]*h10
                )
                scores[agent] = scores.get(agent, 0) + score


    sorted_agents = sorted(scores.items(), key=lambda x: x[1])
    agent_weights = {}
    for i, (agent, _) in enumerate(sorted_agents):
        agent_weights[agent] = 1 - (10 ** -(i + 5))  # z. B. 1-1e-5, 1-1e-6, ...

    return agent_weights


def vote_candidate_selection_entry_point(    vcslam_path: str,
    sample_ids: list[str],
    historical_ids: list[str],
    base_export_dir: str = r"C:\exports",
    calc_weights: bool = False
):
    logger.info(f"Starting vote and candidate selection processing.\nSids: {sample_ids}\n\nExport dir: {base_export_dir}\n\nCalc weights: {calc_weights}")
    base_output_dir = Path(base_export_dir)
    env = load_environment(vcslam_path)
    ontology_str = env["ontology_path"].read_text(encoding="utf-8")


    historical = []
    for hid in historical_ids:
        try:
            j, d, u, m = load_sample(env["base_dir"], hid)
            historical.append({"json_data": j, "documentation": d, "unmapped": u, "mapping": m, "sid": hid})
        except FileNotFoundError:
            logger.warning(f"Error with historical sample: {hid}")

    agents = build_agents(env)
    agent_weights = {}
    if calc_weights:

        agent_weights = calculate_agent_weights(base_output_dir)


        for agent in agents:
            if agent.name in agent_weights:
                agent.set_weight(agent_weights[agent.name])
    else:
        for agent in AGENTS_CONFIG:
            agent_weights[agent["name"]] = agent["weight"]

    for sid in sample_ids:
        logger.info(f"Starting process for {sid}")
        sample_output_dir = base_output_dir / sid
        json_path = sample_output_dir / f"{sid}_mapping_results.json"
        csv_path = sample_output_dir / f"{sid}_mapping_results.csv"

        if not json_path.exists():
            continue


        with open(json_path, "r", encoding="utf-8") as jf:
            data = json.load(jf)


        json_data, documentation, unmapped, reference = load_sample(env["base_dir"], sid)
        filtered_historical = [h for h in historical if h["sid"] != sid]


        agent_results = {}
        for a in [ag.name for ag in agents]:
            if a in data.get("evaluated_models", {}):
                agent_results[a] = {
                    "prefix": data["evaluated_models"][a].get("prefix", ""),
                    "mappings_candidates": data["evaluated_models"][a].get("mappings_candidates", {})
                }

        combined = merge_all_candidates(agent_results)


        top_m_values = [1, 3, 5, 10]

        weighted_variants = {}

        for m in top_m_values:
            weighted_variants[m] = candidate_selection_count(agent_results, agent_weights, top_m=m)
        voted_llm = majority_voting_weighted(agents, combined, json_data, documentation, filtered_historical)

        append_combined_results_to_exports(
            csv_path=csv_path,
            json_path=json_path,
            selected_count=weighted_variants,
            voted_llm=voted_llm,
            reference=reference,
            sample_id=sid
        )
        logger.info(f"Finished process for {sid}")

    logger.info("Evaluation Finished")

def main(
    vcslam_path: str,
    sample_ids: list[str],
    historical_ids: list[str],
    base_export_dir: str = r"exports",
    calc_weights: bool = False,
    voting_and_candidate: bool = False,
    evaluation_run = False
):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_dir = Path(base_export_dir) / timestamp
    base_output_dir.mkdir(parents=True, exist_ok=True)


    logger.info(f"Results will be saved in: {base_output_dir}")

    env = load_environment(vcslam_path)
    ontology_str = env["ontology_path"].read_text(encoding="utf-8")

    historical = []
    for hid in historical_ids:
        try:
            j, d, u, m = load_sample(env["base_dir"], hid)
            historical.append({"json_data": j, "documentation": d, "unmapped": u, "mapping": m, "sid": hid})
        except FileNotFoundError:
            logger.warning(f"Historical Sample {hid} not found – skip")

    agents = build_agents(env)
    for sid in sample_ids:

        sample_output_dir = base_output_dir / sid
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = sample_output_dir / f"{sid}_mapping_results.csv"

        json_data, documentation, unmapped, reference = load_sample(env["base_dir"], sid=sid)
        filtered_historical = [h for h in historical if h["sid"] != sid]

        agent_results = run_agents_on_sample(
            agents, ontology_str, json_data, documentation, unmapped, filtered_historical
        )

        logger.info(f"📊 Evaluating individual agents (Top-K) for {sid} ...")
        for agent_name, model_data in agent_results.items():
            for k in [1, 3, 5, 10]:
                result = evaluate_top_k(k, reference, model_data)

        combined = merge_all_candidates(agent_results)


        agent_weights = {agent.name: agent.get_weight() for agent in agents}

        weighted_variants = {}
        semantic_variants = {}
        voted_llm = {"prefix": combined["prefix"], "mappings_candidates": {}}

        evaluate_and_export(
            sid,
            reference,
            agent_results,
            semantic_variants,
            weighted_variants,
            voted_llm,
            str(output_csv),
            vote_and_selection=voting_and_candidate
        )


        logger.info("Main run done")

        if evaluation_run:
            try:
                logger.info("Starting global evaluations")
                evaluations.run(base_output_dir)
            except Exception as e:
                logger.error(f"Exception during evaluation: {e}")
