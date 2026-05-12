import os
import csv
import json
import hashlib
from collections import Counter, defaultdict
import asyncio
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from args import get_parser
from logic_extract import logic_extract_one
from pseudo_example import pseudo_example_one
from problem_solve import problem_solve

parser = get_parser()
args = parser.parse_args()

INPUT_CSV = "../data/strategyQA/strategyQA.csv"
OUTPUT_DIR = "output"
REINDEXED_INPUT_CSV = os.path.join(OUTPUT_DIR, "strategyqa_reindexed.csv")
LOGIC_CSV = os.path.join(OUTPUT_DIR, "logic.csv")
LOGIC_DP_CSV = os.path.join(OUTPUT_DIR, "logic_dp.csv")
STORY_CSV = os.path.join(OUTPUT_DIR, "story.csv")

LOGIC_EPS = float(3)
MAPPING_EPS = float(1)
MAX_WORKERS = 16
MAX_SAMPLES = 203

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LDP_DIR = os.getenv("LDP_DIR", os.path.join(PROJECT_ROOT, "ldp"))
LDP_EMBEDDING_TYPE = os.getenv("LDP_EMBEDDING_TYPE", "glove.6B.300d")
LDP_MAPPING_STRATEGY = os.getenv("LDP_MAPPING_STRATEGY", "aggressive")
LDP_TOP_K = int(os.getenv("LDP_TOP_K", "20"))
LDP_SAVE_STOP_WORDS = os.getenv("LDP_SAVE_STOP_WORDS", "true").strip().lower() not in {"0", "false", "no"}
LDP_USE_CACHE = os.getenv("LDP_USE_CACHE", "true").strip().lower() not in {"0", "false", "no"}
LDP_CACHE_DIR = os.getenv("LDP_CACHE_DIR", os.path.join(OUTPUT_DIR, "ldp_cache"))

_LDP_EMBEDDING_CACHE = None
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_reindexed_qid(position: int, width: int) -> str:
    return f"1-{position:0{width}d}"


def prepare_reindexed_dataset(input_csv: str, output_csv: str):
    ensure_dir(os.path.dirname(output_csv))
    data = pd.read_csv(input_csv).copy()

    width = max(3, len(str(len(data))))
    data.insert(0, "original_qid", data["qid"].astype(str))
    data["qid"] = [build_reindexed_qid(i, width) for i in range(1, len(data) + 1)]

    data.to_csv(output_csv, index=False)
    print(f"[INFO] Reindexed dataset saved to {output_csv}")


def init_logic_csv():
    ensure_dir(OUTPUT_DIR)
    with open(LOGIC_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "qid",
                "original_qid",
                "facts",
                "question",
                "answer",
                "query",
                "logic",
                "mapping",
            ],
        )
        writer.writeheader()


def load_tasks(input_csv: str, max_samples: int) -> List[Tuple[str, str, str, str, str]]:
    tasks = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_samples is not None and idx >= max_samples:
                break
            tasks.append(
                (
                    str(row["qid"]).strip(),
                    str(row["original_qid"]).strip(),
                    str(row["facts"]).strip(),
                    str(row["question"]).strip(),
                    str(row["answer"]).strip(),
                )
            )
    return tasks


def run_logic_extraction(tasks):
    print(f"[INFO] Launching {len(tasks)} logic extraction jobs")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(logic_extract_one, qid, original_qid, facts, question, answer)
            for qid, original_qid, facts, question, answer in tasks
        ]
        for _ in as_completed(futures):
            pass
    print("[INFO] logic.csv safely saved.")


def run_pseudo_example():
    print("[INFO] Generating pseudo examples")
    pseudo_example_one(
        input_csv=LOGIC_DP_CSV,
        output_csv=STORY_CSV,
        model="gpt-4.1",
        max_workers=MAX_WORKERS,
    )
    print("[INFO] story.csv generated")


def run_problem_solve():
    print("[INFO] Running problem solver")
    asyncio.run(problem_solve(max_samples=MAX_SAMPLES))
    print("[INFO] problem solving finished")


def is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def load_stop_words() -> set:
    fallback = {
        "a", "an", "the", "and", "or", "but", "if", "then", "is", "are",
        "was", "were", "be", "been", "being", "to", "of", "in", "on", "for",
        "with", "as", "by", "at", "from", "that", "this", "these", "those",
        "it", "its", "they", "them", "their", "he", "she", "his", "her",
        "we", "you", "i", "not", "no", "yes", "do", "does", "did", "can",
        "could", "would", "should", "will", "shall", "than", "which", "what",
        "who", "whom", "where", "when", "why", "how",
    }
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except Exception:
        return fallback


def load_ldp_embeddings():
    global _LDP_EMBEDDING_CACHE
    if _LDP_EMBEDDING_CACHE is not None:
        return _LDP_EMBEDDING_CACHE

    embedding_path = os.path.join(LDP_DIR, "embeddings", f"{LDP_EMBEDDING_TYPE}.txt")
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"LDP embedding file not found: {embedding_path}")

    print(f"[LDP] Loading embeddings from {embedding_path}")
    embeddings = []
    idx2word = []
    word2idx = {}

    if LDP_EMBEDDING_TYPE == "glove_840B-300d":
        with open(embedding_path, "r", encoding="utf-8") as f:
            word_embeddings = json.load(f)
        for idx, (word, vector) in enumerate(word_embeddings.items()):
            idx2word.append(word)
            word2idx[word] = idx
            embeddings.append(vector)
        embeddings = np.asarray(embeddings, dtype=np.float32)
    else:
        with open(embedding_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= 2:
                    continue
                word = parts[0]
                try:
                    vector = [float(num) for num in parts[1:]]
                except ValueError:
                    continue
                word2idx[word] = len(idx2word)
                idx2word.append(word)
                embeddings.append(vector)
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        embeddings = embeddings / norm

    idx2word = np.asarray(idx2word)
    _LDP_EMBEDDING_CACHE = embeddings, idx2word, word2idx
    print(f"[LDP] Loaded {len(idx2word)} embeddings")
    return _LDP_EMBEDDING_CACHE


def collect_word_frequency(data: pd.DataFrame, columns: List[str]) -> List[str]:
    counter = Counter()
    for column in columns:
        if column not in data.columns:
            continue
        for value in data[column].fillna("").astype(str):
            if column == "mapping":
                value = value.replace('"', " ")
            counter.update(value.split())
    return [word for word, _ in counter.most_common()]


def normalize_scores(values, *, smaller_is_better: bool) -> List[float]:
    values = np.asarray(values, dtype=np.float64)
    span = float(values.max() - values.min())
    if span == 0.0:
        return [0.0 for _ in values]
    if smaller_is_better:
        return [-(x - values.min()) / span for x in values]
    return [(x - values.min()) / span for x in values]


def softmax_prob(scores: List[float], eps: float) -> List[float]:
    logits = np.asarray([(eps * score / 2.0) for score in scores], dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return probs.tolist()


def ldp_cache_paths(eps: float, word_freq: List[str]) -> Tuple[str, str]:
    vocab_sig = hashlib.md5("\n".join(word_freq).encode("utf-8")).hexdigest()[:12]
    safe_eps = str(eps).replace(".", "p")
    cache_dir = os.path.join(LDP_CACHE_DIR, LDP_EMBEDDING_TYPE, LDP_MAPPING_STRATEGY)
    ensure_dir(cache_dir)
    prefix = f"eps_{safe_eps}_top_{LDP_TOP_K}_vocab_{vocab_sig}"
    return (
        os.path.join(cache_dir, f"sim_word_dict_{prefix}.json"),
        os.path.join(cache_dir, f"p_dict_{prefix}.json"),
    )


def get_ldp_mapping(word_freq: List[str], eps: float):
    sim_cache, p_cache = ldp_cache_paths(eps, word_freq)
    if LDP_USE_CACHE and os.path.exists(sim_cache) and os.path.exists(p_cache):
        with open(sim_cache, "r", encoding="utf-8") as f:
            sim_word_dict = json.load(f)
        with open(p_cache, "r", encoding="utf-8") as f:
            p_dict = json.load(f)
        print(f"[LDP] Loaded cached mapping for eps={eps}: {sim_cache}")
        return sim_word_dict, p_dict

    embeddings, idx2word, word2idx = load_ldp_embeddings()
    top_k = min(LDP_TOP_K, len(idx2word))
    sim_word_dict = {}
    p_dict = {}
    word_hash = defaultdict(str)

    use_euclidean = LDP_EMBEDDING_TYPE == "glove_840B-300d"
    work_embeddings = embeddings.copy() if LDP_MAPPING_STRATEGY == "conservative" else embeddings
    embeddings_t = work_embeddings.T

    print(f"[LDP] Building mapping: eps={eps}, top_k={top_k}, strategy={LDP_MAPPING_STRATEGY}")
    for word in word_freq:
        if word not in word2idx or word in word_hash:
            continue

        word_idx = word2idx[word]
        word_vec = work_embeddings[word_idx]

        if use_euclidean:
            dist = np.linalg.norm(work_embeddings - word_vec.reshape(1, -1), axis=1)
            index_list = dist.argsort()[:top_k]
        else:
            sims = np.dot(word_vec, embeddings_t)
            index_list = sims.argsort()[::-1][:top_k]

        word_list = [str(idx2word[idx]) for idx in index_list]
        embedding_list = np.asarray([work_embeddings[idx] for idx in index_list])

        def assign_mapping(source_word: str):
            source_vec = work_embeddings[word2idx[source_word]]
            if use_euclidean:
                distances = np.linalg.norm(embedding_list - source_vec.reshape(1, -1), axis=1)
                scores = normalize_scores(distances, smaller_is_better=True)
            else:
                similarities = np.dot(source_vec, embedding_list.T)
                scores = normalize_scores(similarities, smaller_is_better=False)
            sim_word_dict[source_word] = word_list
            p_dict[source_word] = softmax_prob(scores, eps)

        if LDP_MAPPING_STRATEGY == "aggressive":
            assign_mapping(word)
        else:
            for candidate in word_list:
                if candidate not in word_hash and candidate in word2idx:
                    word_hash[candidate] = word
                    assign_mapping(candidate)
            if LDP_MAPPING_STRATEGY == "conservative":
                fill_value = 1e9 if use_euclidean else 0.0
                work_embeddings[index_list, :] = fill_value
                embeddings_t = work_embeddings.T

    if LDP_USE_CACHE:
        with open(sim_cache, "w", encoding="utf-8") as f:
            json.dump(sim_word_dict, f, ensure_ascii=False, indent=2)
        with open(p_cache, "w", encoding="utf-8") as f:
            json.dump(p_dict, f, ensure_ascii=False, indent=2)
        print(f"[LDP] Cached mapping to {sim_cache}")

    return sim_word_dict, p_dict


def ldp_perturb_text(text: str, sim_word_dict: Dict[str, List[str]], p_dict: Dict[str, List[float]], stop_words: set) -> str:
    new_record = []
    for word in str(text).split():
        if (LDP_SAVE_STOP_WORDS and word in stop_words) or (word not in sim_word_dict):
            if is_number(word):
                try:
                    word = str(round(float(word)) + np.random.randint(1000))
                except Exception:
                    pass
            new_record.append(word)
        else:
            new_word = np.random.choice(sim_word_dict[word], 1, p=p_dict[word])[0]
            new_record.append(str(new_word))
    return " ".join(new_record)


def run_dp_perturbation():
    data = pd.read_csv(args.raw_path)
    print("[INFO] Using LDP perturbation instead of DP server")
    print(f"[INFO] logic eps = {LOGIC_EPS}; mapping eps = {MAPPING_EPS}")
    print(
        f"[INFO] LDP config: embedding={LDP_EMBEDDING_TYPE}, "
        f"strategy={LDP_MAPPING_STRATEGY}, top_k={LDP_TOP_K}, "
        f"save_stop_words={LDP_SAVE_STOP_WORDS}"
    )
    print(f"length = {len(data)}")

    word_freq = collect_word_frequency(data, [args.text, "mapping"])
    stop_words = load_stop_words()

    logic_sim_word_dict, logic_p_dict = get_ldp_mapping(word_freq, LOGIC_EPS)
    if MAPPING_EPS == LOGIC_EPS:
        mapping_sim_word_dict, mapping_p_dict = logic_sim_word_dict, logic_p_dict
    else:
        mapping_sim_word_dict, mapping_p_dict = get_ldp_mapping(word_freq, MAPPING_EPS)

    for i in range(len(data)):
        print(f"[LDP] Processing sample {i}")

        raw_logic = str(data.loc[i, args.text])
        data.loc[i, args.text] = ldp_perturb_text(
            raw_logic,
            logic_sim_word_dict,
            logic_p_dict,
            stop_words,
        )

        raw_mapping = str(data.loc[i, "mapping"]).replace('"', " ")
        data.loc[i, "mapping"] = ldp_perturb_text(
            raw_mapping,
            mapping_sim_word_dict,
            mapping_p_dict,
            stop_words,
        )

    perturbed_path = LOGIC_DP_CSV
    ensure_dir(os.path.dirname(perturbed_path))
    data.to_csv(perturbed_path, index=False)
    print(f"[INFO] LDP result saved to {perturbed_path}")

def main():
    prepare_reindexed_dataset(INPUT_CSV, REINDEXED_INPUT_CSV)
    init_logic_csv()
    tasks = load_tasks(REINDEXED_INPUT_CSV, MAX_SAMPLES)
    run_logic_extraction(tasks)
    run_dp_perturbation()
    run_pseudo_example()
    run_problem_solve()


if __name__ == "__main__":
    main()
