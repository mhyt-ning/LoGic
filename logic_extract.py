import csv
import json
import re
import time
import threading
from typing import Dict, Tuple

from openai import OpenAI
import os

LOGIC_CSV_PATH = "output/logic.csv"

SYSTEM_PROMPT = """You are a Logic Skeleton Extractor. Given a user query, your task is to convert it into a logical skeleton
together with a complete entity mapping from abstract symbols to concrete items.

[Abstraction Rules]
- Replace ALL entities with symbols (A, B, C, …).
- Replace ALL accurate numbers with symbols (m, n, k, …).

Output rules:
- End with exactly one abstract Question line.
- The skeleton should be sufficient to determine the decision.
"""

USER_TEMPLATE = """Rewrite the following StrategyQA item as a logical skeleton and a mapping.

Facts:
{FACTS}

Question:
{QUESTION}

Output format (STRICT):

[Mapping]
- A = ...
- B = ...
- m = ...
- n = ...

[Abstract Facts]
- ...
- ...

[Decision Question]
- ...
"""


LOCAL_MODEL = ""
client = OpenAI(
    api_key="EMPTY",
    base_url="",
    timeout=600.0,
)

_SYMBOL_RE = re.compile(r"(?<![A-Za-z])([A-Z]|[m-z])(?![A-Za-z])")
logic_lock = threading.Lock()


def extract_raw_logic(facts: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=LOCAL_MODEL,
        temperature=0.0,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(FACTS=facts, QUESTION=question),
            },
        ],
    )

    text = resp.choices[0].message.content
    if not text:
        raise ValueError("Empty response from model.")
    return text.strip()


def extract_symbols_from_logic(logic_text: str) -> set:
    return set(_SYMBOL_RE.findall(logic_text))


def split_mapping_and_logic(text: str) -> Tuple[Dict[str, str], str]:
    lines = text.splitlines()
    mapping = {}
    logic_lines = []
    in_mapping = False

    for line in lines:
        s = line.strip()
        if s.startswith("[Mapping]"):
            in_mapping = True
            continue

        if in_mapping:
            if not s or s.startswith("["):
                in_mapping = False
            else:
                if "=" in s:
                    k, v = s.lstrip("-").split("=", 1)
                    mapping[k.strip()] = v.strip()
                continue

        logic_lines.append(line)

    return mapping, "\n".join(logic_lines).strip()


def validate_mapping_alignment(mapping: Dict[str, str], logic_text: str) -> None:
    used = extract_symbols_from_logic(logic_text)
    missing = used - set(mapping.keys())
    if missing:
        raise ValueError(f"Undefined symbols: {missing}")


def extract_logic_and_mapping(facts: str, question: str) -> Tuple[str, Dict[str, str]]:
    raw = extract_raw_logic(facts, question)
    mapping, logic = split_mapping_and_logic(raw)

    for sym in extract_symbols_from_logic(logic):
        mapping.setdefault(sym, "__UNBOUND__")

    validate_mapping_alignment(mapping, logic)
    return logic, mapping


def append_logic_row(row: Dict):
    os.makedirs(os.path.dirname(LOGIC_CSV_PATH), exist_ok=True)
    with logic_lock:
        with open(LOGIC_CSV_PATH, "a", encoding="utf-8", newline="") as f:
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
            writer.writerow(row)


def logic_extract_one(qid: str, original_qid: str, facts: str, question: str, answer: str):
    query = f"Facts: {facts}\nQuestion: {question}"
    start = time.time()

    try:
        logic, mapping = extract_logic_and_mapping(facts, question)
        append_logic_row(
            {
                "qid": qid,
                "original_qid": original_qid,
                "facts": facts,
                "question": question,
                "answer": answer,
                "query": query,
                "logic": logic,
                "mapping": json.dumps(mapping, ensure_ascii=False),
            }
        )
        print(f"[OK] {qid} | {time.time() - start:.2f}s")
    except Exception as e:
        print(f"[FAIL] {qid} | {time.time() - start:.2f}s | {e}")
