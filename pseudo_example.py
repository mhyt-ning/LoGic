import csv
import time
import threading
import os
import re
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv(
        "OPENAI_API_KEY",
        "",
    ),
    base_url="",
)

SYSTEM_PROMPT = """
You are an expert structure-consistent analogical generator. You will be provided with an abstract logic
skeleton and a corresponding entity mapping. Your task consists of two sequential steps

1) Substitute the placeholders in the skeleton with the provided entities to construct a coherent, natural language query. You must strictly preserve the original logical topology, operations, and constraints
without adding or altering any logic;
2) Solve the newly constructed query with a concise step-by-step reasoning trajectory

Requirements:
- Use only the logical structure provided.
- Preserve the original reasoning pattern, constraints, and truth conditions.
- You may introduce fresh generic entities or settings, but they must be simple, concrete, and easy to reason about.
- Do not reconstruct hidden private details or assume missing facts.
- Do not add constraints that change the answer.
- Prefer short, clean, and structurally faithful examples over creative but noisy ones.
- The final answer must be either True or False.
"""

USER_TEMPLATE = """
Abstract logic:
{LOGIC}

Entity mapping:
{MAPPING}

Generate a natural language problem that instantiates the logic,
and then solve it.

Output format (STRICT):

Pseudo Problem:
...

Reasoning:
Step 1: ...
Step 2: ...
...

Final Answer:
True or False
"""


def parse_pseudo_problem_and_cot(text: str) -> Tuple[str, str]:
    if not text:
        return "", ""

    t = text.replace("\r\n", "\n")
    prob_match = re.search(
        r"Pseudo Problem:\s*(.*?)(?:\n\s*Reasoning:|\Z)",
        t,
        re.DOTALL | re.IGNORECASE,
    )
    cot_match = re.search(r"Reasoning:\s*(.*)", t, re.DOTALL | re.IGNORECASE)

    pseudo_problem = prob_match.group(1).strip() if prob_match else t.strip()
    cot = cot_match.group(1).strip() if cot_match else ""
    return pseudo_problem, cot


def generate_story(
    logic_text: str,
    mapping_text: str,
    *,
    model: str = "gpt-4.1",
    retries: int = 5,
    sleep_s: float = 0.8,
    max_tokens: int = 800,
) -> str:
    last_err: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": USER_TEMPLATE.format(
                            LOGIC=logic_text,
                            MAPPING=mapping_text,
                        ),
                    },
                ],
            )

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response from model.")
            return text
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(sleep_s * attempt)
                continue
            raise RuntimeError(
                f"Failed after {retries} retries. Last error: {last_err}"
            ) from e

    raise RuntimeError(f"Unexpected failure: {last_err}")


csv_lock = threading.Lock()


def append_csv_row(path: str, row: Dict[str, str], fieldnames: List[str]):
    with csv_lock:
        with open(path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)


def process_one(
    qid: str,
    original_qid: str,
    facts: str,
    question: str,
    answer: str,
    query: str,
    logic: str,
    mapping_text: str,
    *,
    output_csv: str,
    model: str,
    fieldnames: List[str],
):
    start = time.time()
    try:
        raw_story = generate_story(logic, mapping_text, model=model)
        pseudo_problem, cot = parse_pseudo_problem_and_cot(raw_story)

        row = {
            "qid": qid,
            "original_qid": original_qid,
            "facts": facts,
            "question": question,
            "answer": answer,
            "query": query,
            "logic": logic,
            "mapping": mapping_text,
            "pseudo_problem": pseudo_problem,
            "cot": cot,
        }
        append_csv_row(output_csv, row, fieldnames)
        print(
            f"[OK] qid={qid} | {time.time() - start:.2f}s | "
            f"{pseudo_problem[:60].replace(chr(10), ' ')}"
        )
    except Exception as e:
        row = {
            "qid": qid,
            "original_qid": original_qid,
            "facts": facts,
            "question": question,
            "answer": answer,
            "query": query,
            "logic": logic,
            "mapping": mapping_text,
            "pseudo_problem": "",
            "cot": f"ERROR: {repr(e)}",
        }
        append_csv_row(output_csv, row, fieldnames)
        print(f"[FAIL] qid={qid} | {time.time() - start:.2f}s | {repr(e)}")


def pseudo_example_one(
    input_csv: str = "logic.csv",
    output_csv: str = "story.csv",
    model: str = "gpt-4.1",
    max_workers: int = 16,
):
    fieldnames = [
        "qid",
        "original_qid",
        "facts",
        "question",
        "answer",
        "query",
        "logic",
        "mapping",
        "pseudo_problem",
        "cot",
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    tasks = []
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = (row.get("qid") or "").strip()
            original_qid = (row.get("original_qid") or "").strip()
            facts = (row.get("facts") or "").strip()
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            query = (row.get("query") or "").strip()
            logic = (row.get("logic") or "").strip()
            mapping_text = (row.get("mapping") or "").strip()

            if not (qid and logic and mapping_text):
                continue

            tasks.append((qid, original_qid, facts, question, answer, query, logic, mapping_text))

    print(f"Loaded {len(tasks)} items from {input_csv}.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_one,
                qid,
                original_qid,
                facts,
                question,
                answer,
                query,
                logic,
                mapping_text,
                output_csv=output_csv,
                model=model,
                fieldnames=fieldnames,
            )
            for qid, original_qid, facts, question, answer, query, logic, mapping_text in tasks
        ]

        for _ in as_completed(futures):
            pass

    print("All story generations completed safely.")


if __name__ == "__main__":
    pseudo_example_one(
        input_csv="logic.csv",
        output_csv="story.csv",
        model="gpt-4.1",
        max_workers=16,
    )
