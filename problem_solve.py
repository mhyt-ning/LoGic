import asyncio
import pandas as pd
import re
import httpx
from typing import Optional, Dict, Any

from openai import AsyncOpenAI


BASE_URL = ""
API_KEY = "EMPTY"
MODEL_NAME = "Qwen2.5-3B-Instruct"

INPUT_CSV = "output/strategyqa_reindexed.csv"
TEMPLATE_CSV = "output/story.csv"
OUTPUT_CSV = "output/final_answers.csv"

MAX_CONCURRENCY = 16

http_client = httpx.AsyncClient(trust_env=False)
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=600.0,
    http_client=http_client,
)

SYSTEM_PROMPT = """
You are a structure-aligned reasoning executor. You will be given a private query and a structure-consistent
analogical example (including its reasoning trajectory). Your task is to solve the private query by transferring 
the reasoning pattern from the analogical example.

Requirements:
- Follow the reasoning pattern in the analogical reasoning trajectory.
- Apply the same reasoning steps to the private query using its own entities.
- Verify each reasoning step against the current facts, not the example facts.
- If the example is imperfect or only partially useful, still reason from the current problem.
- Output exactly two sections: Reasoning, then Final Answer.
- Final Answer must be either True or False.
"""


def normalize_bool(value) -> Optional[bool]:
    if value is None:
        return None

    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def build_user_prompt(
    facts: str,
    question: str,
    example: Optional[Dict[str, str]],
) -> str:
    example_block = ""
    if example:
        example_block = f"""
Solved Example (for learning strategy only):

Problem:
{example['fake_query']}

Reasoning:
{example['solve_template']}

---
"""

    return f"""{example_block}

Now solve the following StrategyQA item.

Facts:
{facts}

Question:
{question}

Output format (STRICT):

Reasoning:
Step 1: ...
Step 2: ...

Final Answer: True or False
"""


def parse_answer(text: str) -> Optional[bool]:
    if not text:
        return None

    lowered = text.lower()
    if "final answer" in lowered:
        lowered = lowered.split("final answer", 1)[-1]

    if re.search(r"\btrue\b", lowered):
        return True
    if re.search(r"\bfalse\b", lowered):
        return False
    return None


def parse_reasoning(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n")
    match = re.search(
        r"Reasoning:\s*(.*?)(?:\n\s*Final Answer\s*:|\Z)",
        normalized,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def load_templates(path: str) -> Dict[str, Dict[str, str]]:
    print("[INFO] Loading story templates...")
    df = pd.read_csv(path)

    templates = {}
    for _, row in df.iterrows():
        if pd.isna(row["cot"]) or pd.isna(row["pseudo_problem"]):
            continue

        qid = str(row["qid"])
        templates[qid] = {
            "fake_query": str(row["pseudo_problem"]),
            "solve_template": str(row["cot"]),
        }

    print(f"[INFO] Loaded {len(templates)} templates")
    return templates


async def solve_one(
    sem: asyncio.Semaphore,
    templates: Dict[str, Dict[str, str]],
    qid: str,
    facts: str,
    question: str,
    gold: Optional[bool],
) -> Dict[str, Any]:
    example = templates.get(qid)
    if example is None:
        print(f"[WARN] No template for {qid}")

    prompt = build_user_prompt(facts, question, example)

    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            text = (resp.choices[0].message.content or "").strip()
            pred = parse_answer(text)
            reasoning = parse_reasoning(text)
            print(f"[{qid}] Pred={pred} | Gold={gold}")
            return {
                "pred_bool": pred,
                "reasoning": reasoning,
                "raw_response": text,
                "solver_error": "",
            }
        except Exception as e:
            print(f"[{qid}] ERROR: {e}")
            return {
                "pred_bool": None,
                "reasoning": "",
                "raw_response": "",
                "solver_error": repr(e),
            }


async def problem_solve(max_samples: int = None):
    print("[INFO] Loading dataset...")
    df = pd.read_csv(INPUT_CSV).copy()
    if max_samples is not None:
        df = df.head(max_samples).copy()

    required = {"qid", "original_qid", "facts", "question", "answer"}
    assert required.issubset(df.columns)

    df["qid"] = df["qid"].astype(str)
    df["original_qid"] = df["original_qid"].astype(str)
    df["answer_bool"] = df["answer"].apply(normalize_bool)
    templates = load_templates(TEMPLATE_CSV)

    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []

    for _, row in df.iterrows():
        tasks.append(
            solve_one(
                sem,
                templates,
                row["qid"],
                str(row["facts"]),
                str(row["question"]),
                row["answer_bool"],
            )
        )

    print(f"[INFO] Launching {len(tasks)} jobs")
    results = await asyncio.gather(*tasks)

    df["pred_bool"] = [item["pred_bool"] for item in results]
    df["reasoning"] = [item["reasoning"] for item in results]
    df["raw_response"] = [item["raw_response"] for item in results]
    df["solver_error"] = [item["solver_error"] for item in results]
    df["pred"] = df["pred_bool"].map(
        lambda x: "True" if x is True else "False" if x is False else None
    )
    df["is_correct"] = (
        (df["pred_bool"].notna()) &
        (df["pred_bool"] == df["answer_bool"])
    ).astype(int)

    print("\n========== Evaluation ==========")
    print(f"Total    : {len(df)}")
    print(f"Coverage : {df['pred_bool'].notna().mean():.4f}")
    print(f"Accuracy : {df['is_correct'].mean():.4f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(problem_solve(max_samples=10))
