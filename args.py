import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4.1")
    parser.add_argument("--eps", type=float, default=3)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--text", type=str, default="logic")
    parser.add_argument("--task", type=str, default="strategyqa")
    parser.add_argument(
        "--raw_path",
        type=str,
        default="output/logic.csv",
    )
    return parser
