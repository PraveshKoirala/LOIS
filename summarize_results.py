from __future__ import annotations

import argparse
import glob

import pandas as pd


def list_csv_files(prefix: str) -> list[str]:
    files = glob.glob(f"{prefix}-*.csv")
    return sorted(files, key=lambda path: int(path.split("-")[-1].split(".")[0]))


def process_csv(prefix: str, column_operations: list[tuple[str, str]]) -> None:
    for file_path in list_csv_files(prefix):
        frame = pd.read_csv(file_path)
        values = []
        for column, operation in column_operations:
            if column not in frame.columns:
                continue
            if operation == "mean":
                values.append(f"{column}={frame[column].mean():.2f}")
            elif operation == "range":
                values.append(f"{column}=[{frame[column].min():.2f}-{frame[column].max():.2f}]")
        print(", ".join(values), file_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize experiment CSVs for the LOIS paper.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--columns",
        nargs="+",
        default=["t:mean", "PoD:mean", "PoA:mean"],
        help="Pairs in the form column:operation. Supported operations: mean, range.",
    )
    args = parser.parse_args()
    operations = [tuple(item.split(":", 1)) for item in args.columns]
    process_csv(args.prefix, operations)


if __name__ == "__main__":
    main()
