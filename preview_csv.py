#!/usr/bin/env python3
"""读取 CSV 文件的前 10 条和尾 10 条记录。"""

import argparse
import csv
import sys
from pathlib import Path


def preview_csv(filepath: str, n: int = 10) -> None:
    """读取并打印 CSV 文件的前 N 条和尾 N 条记录。

    Args:
        filepath: CSV 文件路径。
        n: 显示的头尾记录数，默认 10。
    """
    path = Path(filepath)
    if not path.exists():
        print(f"错误: 文件不存在 — {filepath}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = list(csv.reader(f))

    if not reader:
        print("文件为空。")
        return

    header = reader[0]
    rows = reader[1:] if len(reader) > 1 else []

    total = len(rows)

    # 表头
    print("=" * 60)
    print(f"文件: {path.name}  |  总记录数: {total}")
    print("=" * 60)

    def print_rows(title: str, data: list, start_idx: int = 0) -> None:
        print(f"\n--- {title} ---")
        # 打印列名
        print(", ".join(header))
        print("-" * 40)
        for i, row in enumerate(data):
            print(f"[{start_idx + i + 1:>{len(str(total))}}]  {', '.join(row)}")

    if total <= n * 2:
        # 数据不多，全部打印
        print_rows("全部记录", rows)
    else:
        print_rows(f"前 {n} 条", rows[:n], start_idx=0)
        print(f"\n  ... 省略中间 {total - n * 2} 条 ...")
        print_rows(f"尾 {n} 条", rows[-n:], start_idx=total - n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取 CSV 文件的前 N 条和尾 N 条记录"
    )
    parser.add_argument("filepath", help="CSV 文件路径")
    parser.add_argument(
        "-n", "--num", type=int, default=10,
        help="显示的头尾记录数（默认 10）"
    )
    args = parser.parse_args()
    preview_csv(args.filepath, args.n)


if __name__ == "__main__":
    preview_csv(filepath=r"C:\Users\Administrator\Desktop\oai_mcp_linux\src\datasets\labels_train.csv")
