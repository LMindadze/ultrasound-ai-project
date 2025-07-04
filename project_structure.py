#!/usr/bin/env python3
import os
from pathlib import Path

def print_tree(path: Path, prefix: str = ""):
    # Gather directories and .py files
    children = [p for p in path.iterdir() if p.is_dir() or (p.is_file() and p.suffix == '.py')]
    children = sorted(children, key=lambda p: p.name)
    total = len(children)
    for idx, child in enumerate(children):
        connector = "└── " if idx == total - 1 else "├── "
        if child.is_dir():
            print(f"{prefix}{connector}{child.name}/")
            extension = "    " if idx == total - 1 else "│   "
            print_tree(child, prefix + extension)
        else:
            print(f"{prefix}{connector}{child.name}")

if __name__ == "__main__":
    root = Path.cwd()
    print(f"{root.name}/")
    print_tree(root)
