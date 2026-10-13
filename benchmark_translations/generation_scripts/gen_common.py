"""Shared helpers for the scaling-variant generators (generate_scaling_*.py).

Each generator reads the sweep data from scaling_data.json and writes one
program per (family, size) into a translations directory. Generated programs
are written without comments, like the rest of benchmark_translations/, and an
existing file is never overwritten unless --overwrite is given.
"""

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSLATIONS = os.path.dirname(HERE)

with open(os.path.join(HERE, "scaling_data.json")) as f:
    SD = json.load(f)
DATA = SD["data"]
VALUES = SD["values"]
FILESPEC = SD["filespec"]
CONST = SD["constants"]


def fname(family, v, ext):
    prefix, pad = FILESPEC[family]
    return f"{family}_{prefix}{str(v).zfill(pad)}.{ext}"


def strip_comments(text, marker):
    """Drop `marker` comments (outside string literals), trailing whitespace and
    repeated blank lines. Line-preserving: code lines are never joined."""
    out = []
    for line in text.splitlines():
        in_str = False
        cut = len(line)
        for i, ch in enumerate(line):
            if ch == '"' and (i == 0 or line[i - 1] != "\\"):
                in_str = not in_str
            elif not in_str and line.startswith(marker, i):
                cut = i
                break
        code = line[:cut].rstrip()
        if not code and line.strip() and cut < len(line):
            continue                      # a comment-only line
        if not code and (not out or not out[-1]):
            continue                      # leading or repeated blank line
        out.append(code)
    while out and not out[-1]:
        out.pop()
    return "\n".join(out) + "\n"


class Writer:
    """Writes generated programs into `out_dir`, skipping existing files unless
    `overwrite`; records what was written and skipped."""

    def __init__(self, out_dir, overwrite, marker):
        self.out_dir, self.overwrite, self.marker = out_dir, overwrite, marker
        self.written, self.skipped = [], []
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, name, text):
        path = os.path.join(self.out_dir, name)
        if os.path.exists(path) and not self.overwrite:
            self.skipped.append(name)
        else:
            with open(path, "w") as fh:
                fh.write(strip_comments(text, self.marker))
            self.written.append(name)
        return path

    def report(self):
        print(f"wrote {len(self.written)} file(s) to {self.out_dir}"
              + (f"; skipped {len(self.skipped)} existing (use --overwrite)" if self.skipped else ""))
        for name in self.written:
            print(f"  + {name}")


def parse_args(description, default_out):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=default_out,
                    help=f"output directory (default: {os.path.relpath(default_out)})")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite programs that already exist")
    return ap.parse_args()
