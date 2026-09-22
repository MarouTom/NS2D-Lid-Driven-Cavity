#!/usr/bin/env python3
"""Build report/report.pdf using Tectonic or XeLaTeX/LuaLaTeX.

Times New Roman must be installed for the report's fontspec configuration.
"""
import argparse
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--engine", help="Path to a Tectonic, latexmk, xelatex or lualatex executable"
    )
    a = ap.parse_args()
    engine = (
        a.engine
        or shutil.which("tectonic")
        or shutil.which("latexmk")
        or shutil.which("xelatex")
        or shutil.which("lualatex")
    )
    if not engine:
        raise SystemExit(
            "Install Tectonic or TeX Live with XeLaTeX and Times New Roman, then rerun. "
            "The built report/report.pdf is already readable."
        )
    engine = str(Path(engine).resolve())
    cwd = ROOT / "report"
    if "tectonic" in Path(engine).name:
        commands = [[engine, "--keep-logs", "report.tex"]]
    elif "latexmk" in Path(engine).name:
        commands = [
            [engine, "-xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
        ]
    else:
        commands = [
            [engine, "-interaction=nonstopmode", "-halt-on-error", "report.tex"]
        ] * 2
    for command in commands:
        subprocess.run(command, cwd=cwd, check=True)
    print(cwd / "report.pdf")


if __name__ == "__main__":
    main()
