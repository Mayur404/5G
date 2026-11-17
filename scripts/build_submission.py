"""Sync the paper with the generated results and package the course submission.

    python scripts/build_submission.py sync      # copy figures/tables/macros into paper/
    python scripts/build_submission.py pdf       # sync, then compile paper/main.pdf
    python scripts/build_submission.py package   # sync, then write dist/*.zip
    python scripts/build_submission.py           # sync + package (no compile)

The paper never reads results/ directly, so that paper/ can be uploaded to
Overleaf on its own. ``sync`` is the only way results reach the manuscript, and
tests/test_paper.py fails if paper/ and results/ ever disagree.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
DIST = ROOT / "dist"

# (source, destination inside paper/)
PAPER_ASSETS = [
    *(
        (f"results/care/figures/{name}.pdf", f"figures/{name}.pdf")
        for name in ("calibration", "tradeoff", "frontier", "training", "policy", "sweeps")
    ),
    ("results/demo/handover_rescue.jpg", "figures/handover_rescue.jpg"),
    ("results/care/paper_macros.tex", "generated/paper_macros.tex"),
    ("results/care/main_table.tex", "generated/main_table.tex"),
    ("results/care/frontier_table.tex", "generated/frontier_table.tex"),
]
OVERLEAF_FILES = ["main.tex", "references.bib", "figures", "generated"]

# Everything a grader needs to install, run and regenerate the study.
CODE_ITEMS = [
    "README.md",
    "REPRODUCE.md",
    "requirements.txt",
    "requirements-web.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "bootstrap.ps1",
    "Dockerfile",
    ".dockerignore",
    "src",
    "web",
    "tests",
    "scripts",
    "results",
    "docs/problem_statement.txt",
    "docs/project_brief.txt",
    "revision_reports",
]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".aux", ".log", ".out", ".blg", ".synctex.gz"}


def sync() -> None:
    for source, dest in PAPER_ASSETS:
        target = PAPER / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, target)
    print(f"synced {len(PAPER_ASSETS)} result files into paper/")


def compile_pdf() -> None:
    """pdflatex + bibtex, as Overleaf does. Falls back to Tectonic if present."""
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        latex = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        steps = [latex, [shutil.which("bibtex") or "bibtex", "main"], latex, latex]
        try:
            for cmd in steps:
                subprocess.run(cmd, cwd=PAPER, check=True, capture_output=True)
            return
        except subprocess.CalledProcessError:
            print("pdflatex failed (missing packages?); trying Tectonic", file=sys.stderr)
    tectonic = shutil.which("tectonic") or ROOT / ".tools" / "tectonic" / "tectonic.exe"
    if not Path(tectonic).exists():
        raise SystemExit("No working LaTeX found. Install TeX Live/MiKTeX, or use Overleaf.")
    subprocess.run([str(tectonic), "main.tex"], cwd=PAPER, check=True, capture_output=True)
    print("compiled paper/main.pdf with Tectonic")


def _files(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and not SKIP_PARTS & set(item.parts):
            if not any(item.name.endswith(s) for s in SKIP_SUFFIXES):
                yield item


def package() -> None:
    DIST.mkdir(exist_ok=True)
    overleaf = DIST / "Overleaf_Source.zip"
    with zipfile.ZipFile(overleaf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in OVERLEAF_FILES:
            for file in _files(PAPER / name):
                zf.write(file, file.relative_to(PAPER).as_posix())
    print(f"wrote {overleaf.relative_to(ROOT)} ({overleaf.stat().st_size / 1e6:.1f} MB)")

    bundle = DIST / "CARE_Final_Submission.zip"
    prefix = "CARE_Final_Submission"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(overleaf, f"{prefix}/Overleaf_Source.zip")
        pdf = PAPER / "main.pdf"
        if pdf.exists():
            zf.write(pdf, f"{prefix}/CARE_paper.pdf")
        else:
            print("warning: paper/main.pdf not found; run `pdf` first", file=sys.stderr)
        for name in OVERLEAF_FILES:
            for file in _files(PAPER / name):
                zf.write(file, f"{prefix}/paper/{file.relative_to(PAPER).as_posix()}")
        for name in CODE_ITEMS:
            path = ROOT / name
            if not path.exists():
                print(f"warning: {name} missing, skipped", file=sys.stderr)
                continue
            for file in _files(path):
                zf.write(file, f"{prefix}/{file.relative_to(ROOT).as_posix()}")
    size = bundle.stat().st_size / 1e6
    print(f"wrote {bundle.relative_to(ROOT)} ({size:.1f} MB)")
    if size > 100:
        print("warning: bundle exceeds the 100 MB upload limit", file=sys.stderr)


def main(argv: list[str]) -> None:
    steps = argv or ["sync", "package"]
    for step in steps:
        if step == "sync":
            sync()
        elif step == "pdf":
            sync()
            compile_pdf()
        elif step == "package":
            sync()
            package()
        else:
            raise SystemExit(f"unknown step {step!r}; use sync, pdf or package")


if __name__ == "__main__":
    main(sys.argv[1:])
