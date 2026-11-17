import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"

# Uppercase commands that come from LaTeX packages, not from generated results.
STANDARD = {
    "Big",
    "Delta",
    "Pr",
    "IEEEkeywords",
    "IEEEauthorblockN",
    "IEEEauthorblockA",
    "IEEEproof",
    "REQUIRE",
    "ENSURE",
    "STATE",
    "FOR",
    "ENDFOR",
    "IF",
    "ENDIF",
    "RETURN",
    "TO",
}


def test_care_paper_is_complete_and_traceable_to_results():
    paper = (PAPER / "main.tex").read_text(encoding="utf-8")
    macros_path = PAPER / "generated" / "paper_macros.tex"
    if not macros_path.exists():
        pytest.skip("run `python scripts/build_submission.py sync` first")
    macros = macros_path.read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    defined |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", paper))
    body = paper.split(r"\begin{document}", 1)[1]
    # Generated result macros are CamelCase; standard LaTeX commands are lowercase.
    used = set(re.findall(r"(?<!\\)\\([A-Z][A-Za-z]*)", body))
    assert used - STANDARD <= defined, sorted(used - STANDARD - defined)
    assert len(used & defined) > 100  # Results come from macros, not typed numbers.
    for table in re.findall(r"\\input\{([^}]+)\}", body):
        assert (PAPER / table).exists(), table
    for stem in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", body):
        assert (PAPER / "figures" / stem).exists(), stem
    citations = {
        key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for key in group.split(",")
    }
    entries = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "references.bib").read_text("utf-8")))
    assert citations and citations <= entries
    assert "template text" not in body.lower()


def test_paper_copies_match_generated_results():
    """paper/ is uploaded to Overleaf on its own, so it must equal results/ exactly."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_submission", ROOT / "scripts" / "build_submission.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    for source, dest in build.PAPER_ASSETS:
        src, copy = ROOT / source, PAPER / dest
        if not src.exists():
            pytest.skip(f"{source} has not been generated")
        assert copy.exists(), f"{dest} missing; run scripts/build_submission.py sync"
        assert src.read_bytes() == copy.read_bytes(), f"{dest} is stale; rerun sync"
