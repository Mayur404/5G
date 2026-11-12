import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_care_paper_is_complete_and_traceable_to_results():
    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    assert "%ABSTRACT%" not in paper and "%BODY%" not in paper
    care = ROOT / "results" / "care"
    if not (care / "paper_macros.tex").exists():
        pytest.skip("CARE report has not been generated")
    macros = (care / "paper_macros.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", macros))
    body = paper.split(r"\begin{document}", 1)[1]
    # Generated result macros are CamelCase; standard LaTeX commands are lowercase.
    used = set(re.findall(r"\\([A-Z][A-Za-z]*)", body))
    standard = {"Big", "Delta", "Pr", "IEEEkeywords"}
    assert used - standard <= defined, sorted(used - standard - defined)
    assert len(used & defined) > 100  # Results come from macros, not typed numbers.
    for table in re.findall(r"\\input\{([^}]+)\}", body):
        assert (ROOT / table).exists(), table
    for stem in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", body):
        assert (care / "figures" / stem).exists(), stem
    citations = {
        key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", body) for key in group.split(",")
    }
    entries = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "references.bib").read_text("utf-8")))
    assert citations and citations <= entries
