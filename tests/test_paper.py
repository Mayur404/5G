from pathlib import Path
import csv
import json
import re

from proran.experiment import _write_latex_macros


ROOT = Path(__file__).resolve().parents[1]


def test_paper_macros_match_measured_rows(tmp_path):
    rows = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))["metrics"]
    with (ROOT / "results" / "paired_comparisons.csv").open(encoding="utf-8") as fh:
        paired = [{key: value if key in {"reference", "metric"} else float(value)
                   for key, value in row.items()} for row in csv.DictReader(fh)]
    expected = tmp_path / "results_macros.tex"
    _write_latex_macros(rows, paired, expected)
    actual = (ROOT / "results" / "results_macros.tex").read_text(encoding="utf-8")
    assert actual == expected.read_text(encoding="utf-8")


def test_paper_has_every_used_result_macro_and_figure():
    paper = (ROOT / "paper.tex").read_text(encoding="utf-8")
    macros = "\n".join((ROOT / "results" / name).read_text(encoding="utf-8")
                       for name in ("results_macros.tex", "scenario_macros.tex"))
    defined = set(re.findall(r"\\newcommand\{(\\[A-Za-z]+)\}", macros))
    used = set(re.findall(r"\\(?:NoRes|Reactive|Certainty|Aware|Oracle|PPO|Urban|Rush|Highway|URLLC)[A-Za-z]+", paper))
    assert used <= defined
    for stem in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", paper):
        assert (ROOT / "results" / "figures" / stem).exists()
    citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", paper)
                 for key in group.split(",")}
    bibliography = (ROOT / "references.bib").read_text(encoding="utf-8")
    entries = set(re.findall(r"@\w+\{([^,]+),", bibliography))
    assert citations <= entries
    for doi in ("10.1109/TCCN.2026.3683197", "10.1109/LCOMM.2020.3001227",
                "10.1109/TNSM.2023.3243837", "10.1109/JSAC.2019.2959245"):
        assert doi in bibliography


def test_exploratory_scenario_audit_has_matched_rows():
    with (ROOT / "results" / "scenario_audit.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 32
    scenarios = {row["scenario"] for row in rows}
    assert scenarios == {"urban", "rush", "highway", "urllc"}
    seed_sets = [{int(row["seed"]) for row in rows if row["scenario"] == name}
                 for name in sorted(scenarios)]
    assert all(seed_set == seed_sets[0] for seed_set in seed_sets)
    assert all(int(row["handovers"]) > 0 for row in rows)


def test_latency_samples_include_each_slice_and_method():
    with (ROOT / "results" / "latency_samples.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {int(row["slice_id"]) for row in rows} == {0, 1, 2}
    assert {row["method"] for row in rows} >= {"No reservation", "PPO (proposed)"}
    assert all(float(row["latency_ms"]) >= 0 for row in rows)
