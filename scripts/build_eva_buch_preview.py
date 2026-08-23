#!/usr/bin/env python3
"""Rebuild eva_buch_preview.json from eva_buch's real per-concept UMAP exports.

Reads bertopic/src/data/<Concept>.json from a sibling eva_buch checkout
(openevo-ccs-lab/eva_buch), for the three concepts shown on the CV site:
Mensch (Human), Evolution, Verhalten (Behavior). Outliers are dropped
(is_outlier: true) - same rule the original single-concept preview used.
Mensch's corpus is far larger than the other two (~3785 vs ~850 non-outlier
excerpts), so it's stratified-sampled down to roughly match, keeping its
real per-subject proportions. Subject-to-color index is assigned from one
global, alphabetically-independent list so the same subject gets the same
color across all three panels.

Run from anywhere; paths are relative to this script's location, assuming
the usual sibling-repo layout (D:\\dev\\eirdosh_cv and
D:\\dev\\openevo-ccs-lab\\eva_buch side by side).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EVA_BUCH_DATA = REPO_ROOT.parent / "openevo-ccs-lab" / "eva_buch" / "bertopic" / "src" / "data"
OUT_PATH = REPO_ROOT / "eva_buch_preview.json"

# Global subject order/index, shared across all three concept panels so a
# subject's color is consistent everywhere it appears.
GLOBAL_SUBJECTS = ["Bio", "Ethik, Philo", "Geschichte", "Psychologie", "Sozialkunde"]

MENSCH_SAMPLE_TARGET = 850
RANDOM_SEED = 42

CONCEPTS = [
    ("Mensch", "Human"),
    ("Evolution", "Evolution"),
    ("Verhalten", "Behavior"),
]


def load_concept(name: str) -> list[dict]:
    path = EVA_BUCH_DATA / f"{name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [p for p in data if not p.get("is_outlier")]


def stratified_sample(points: list[dict], target: int, seed: int) -> list[dict]:
    if len(points) <= target:
        return points
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        by_subject[p["subject"]].append(p)
    rng = random.Random(seed)
    total = len(points)
    sampled = []
    for subject, group in by_subject.items():
        share = round(target * len(group) / total)
        sampled.extend(rng.sample(group, min(share, len(group))))
    return sampled


def build_concept_entry(name: str, name_en: str) -> dict:
    points = load_concept(name)
    total_before = len(points)

    subject_counts = defaultdict(int)
    for p in points:
        subject_counts[p["subject"]] += 1
    distribution = {
        s: round(100 * n / total_before, 1) for s, n in subject_counts.items()
    }

    if name == "Mensch":
        points = stratified_sample(points, MENSCH_SAMPLE_TARGET, RANDOM_SEED)

    out_points = []
    for p in points:
        subj = p["subject"]
        if subj not in GLOBAL_SUBJECTS:
            raise ValueError(f"Unexpected subject '{subj}' in {name}.json - add it to GLOBAL_SUBJECTS")
        x, y, z = p["umap_3d"]
        out_points.append({
            "p": [round(x, 3), round(y, 3), round(z, 3)],
            "s": GLOBAL_SUBJECTS.index(subj),
            "e": p["excerpt"],
        })

    return {
        "topic": name,
        "topic_en": name_en,
        "sampled_from": total_before,
        "distribution_pct": distribution,
        "points": out_points,
    }


def main():
    concepts = [build_concept_entry(name, name_en) for name, name_en in CONCEPTS]

    out = {
        "source_repo": "openevo-ccs/eva_buch",
        "note": (
            "German K-12 curriculum-standard excerpts, grouped by BERTopic "
            "(a language model) into topics and projected from sentence-"
            "embedding space into 3D with UMAP, so excerpts with related "
            "meaning sit near each other."
        ),
        "subjects": GLOBAL_SUBJECTS,
        "concepts": concepts,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    for c in concepts:
        print(f"{c['topic']} ({c['topic_en']}): {len(c['points'])} points "
              f"(sampled from {c['sampled_from']}), distribution={c['distribution_pct']}")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
