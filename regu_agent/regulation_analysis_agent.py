from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class Evidence:
    text: str
    start: int
    end: int


@dataclass
class HeightRule:
    label: str  # full | half | excluded | unknown
    comparator: Optional[str]  # ">=", ">", "<=", "<", "between", None
    threshold_min_m: Optional[float]
    threshold_max_m: Optional[float]
    unit_normalized: str = "m"
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class FeatureRule:
    feature_key: str
    label: str  # full | half | excluded | conditional | unknown
    notes: str = ""
    evidence: List[Evidence] = field(default_factory=list)


@dataclass
class RegionRegulation:
    region: str
    source_name: Optional[str]
    source_path: Optional[str]
    text_length: int
    height_rules: List[HeightRule] = field(default_factory=list)
    cover_enclosure_rules: List[FeatureRule] = field(default_factory=list)
    special_use_rules: List[FeatureRule] = field(default_factory=list)


@dataclass
class AnalysisResult:
    per_region: Dict[str, RegionRegulation]
    comparison: Dict[str, Any]


# ----------------------------
# Regex utils
# ----------------------------

NON_DIGIT_SPACE = re.compile(r"\s+")
# No language-specific punctuation normalization to keep code English-only

RANGE_PATTERN = re.compile(
    r"(?P<min>\d+(?:\.\d+)?)\s*(?P<unit>m|meter|meters|ft|feet)?\s*(?:to|and|\-|~|—)\s*(?P<max>\d+(?:\.\d+)?)\s*(?P<unit2>m|meter|meters|ft|feet)?",
    re.IGNORECASE,
)

CMP_PATTERN = re.compile(
    r"(?P<cmp>>=|<=|>|<|≥|≤)\s*(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>m|meter|meters|ft|feet)",
    re.IGNORECASE,
)

SIMPLE_NUMBER_PATTERN = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>m|meter|meters|ft|feet)", re.IGNORECASE
)

# Label keywords (English only)
LABEL_FULL_KWS = [
    "included",
    "counted",
    "included in gfa",
    "counts towards gfa",
    "include in gross floor area",
]
LABEL_HALF_KWS = [
    "half",
    "50%",
    "0.5",
    "counted as 50%",
]
LABEL_EXCLUDED_KWS = [
    "excluded",
    "not included",
    "exempt from gfa",
]

HEIGHT_KEYWORDS = [
    "storey height",
    "clear height",
    "ceiling height",
    "floor-to-floor",
    "floor to floor",
    "floor-to-ceiling",
    "floor to ceiling",
]

# Feature synonyms (English only)
FEATURE_SYNONYMS: Dict[str, List[str]] = {
    # Covering / enclosure
    "balcony": ["balcony", "terrace", "veranda"],
    "canopy": ["canopy", "awning", "eaves"],
    "roof": ["roof", "roof terrace", "roof deck"],
    "lift": ["lift", "elevator", "lift shaft", "elevator shaft"],
    "stair": ["stair", "staircase", "stairwell"],
    "bay_window": ["bay window"],
    "corridor": ["corridor", "arcade", "gallery"],
    "atrium_void": ["atrium", "void", "light well"],
    # Use / special
    "parking": ["car park", "parking", "parking garage"],
    "basement": ["basement", "cellar"],
    "mezzanine": ["mezzanine"],
    "attic": ["attic", "loft"],
    "equipment": ["plant room", "equipment room", "mechanical room", "m&e"],
    "refuse": ["refuse room", "garbage room", "trash room"],
}

# Label detection


def detect_label(window_text: str) -> str:
    text = window_text.lower()
    for kw in LABEL_EXCLUDED_KWS:
        if kw in text:
            return "excluded"
    for kw in LABEL_HALF_KWS:
        if kw in text:
            return "half"
    for kw in LABEL_FULL_KWS:
        if kw in text:
            return "full"
    return "unknown"


def to_meters(value: float, unit: Optional[str]) -> float:
    if not unit:
        return value
    u = unit.lower()
    if u in {"m", "meter", "meters"}:
        return value
    if u in {"ft", "feet"}:
        return value * 0.3048
    return value


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # No special normalization required
    return text


def sentence_split(text: str) -> List[Tuple[int, int, str]]:
    """Rudimentary sentence splitter. Returns (start, end, sentence)."""
    spans: List[Tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[\.!?;\n]", text):
        end = m.end()
        seg = text[start:end].strip()
        if seg:
            spans.append((start, end, seg))
        start = end
    if start < len(text):
        seg = text[start:].strip()
        if seg:
            spans.append((start, len(text), seg))
    return spans


# ----------------------------
# Rule extraction
# ----------------------------


def extract_height_rules(text: str) -> List[HeightRule]:
    text_norm = normalize_text(text)
    sentences = sentence_split(text_norm)
    results: List[HeightRule] = []

    def contains_height_kw(s: str) -> bool:
        s_lower = s.lower()
        return any(kw in s_lower for kw in [k.lower() for k in HEIGHT_KEYWORDS])

    for start, end, sent in sentences:
        if not contains_height_kw(sent):
            continue
        window = sent
        label = detect_label(window)

        # Range thresholds
        for m in RANGE_PATTERN.finditer(window):
            vmin = to_meters(float(m.group("min")), m.group("unit"))
            vmax = to_meters(float(m.group("max")), m.group("unit2") or m.group("unit"))
            results.append(
                HeightRule(
                    label=label,
                    comparator="between",
                    threshold_min_m=min(vmin, vmax),
                    threshold_max_m=max(vmin, vmax),
                    evidence=[Evidence(text=window, start=start, end=end)],
                )
            )

        # Comparator thresholds
        for m in CMP_PATTERN.finditer(window):
            val = to_meters(float(m.group("val")), m.group("unit"))
            cmp_symbol = m.group("cmp")
            cmp_norm = {"≥": ">=", "≤": "<="}.get(cmp_symbol, cmp_symbol)
            results.append(
                HeightRule(
                    label=label,
                    comparator=cmp_norm,
                    threshold_min_m=val if cmp_norm in {">", ">="} else None,
                    threshold_max_m=val if cmp_norm in {"<", "<="} else None,
                    evidence=[Evidence(text=window, start=start, end=end)],
                )
            )

        # If no numeric threshold captured, still keep an entry for traceability
        if not any(
            e.start == start and e.end == end for r in results for e in r.evidence
        ):
            results.append(
                HeightRule(
                    label=label,
                    comparator=None,
                    threshold_min_m=None,
                    threshold_max_m=None,
                    evidence=[Evidence(text=window, start=start, end=end)],
                )
            )

    return merge_similar_height_rules(results)


def merge_similar_height_rules(rules: List[HeightRule]) -> List[HeightRule]:
    """Merge duplicated/very similar rules to reduce noise."""
    merged: List[HeightRule] = []
    for rule in rules:
        found = False
        for m in merged:
            if (
                m.label == rule.label
                and m.comparator == rule.comparator
                and (
                    m.threshold_min_m == rule.threshold_min_m
                    or (
                        m.threshold_min_m
                        and rule.threshold_min_m
                        and abs(m.threshold_min_m - rule.threshold_min_m) < 1e-6
                    )
                )
                and (
                    m.threshold_max_m == rule.threshold_max_m
                    or (
                        m.threshold_max_m
                        and rule.threshold_max_m
                        and abs(m.threshold_max_m - rule.threshold_max_m) < 1e-6
                    )
                )
            ):
                m.evidence.extend(rule.evidence)
                found = True
                break
        if not found:
            merged.append(rule)
    return merged


def extract_feature_rules(text: str, feature_keys: List[str]) -> List[FeatureRule]:
    text_norm = normalize_text(text)
    sentences = sentence_split(text_norm)
    lower_sentences = [(s.lower(), (start, end, s)) for start, end, s in sentences]

    results: List[FeatureRule] = []

    for feature in feature_keys:
        synonyms = FEATURE_SYNONYMS.get(feature, [feature])
        synonyms_lower = [s.lower() for s in synonyms]

        for s_lower, (start, end, s_orig) in lower_sentences:
            if not any(kw in s_lower for kw in synonyms_lower):
                continue
            label = detect_label(s_orig)
            notes = ""

            # Capture percentage-style mentions (e.g., 50%)
            perc = re.search(r"(\d+\.?\d*)\s*%", s_orig)
            if perc and label == "unknown":
                notes = f"ratio: {perc.group(1)}%"

            results.append(
                FeatureRule(
                    feature_key=feature,
                    label=(
                        label
                        if label != "unknown"
                        else ("conditional" if perc else "unknown")
                    ),
                    notes=notes,
                    evidence=[Evidence(text=s_orig, start=start, end=end)],
                )
            )

    return merge_feature_rules(results)


def merge_feature_rules(rules: List[FeatureRule]) -> List[FeatureRule]:
    merged: Dict[Tuple[str, str, str], FeatureRule] = {}
    for r in rules:
        key = (r.feature_key, r.label, r.notes)
        if key not in merged:
            merged[key] = r
        else:
            merged[key].evidence.extend(r.evidence)
    return list(merged.values())


# ----------------------------
# Region parsing and comparison
# ----------------------------


def parse_region_text(
    region: str,
    text: str,
    source_name: Optional[str] = None,
    source_path: Optional[str] = None,
) -> RegionRegulation:
    text = text or ""
    height_rules = extract_height_rules(text)
    cover_enclosure = extract_feature_rules(
        text,
        [
            "balcony",
            "canopy",
            "roof",
            "lift",
            "stair",
            "bay_window",
            "corridor",
            "atrium_void",
        ],
    )
    special = extract_feature_rules(
        text,
        [
            "parking",
            "basement",
            "mezzanine",
            "attic",
            "equipment",
            "refuse",
        ],
    )

    return RegionRegulation(
        region=region,
        source_name=source_name,
        source_path=source_path,
        text_length=len(text),
        height_rules=height_rules,
        cover_enclosure_rules=cover_enclosure,
        special_use_rules=special,
    )


def compare_regions(per_region: Dict[str, RegionRegulation]) -> Dict[str, Any]:
    """Produce cross-region differences for the three aspects."""
    comparison: Dict[str, Any] = {
        "height_rules": {},
        "cover_enclosure": {},
        "special_use": {},
    }

    # 1) Height rules summary by label
    height_summary: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for region, rr in per_region.items():
        height_summary[region] = {}
        for r in rr.height_rules:
            label = r.label
            if label not in height_summary[region]:
                height_summary[region][label] = []
            height_summary[region][label].append(
                {
                    "comparator": r.comparator,
                    "min_m": r.threshold_min_m,
                    "max_m": r.threshold_max_m,
                }
            )
    comparison["height_rules"] = height_summary

    # 2) Covering/enclosure feature mapping (pick strongest label)
    def map_feature(feature_rules: List[FeatureRule]) -> Dict[str, str]:
        fmap: Dict[str, str] = {}
        # Priority: excluded > half > full > conditional > unknown
        order = {"excluded": 4, "half": 3, "full": 2, "conditional": 1, "unknown": 0}
        for fr in feature_rules:
            prev = fmap.get(fr.feature_key)
            if prev is None or order.get(fr.label, 0) > order.get(prev, 0):
                fmap[fr.feature_key] = fr.label
        return fmap

    cover_map: Dict[str, Dict[str, str]] = {}
    special_map: Dict[str, Dict[str, str]] = {}

    for region, rr in per_region.items():
        cover_map[region] = map_feature(rr.cover_enclosure_rules)
        special_map[region] = map_feature(rr.special_use_rules)

    comparison["cover_enclosure"] = cover_map
    comparison["special_use"] = special_map

    return comparison


def analyze_regulations(inputs: List[Dict[str, Any]]) -> AnalysisResult:
    """
    inputs: List[{
        "region": "CN/HK/US/EU/...",
        "text": "regulation text (plain)",  # or provide "file"
        "file": "/abs/path/to/file.txt",
        "source_name": "optional name"
    }]
    """
    per_region: Dict[str, RegionRegulation] = {}

    for item in inputs:
        region = item.get("region")
        if not region:
            raise ValueError("Each input item must include region")
        source_name = item.get("source_name")
        src_path = item.get("file")
        text: Optional[str] = item.get("text")

        if text is None and src_path:
            p = Path(src_path)
            if not p.exists():
                raise FileNotFoundError(f"File not found: {src_path}")
            text = p.read_text(encoding="utf-8", errors="ignore")
        if text is None:
            text = ""

        per_region[region] = parse_region_text(
            region=region, text=text, source_name=source_name, source_path=src_path
        )

    comparison = compare_regions(per_region)
    return AnalysisResult(per_region=per_region, comparison=comparison)


# ----------------------------
# Markdown output
# ----------------------------


def to_markdown(result: AnalysisResult) -> str:
    lines: List[str] = []

    lines.append("## 1) Storey height-related GFA calculation rules")
    for region, rr in result.per_region.items():
        lines.append(f"### {region}")
        if not rr.height_rules:
            lines.append("- No obvious extraction")
            continue
        for r in rr.height_rules:
            rng = ""
            if r.comparator == "between":
                rng = f"{fmt_m(r.threshold_min_m)} to {fmt_m(r.threshold_max_m)}"
            elif r.comparator in {">=", ">", "<=", "<"}:
                val = (
                    r.threshold_min_m
                    if r.comparator in {">=", ">"}
                    else r.threshold_max_m
                )
                rng = f"{r.comparator} {fmt_m(val)}"
            else:
                rng = "No threshold extracted"
            lines.append(f"- Label: **{r.label}**; Threshold: {rng}")

    lines.append(
        "\n## 2) Covering/enclosure components (balcony/canopy/roof/lift etc.) comparison"
    )
    feature_keys = [
        "balcony",
        "canopy",
        "roof",
        "lift",
        "stair",
        "bay_window",
        "corridor",
        "atrium_void",
    ]
    header = "| Feature | " + " | ".join(result.per_region.keys()) + " |"
    sep = "|" + "---|" * (len(result.per_region) + 1)
    lines.append(header)
    lines.append(sep)
    for fk in feature_keys:
        row = [fk]
        for region, rr in result.per_region.items():
            label = next(
                (fr.label for fr in rr.cover_enclosure_rules if fr.feature_key == fk),
                "-",
            )
            row.append(label)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 3) Use/special spaces (parking/basement etc.) comparison")
    feature_keys2 = ["parking", "basement", "mezzanine", "attic", "equipment", "refuse"]
    header = "| Use/Space | " + " | ".join(result.per_region.keys()) + " |"
    sep = "|" + "---|" * (len(result.per_region) + 1)
    lines.append(header)
    lines.append(sep)
    for fk in feature_keys2:
        row = [fk]
        for region, rr in result.per_region.items():
            label = next(
                (fr.label for fr in rr.special_use_rules if fr.feature_key == fk), "-"
            )
            row.append(label)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def fmt_m(v: Optional[float]) -> str:
    if v is None:
        return "-"
    if v >= 10 or abs(v - round(v)) < 1e-6:
        return f"{int(round(v))} m"
    return f"{v:.2f} m"


# ----------------------------
# CLI
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regulation Analysis Agent: Cross-region GFA rule comparison (regex/keyword baseline)"
    )
    p.add_argument(
        "--input", required=True, help="Path to input JSON file (list of regions)"
    )
    p.add_argument("--out-json", help="Path to write structured comparison JSON")
    p.add_argument("--out-md", help="Path to write Markdown summary")
    return p


def main_cli():
    parser = build_arg_parser()
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list; each element describes a region")

    result = analyze_regulations(payload)

    if args.out_json:
        Path(args.out_json).write_text(
            json.dumps(
                {
                    "per_region": {k: asdict(v) for k, v in result.per_region.items()},
                    "comparison": result.comparison,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    md = to_markdown(result)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")

    print(md)


if __name__ == "__main__":
    main_cli()
