from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional: if the openai package is not installed, instruct users to install via README
try:
    from openai import OpenAI
except Exception:  # pragma: no cover - fallback when dependency is missing
    OpenAI = None  # type: ignore


# ----------------------------
# Data structures (aligned with regex agent to reuse markdown/comparison logic)
# ----------------------------


@dataclass
class Evidence:
    text: str
    start: int = -1
    end: int = -1


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
# OpenAI client and call helpers
# ----------------------------


def create_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Please `pip install openai>=1.0.0` first."
        )

    # 默认配置，可以通过环境变量覆盖
    default_api_key = "sk-Dppz8ZLK62lWyZc7G3w3LtkJx7sNgmjdF65kzR7hKtQlnQDL"
    default_base_url = "https://yunwu.zeabur.app/v1"

    api_key = os.getenv("OPENAI_API_KEY", default_api_key)
    if not api_key or api_key == "your_openai_api_key_here":
        raise EnvironmentError(
            "Please set OPENAI_API_KEY environment variable or modify the default_api_key in the code"
        )

    base_url = os.getenv("OPENAI_BASE_URL", default_base_url)
    # 使用配置的base_url
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    else:
        return OpenAI(api_key=api_key)


LLM_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


def chunk_text(text: str, max_len: int = 6000) -> List[str]:
    if len(text) <= max_len:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_len)
        chunks.append(text[start:end])
        start = end
    return chunks


SYSTEM_PROMPT = (
    "You are a senior expert in building regulations/GFA compliance. Strictly extract rules from the provided regulation text related to Gross Floor Area (GFA) classification into full area, half area, or excluded. Cover: "
    "1) storey height rules; 2) covering/enclosure components (balcony, canopy, roof, arcade/corridor, lift/elevator shaft, stairwell, bay window, atrium/light well); "
    "3) use/special spaces (car park, basement, mezzanine, attic, plant room, refuse room). "
    "Return English only, JSON only, follow the schema exactly. If a field is missing, return empty arrays or null. Do not add explanations."
)

USER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "height_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["full", "half", "excluded", "unknown"],
                    },
                    "thresholds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "comparator": {
                                    "type": ["string", "null"],
                                    "enum": [">=", ">", "<=", "<", "between", None],
                                },
                                "min_m": {"type": ["number", "null"]},
                                "max_m": {"type": ["number", "null"]},
                            },
                            "required": ["comparator", "min_m", "max_m"],
                        },
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "thresholds", "evidence"],
            },
        },
        "cover_enclosure_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["full", "half", "excluded", "conditional", "unknown"],
                    },
                    "notes": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["feature", "label", "evidence"],
            },
        },
        "special_use_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "label": {
                        "type": "string",
                        "enum": ["full", "half", "excluded", "conditional", "unknown"],
                    },
                    "notes": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["feature", "label", "evidence"],
            },
        },
    },
    "required": ["height_rules", "cover_enclosure_rules", "special_use_rules"],
}


def call_openai_json(client: OpenAI, model: str, text: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Extract strictly from the regulation text below and return JSON only.\n\n"
                + json.dumps({"schema": USER_JSON_SCHEMA}, ensure_ascii=False)
                + "\n\nRegulation text:\n"
                + text
            ),
        },
    ]

    content = "{}"
    
    # Prefer JSON mode; if not supported, fall back to plain text and attempt to parse
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        if resp and resp.choices and len(resp.choices) > 0:
            content = resp.choices[0].message.content or "{}"
    except Exception as e:
        print(f"JSON mode failed: {e}. Trying plain text mode...")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            if resp and resp.choices and len(resp.choices) > 0:
                content = resp.choices[0].message.content or "{}"
        except Exception as e2:
            print(f"Plain text mode also failed: {e2}. Using default empty response.")
            content = "{}"

    # Try parsing JSON
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{[\s\S]*\}\s$|\{[\s\S]*\}$", content)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {
            "height_rules": [],
            "cover_enclosure_rules": [],
            "special_use_rules": [],
        }


def parse_llm_region_output(
    region: str,
    source_name: Optional[str],
    source_path: Optional[str],
    original_text: str,
    data: Dict[str, Any],
) -> RegionRegulation:
    def to_float(x: Any) -> Optional[float]:
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    height_rules: List[HeightRule] = []
    for item in data.get("height_rules", []) or []:
        label = item.get("label") or "unknown"
        thresholds = []
        for th in item.get("thresholds", []) or []:
            comparator = th.get("comparator")
            min_m = to_float(th.get("min_m"))
            max_m = to_float(th.get("max_m"))
            thresholds.append((comparator, min_m, max_m))
        evs = [Evidence(text=e) for e in (item.get("evidence") or [])]

        if thresholds:
            for comparator, min_m, max_m in thresholds:
                height_rules.append(
                    HeightRule(
                        label=label,
                        comparator=comparator,
                        threshold_min_m=min_m,
                        threshold_max_m=max_m,
                        evidence=evs,
                    )
                )
        else:
            height_rules.append(
                HeightRule(
                    label=label,
                    comparator=None,
                    threshold_min_m=None,
                    threshold_max_m=None,
                    evidence=evs,
                )
            )

    def parse_features(arr: List[Dict[str, Any]]) -> List[FeatureRule]:
        out: List[FeatureRule] = []
        for item in arr or []:
            feature = item.get("feature") or "unknown"
            label = item.get("label") or "unknown"
            notes = item.get("notes") or ""
            evs = [Evidence(text=e) for e in (item.get("evidence") or [])]
            out.append(
                FeatureRule(feature_key=feature, label=label, notes=notes, evidence=evs)
            )
        return out

    cover = parse_features(data.get("cover_enclosure_rules", []) or [])
    special = parse_features(data.get("special_use_rules", []) or [])

    return RegionRegulation(
        region=region,
        source_name=source_name,
        source_path=source_path,
        text_length=len(original_text),
        height_rules=height_rules,
        cover_enclosure_rules=cover,
        special_use_rules=special,
    )


# ----------------------------
# Aggregation/comparison and Markdown
# ----------------------------


def compare_regions(per_region: Dict[str, RegionRegulation]) -> Dict[str, Any]:
    comparison: Dict[str, Any] = {
        "height_rules": {},
        "cover_enclosure": {},
        "special_use": {},
    }

    height_summary: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for region, rr in per_region.items():
        height_summary[region] = {}
        for r in rr.height_rules:
            height_summary[region].setdefault(r.label, []).append(
                {
                    "comparator": r.comparator,
                    "min_m": r.threshold_min_m,
                    "max_m": r.threshold_max_m,
                }
            )
    comparison["height_rules"] = height_summary

    def map_feature(feature_rules: List[FeatureRule]) -> Dict[str, str]:
        fmap: Dict[str, str] = {}
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


def to_markdown(
    per_region: Dict[str, RegionRegulation], comparison: Dict[str, Any]
) -> str:
    lines: List[str] = []

    lines.append("## 1) Storey height-related GFA calculation rules")
    for region, rr in per_region.items():
        lines.append(f"### {region}")
        if not rr.height_rules:
            lines.append("- No obvious extraction")
            continue
        for r in rr.height_rules:
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
    header = "| Feature | " + " | ".join(per_region.keys()) + " |"
    sep = "|" + "---|" * (len(per_region) + 1)
    lines.append(header)
    lines.append(sep)
    for fk in feature_keys:
        row = [fk]
        for region, rr in per_region.items():
            label = next(
                (fr.label for fr in rr.cover_enclosure_rules if fr.feature_key == fk),
                "-",
            )
            row.append(label)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("\n## 3) Use/special spaces (parking/basement etc.) comparison")
    feature_keys2 = ["parking", "basement", "mezzanine", "attic", "equipment", "refuse"]
    header = "| Use/Space | " + " | ".join(per_region.keys()) + " |"
    sep = "|" + "---|" * (len(per_region) + 1)
    lines.append(header)
    lines.append(sep)
    for fk in feature_keys2:
        row = [fk]
        for region, rr in per_region.items():
            label = next(
                (fr.label for fr in rr.special_use_rules if fr.feature_key == fk), "-"
            )
            row.append(label)
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def fmt_m(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.2f} m" if v < 10 and abs(v - round(v)) > 1e-6 else f"{int(round(v))} m"


# ----------------------------
# Main
# ----------------------------


def analyze_with_llm(
    inputs: List[Dict[str, Any]], model: str = LLM_DEFAULT_MODEL
) -> AnalysisResult:
    client = create_openai_client()

    per_region: Dict[str, RegionRegulation] = {}

    for item in inputs:
        region = item.get("region")
        if not region:
            raise ValueError("Each input item must include a region field")
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

        parts = chunk_text(text)
        merged: Dict[str, Any] = {
            "height_rules": [],
            "cover_enclosure_rules": [],
            "special_use_rules": [],
        }
        for idx, part in enumerate(parts):
            data = call_openai_json(client, model=model, text=part)
            for k in merged.keys():
                merged[k].extend(data.get(k, []) or [])

        rr = parse_llm_region_output(region, source_name, src_path, text, merged)
        per_region[region] = rr

    comparison = compare_regions(per_region)
    return AnalysisResult(per_region=per_region, comparison=comparison)


# ----------------------------
# CLI
# ----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM Regulation Analysis Agent: Cross-region GFA rule comparison"
    )
    p.add_argument("--input", required=True, help="Path to input JSON file")
    p.add_argument("--out-json", help="Path to write structured comparison JSON")
    p.add_argument("--out-md", help="Path to write Markdown summary")
    p.add_argument(
        "--model",
        default=LLM_DEFAULT_MODEL,
        help="LLM model name; defaults to $OPENAI_MODEL or gpt-4o-mini",
    )
    return p


def main_cli():
    args = build_arg_parser().parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list")

    result = analyze_with_llm(payload, model=args.model)

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

    md = to_markdown(result.per_region, result.comparison)
    if args.out_md:
        Path(args.out_md).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main_cli()
