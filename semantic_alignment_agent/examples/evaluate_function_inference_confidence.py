import os
import re
import json
from typing import Dict, Any, Tuple, List

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

try:
    from semantic_alignment_agent.core.function_inference import FunctionInferenceEngine
    from semantic_alignment_agent.utils import (
        GeometricFeatures,
        BoundingBox,
        Point3D,
        IfcElementInfo,
        FunctionType,
    )
except ModuleNotFoundError:
    import sys as _sys
    import os as _os
    # Add project root (parent of the package directory) to sys.path
    _PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    _sys.path.insert(0, _PROJECT_ROOT)
    from semantic_alignment_agent.core.function_inference import FunctionInferenceEngine
    from semantic_alignment_agent.utils import (
        GeometricFeatures,
        BoundingBox,
        Point3D,
        IfcElementInfo,
        FunctionType,
    )


DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "dataset"
)


def _bbox(w: float, d: float, h: float) -> BoundingBox:
    return BoundingBox(
        min_point=Point3D(0, 0, 0), max_point=Point3D(w, d, h)
    )


def _make_features(
    thickness: float,
    w: float,
    d: float,
    floors_spanned: int = 1,
    support_function: str = "unknown",
    location_indicator: str = "unknown",
) -> GeometricFeatures:
    gf = GeometricFeatures(
        bounding_box=_bbox(w, d, thickness),
        area=w * d,
        volume=w * d * thickness,
        thickness=thickness,
    )
    # attach extra attributes expected by inference engine
    gf.aspect_ratio = (max(w, d) / max(1e-6, min(w, d))) if min(w, d) > 0 else 1.0
    gf.vertical_span = {"floors_spanned": floors_spanned}
    gf.support_function = support_function
    gf.location_indicator = location_indicator
    gf.thickness_indicator = (
        "thin" if thickness is not None and thickness < 0.1 else "thick"
    )
    gf.position = {
        "floor_level": 0,
        "hint": "synthetic",
    }
    return gf


def _label_to_function_type(label: str) -> FunctionType:
    s = label.strip().lower()
    if any(k in s for k in ["atrium", "lobby", "central void", "double-height"]):
        return FunctionType.ATRIUM
    if any(k in s for k in ["shaft", "elevator", "hvac", "electrical"]):
        return FunctionType.SHAFT
    if "stairwell" in s or "stair" in s:
        return FunctionType.STAIRWELL
    if "mechanical" in s:
        return FunctionType.MECHANICAL_ROOM
    if any(k in s for k in ["covered parking", "parking", "semi-outdoor", "sunken plaza", "bridge"]):
        return FunctionType.AUXILIARY_CONSTRUCTION
    if any(k in s for k in ["interior floor slab", "stair landing", "foundation base slab", "interior floor"]):
        return FunctionType.STRUCTURAL_FLOOR
    if any(k in s for k in ["roof slab", "exterior floor", "exterior landing", "balcony", "roof"]):
        return FunctionType.DECORATION_PLATFORM
    if any(k in s for k in ["interior space", "gross floor area"]):
        return FunctionType.GENERAL_USE_SPACE
    if "exterior space" in s or "light well" in s:
        return FunctionType.AUXILIARY_CONSTRUCTION
    return FunctionType.UNKNOWN


def _infer(engine: FunctionInferenceEngine, info: IfcElementInfo, gf: GeometricFeatures):
    info.geometric_features = gf
    return engine.infer_function(info, gf, spatial_context=None)


def eval_group_a(engine: FunctionInferenceEngine) -> Tuple[int, int, List[Tuple[float, int]], List[Dict[str, Any]]]:
    path = os.path.join(DATASET_DIR, "tableConvert.com_A.xlsx")
    df = pd.read_excel(path)
    results = []
    details: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    for _, row in df.iterrows():
        direct = str(row.get("Direct Mapping", "")).strip()
        ifc_type = str(row.get("IFC Type", "")).strip() or "IfcSlab"
        predefined = str(row.get("PredefinedType", "")).strip()
        is_external = str(row.get("IsExternal", "")).strip().upper() == "TRUE"

        expected = _label_to_function_type(direct)

        # synthetic geometry based on type
        if ifc_type == "IfcSpace":
            gf = _make_features(
                thickness=3.0,
                w=6.0,
                d=5.0,
                floors_spanned=1,
                support_function="supports_occupancy",
                location_indicator="habitable_space" if not is_external else "outdoor_construction",
            )
        else:
            # slab
            thickness = 0.2 if not is_external else 0.08
            gf = _make_features(
                thickness=thickness,
                w=6.0,
                d=5.0,
                floors_spanned=1,
                support_function="supports_occupancy" if thickness >= 0.15 else "unknown",
                location_indicator="habitable_space" if not is_external else "auxiliary_construction",
            )

        info = IfcElementInfo(
            guid=f"A_{row.get('No.', 'N')}",
            ifc_type=ifc_type,
            properties={"PredefinedType": predefined, "IsExternal": is_external},
        )

        inf = _infer(engine, info, gf)
        is_evaluable = expected != FunctionType.UNKNOWN
        if is_evaluable:
            is_correct = int(inf.primary_function == expected)
            total += 1
            correct += is_correct
            results.append((inf.confidence, is_correct))
        else:
            is_correct = None
        details.append({
            "guid": info.guid,
            "ifc_type": ifc_type,
            "expected": expected.value,
            "predicted": inf.primary_function.value,
            "confidence": float(inf.confidence),
            "correct": is_correct,
            "evaluable": is_evaluable,
            "expected_raw": direct
        })

    return correct, total, results, details


def eval_group_b(engine: FunctionInferenceEngine) -> Tuple[int, int, List[Tuple[float, int]], List[Dict[str, Any]]]:
    path = os.path.join(DATASET_DIR, "tableConvert.com_B.xlsx")
    df = pd.read_excel(path)
    results = []
    details: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    for _, row in df.iterrows():
        ifc = str(row.get("IFC", "IfcSpace")).strip()
        predefined = str(row.get("PredefinedType", "")).strip()
        feats = str(row.get("Key Geometry Features", "")).lower()
        label = str(row.get("Actual Function", "")).strip()
        expected = _label_to_function_type(label)

        # parse geometry hints
        h = 10.0
        w = 3.0
        d = 2.0
        floors = 1
        if m := re.search(r"h>\s*(\d+)m", feats):
            h = float(m.group(1))
        if m := re.search(r"w<\s*(\d+)m", feats):
            w = float(m.group(1)) * 0.9
        if m := re.search(r"w>\s*(\d+)m", feats):
            w = float(m.group(1)) * 1.1
        if "floor" in feats or "multi-floor" in feats or "floor_count" in feats:
            floors = 3

        # for atrium set wide, for shafts set very slender
        if expected == FunctionType.ATRIUM:
            w, d = max(w, 8.0), max(d, 8.0)
        elif expected == FunctionType.SHAFT or expected == FunctionType.STAIRWELL:
            w, d = min(w, 3.0), min(d, 2.0)

        gf = _make_features(
            thickness=h,
            w=w,
            d=d,
            floors_spanned=floors,
            support_function="supports_occupancy",
            location_indicator="habitable_space",
        )

        info = IfcElementInfo(
            guid=f"B_{row.get('No.', 'N')}",
            ifc_type=ifc,
            properties={"PredefinedType": predefined},
        )

        inf = _infer(engine, info, gf)
        is_evaluable = expected != FunctionType.UNKNOWN
        if is_evaluable:
            is_correct = int(inf.primary_function == expected)
            total += 1
            correct += is_correct
            results.append((inf.confidence, is_correct))
        else:
            is_correct = None
        details.append({
            "guid": info.guid,
            "ifc_type": ifc,
            "expected": expected.value,
            "predicted": inf.primary_function.value,
            "confidence": float(inf.confidence),
            "correct": is_correct,
            "evaluable": is_evaluable,
            "expected_raw": label
        })

    return correct, total, results, details


def eval_group_c(engine: FunctionInferenceEngine) -> Tuple[int, int, List[Tuple[float, int]], List[Dict[str, Any]]]:
    path = os.path.join(DATASET_DIR, "tableConvert.com_C.xlsx")
    df = pd.read_excel(path)
    results = []
    details: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    for _, row in df.iterrows():
        ifc = str(row.get("IFC", "IfcOpeningElement")).strip()
        f1 = str(row.get("Geometry Feature 1", "")).lower()
        f2 = str(row.get("Geometry Feature 2", "")).lower()
        f3 = str(row.get("Geometry Feature 3", "")).lower()
        label = str(row.get("Actual Function", "")).strip()
        expected = _label_to_function_type(label)

        h = 12.0
        w = 3.0
        d = 2.0
        floors = 2
        for f in (f1, f2, f3):
            if m := re.search(r"h>\s*(\d+)m", f):
                h = float(m.group(1))
            if m := re.search(r"w>\s*(\d+)m", f):
                w = float(m.group(1))
            if m := re.search(r"w<\s*(\d+)m", f):
                w = float(m.group(1)) * 0.9
            if "multi" in f or "multiple floors" in f:
                floors = 3

        if expected == FunctionType.ATRIUM:
            w, d = max(w, 8.0), max(d, 8.0)
        elif expected in (FunctionType.SHAFT, FunctionType.STAIRWELL):
            w, d = min(w, 3.0), min(d, 2.0)

        gf = _make_features(
            thickness=h,
            w=w,
            d=d,
            floors_spanned=floors,
            support_function="supports_occupancy",
            location_indicator="habitable_space",
        )
        info = IfcElementInfo(
            guid=f"C_{row.get('No.', 'N')}",
            ifc_type=ifc,
            properties={},
        )
        inf = _infer(engine, info, gf)
        is_evaluable = expected != FunctionType.UNKNOWN
        if is_evaluable:
            is_correct = int(inf.primary_function == expected)
            total += 1
            correct += is_correct
            results.append((inf.confidence, is_correct))
        else:
            is_correct = None
        details.append({
            "guid": info.guid,
            "ifc_type": ifc,
            "expected": expected.value,
            "predicted": inf.primary_function.value,
            "confidence": float(inf.confidence),
            "correct": is_correct,
            "evaluable": is_evaluable,
            "expected_raw": label
        })

    return correct, total, results, details


def eval_group_d(engine: FunctionInferenceEngine) -> Tuple[int, int, List[Tuple[float, int]], List[Dict[str, Any]]]:
    path = os.path.join(DATASET_DIR, "tableConvert.com_D.xlsx")
    df = pd.read_excel(path)
    results = []
    details: List[Dict[str, Any]] = []
    total = 0
    correct = 0

    for _, row in df.iterrows():
        label = str(row.get("Function", "")).strip()
        discr = str(row.get("Geometry Discriminator", "")).lower()
        expected = _label_to_function_type(label)

        # choose IFC type by label
        if "opening" in discr or "well" in label:
            ifc = "IfcOpeningElement"
        elif "space" in label or "lobby" in label or "parking" in label or "bridge" in label:
            ifc = "IfcSpace"
        else:
            ifc = "IfcSlab"

        # synthetic geometry
        if expected == FunctionType.ATRIUM:
            gf = _make_features(thickness=6.0, w=12.0, d=10.0, floors_spanned=2, support_function="supports_occupancy", location_indicator="habitable_space")
        elif expected in (FunctionType.SHAFT, FunctionType.STAIRWELL):
            gf = _make_features(thickness=15.0, w=2.5, d=2.0, floors_spanned=3, support_function="supports_occupancy", location_indicator="habitable_space")
        elif expected == FunctionType.MECHANICAL_ROOM:
            gf = _make_features(thickness=3.0, w=6.0, d=5.0, floors_spanned=1, support_function="supports_equipment", location_indicator="auxiliary_construction")
        elif expected == FunctionType.AUXILIARY_CONSTRUCTION:
            gf = _make_features(thickness=3.0, w=8.0, d=6.0, floors_spanned=1, support_function="unknown", location_indicator="outdoor_construction")
        else:
            gf = _make_features(thickness=3.0, w=6.0, d=5.0, floors_spanned=1)

        info = IfcElementInfo(guid=f"D_{row.get('No.', 'N')}", ifc_type=ifc, properties={})
        inf = _infer(engine, info, gf)
        is_evaluable = expected != FunctionType.UNKNOWN
        if is_evaluable:
            is_correct = int(inf.primary_function == expected)
            total += 1
            correct += is_correct
            results.append((inf.confidence, is_correct))
        else:
            is_correct = None
        details.append({
            "guid": info.guid,
            "ifc_type": ifc,
            "expected": expected.value,
            "predicted": inf.primary_function.value,
            "confidence": float(inf.confidence),
            "correct": is_correct,
            "evaluable": is_evaluable,
            "expected_raw": label
        })

    return correct, total, results, details


def summarize(results: List[Tuple[int, int, List[Tuple[float, int]], List[Dict[str, Any]]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    groups = ["A", "B", "C", "D"]
    total_correct = 0
    total_count = 0
    confidences: List[float] = []
    correctness: List[int] = []
    for i, (c, t, pairs, _details) in enumerate(results):
        acc = (c / t * 100.0) if t > 0 else 0.0
        summary[f"Group {groups[i]} Accuracy"] = acc
        total_correct += c
        total_count += t
        for conf, ok in pairs:
            confidences.append(conf)
            correctness.append(ok)
    summary["Overall Accuracy"] = (total_correct / total_count * 100.0) if total_count > 0 else 0.0
    # confidence bins
    bins = {"low": [], "medium": [], "high": []}
    for conf, ok in zip(confidences, correctness):
        if conf < 0.5:
            bins["low"].append(ok)
        elif conf < 0.8:
            bins["medium"].append(ok)
        else:
            bins["high"].append(ok)
    summary["Confidence-Accuracy by Bin"] = {
        k: (np.mean(v) * 100.0 if len(v) > 0 else None) for k, v in bins.items()
    }
    # Spearman correlation
    if len(confidences) > 2:
        corr, p = spearmanr(confidences, correctness)
        summary["Spearman(confidence, correctness)"] = float(corr)
        summary["Spearman p-value"] = float(p)
    return summary


def main():
    engine = FunctionInferenceEngine(enable_llm=False)
    a = eval_group_a(engine)
    b = eval_group_b(engine)
    c = eval_group_c(engine)
    d = eval_group_d(engine)
    summary = summarize([a, b, c, d])

    print("\n=== Function Inference Confidence Evaluation ===")
    print(f"Group A Accuracy: {summary['Group A Accuracy']:.1f}%")
    print(f"Group B Accuracy: {summary['Group B Accuracy']:.1f}%")
    print(f"Group C Accuracy: {summary['Group C Accuracy']:.1f}%")
    print(f"Group D Accuracy: {summary['Group D Accuracy']:.1f}%")
    print(f"Overall Accuracy: {summary['Overall Accuracy']:.1f}%")
    print("Confidence-Accuracy by Bin:")
    for k, v in summary["Confidence-Accuracy by Bin"].items():
        print(f"  {k}: {v:.1f}%" if v is not None else f"  {k}: N/A")
    if "Spearman(confidence, correctness)" in summary:
        print(
            f"Spearman(conf, corr): {summary['Spearman(confidence, correctness)']:.3f} (p={summary['Spearman p-value']:.3g})"
        )

    # Detailed results per group
    print("\n--- Detailed Results ---")
    for group_name, group_results in zip(["A", "B", "C", "D"], [a, b, c, d]):
        _, _, _, details = group_results
        print(f"Group {group_name} ({len(details)} items):")
        if not details:
            print("  (no items)")
            continue
        for item in details:
            print(
                f"  {item['guid']}: ifc={item['ifc_type']}, expected={item['expected']}, "
                f"pred={item['predicted']}, conf={item['confidence']:.3f}, correct={item['correct']}"
            )

    # Save detailed results and summary
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(out_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for group_name, group_results in zip(["A", "B", "C", "D"], [a, b, c, d]):
        _, _, _, details = group_results
        for item in details:
            row = {
                "group": group_name,
                **item,
            }
            all_rows.append(row)

    # Write CSV
    try:
        import pandas as pd  # ensure available in runtime
        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(out_dir, "function_eval_details.csv")
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Failed to write CSV: {e}")

    # Write JSON
    json_path = os.path.join(out_dir, "function_eval_details.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    # Write summary JSON
    summary_path = os.path.join(out_dir, "function_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()