"""
UndertriAI — Dataset Preparation
Converts indian_bail_judgments.csv -> structured JSONL episode files for 4 curriculum stages.

Usage:
    python prepare_dataset.py --csv /path/to/indian_bail_judgments.csv --output ./episodes
"""

import ast, json, os, argparse, random, re, csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

random.seed(42)


def parse_list_field(val: str) -> List[str]:
    if not val or val.strip() in ("", "[]"):
        return []
    try:
        result = ast.literal_eval(val)
        return [str(x).strip() for x in result if str(x).strip()]
    except Exception:
        return [s.strip() for s in val.strip("[]").split(",") if s.strip()]


def parse_bool(val: str) -> bool:
    return str(val).strip().lower() in ("true", "1", "yes")


def split_arguments(legal_issues: List[str]) -> Tuple[List[str], List[str]]:
    prosecution, defence = [], []
    for i, issue in enumerate(legal_issues):
        low = issue.lower()
        if any(k in low for k in ["cancel","reject","deny","gravity","tamper","abscond","repeat","custodial","investigate"]):
            prosecution.append(issue)
        elif any(k in low for k in ["grant","parity","cooperat","local","surety","eligible","half","undertrial"]):
            defence.append(issue)
        else:
            (prosecution if i % 2 == 0 else defence).append(issue)
    return prosecution, defence


def infer_custody_months(facts: str, bail_type: str) -> float:
    for pat, mult in [(r"(\d+)\s+years?\s+in\s+(?:custody|prison|jail)", 12),
                      (r"(\d+)\s+months?\s+in\s+(?:custody|prison|jail)", 1)]:
        m = re.search(pat, facts, re.IGNORECASE)
        if m:
            return float(int(m.group(1)) * mult)
    return {"Regular": 6.0, "Anticipatory": 0.0, "Interim": 1.0}.get(bail_type, 4.0)


def infer_max_sentence(sections: List[str]) -> float:
    MAP = {"302":99,"103":99,"307":10,"109":10,"376":14,"64":14,"304B":14,"80":14,
           "395":10,"310":10,"392":10,"309":10,"420":7,"318":7,"498A":3,"85":3,
           "406":3,"316":3,"465":2,"336":2,"323":1,"115":1,"354":2,"74":2,
           "120B":3,"61":3,"506":2,"351":2}
    return max((MAP.get(s.strip(), 0) for s in sections), default=5.0) or 5.0


def assign_stage(row: Dict[str, Any]) -> int:
    landmark = parse_bool(row.get("landmark_case","False"))
    cancel   = parse_bool(row.get("bail_cancellation_case","False"))
    region   = row.get("region","")
    if landmark and not cancel: return 1
    if cancel:                  return 3
    if region in {"Assam","Tamil Nadu","Kerala","Punjab","Maharashtra"}: return 4
    return 2


def build_episode(row: Dict[str, Any]) -> Dict[str, Any]:
    sections = parse_list_field(row.get("ipc_sections","[]"))
    issues   = parse_list_field(row.get("legal_issues","[]"))
    pros, def_ = split_arguments(issues)
    facts    = row.get("facts","")
    btype    = row.get("bail_type","Regular")
    reason   = row.get("judgment_reason","").lower()

    gt_outcome = "Bail Granted" if row.get("bail_outcome","").lower()=="granted" else "Bail Denied"
    if any(k in reason for k in ["not a flight","local ties","cooperat","permanent resident"]):
        gt_risk = "Low"
    elif any(k in reason for k in ["abscond","tamper","influential","intimidat","repeat offend","serious"]):
        gt_risk = "High"
    else:
        gt_risk = "Medium"

    docs = ["FIR Copy","Charge Sheet"]
    if "surety"  in facts.lower(): docs.append("Surety Affidavit")
    if "medical" in facts.lower(): docs.append("Medical Report")
    if "prior"   in facts.lower(): docs.append("Criminal History Record")

    return {
        "case_id": row.get("case_id",""), "case_title": row.get("case_title",""),
        "court": row.get("court",""), "date": row.get("date",""),
        "charge_sheet": facts, "ipc_sections": sections,
        "crime_type": row.get("crime_type","Unknown"), "bail_type": btype,
        "prosecution_arguments": pros, "defence_arguments": def_,
        "legal_principles": parse_list_field(row.get("legal_principles_discussed","[]")),
        "documents_available": docs, "summary": row.get("summary",""),
        "accused_profile": {
            "name": row.get("accused_name","Unknown"), "gender": row.get("accused_gender","Unknown"),
            "occupation": None, "region": row.get("region","Unknown"),
            "prior_cases": row.get("prior_cases","Unknown"), "bail_type": btype,
        },
        "custody_months": infer_custody_months(facts, btype),
        "max_sentence_years": infer_max_sentence(sections),
        "ground_truth": {
            "outcome": gt_outcome, "implicit_flight_risk": gt_risk,
            "judgment_reason": row.get("judgment_reason",""),
            "outcome_detail": row.get("bail_outcome_label_detailed",""),
            "bias_flag": parse_bool(row.get("bias_flag","False")),
            "parity_argument_used": parse_bool(row.get("parity_argument_used","False")),
        },
        "curriculum_stage": assign_stage(row),
        "landmark_case": parse_bool(row.get("landmark_case","False")),
        "bail_cancellation_case": parse_bool(row.get("bail_cancellation_case","False")),
        "region": row.get("region","Unknown"),
        "special_laws": row.get("special_laws",""),
        "schema_drift_eligible": row.get("region","") in {"Assam","Tamil Nadu","Kerala","Punjab","Maharashtra"},
    }


def prepare(csv_path: str, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stages: Dict[int, list] = {1:[], 2:[], 3:[], 4:[]}
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            try:
                ep = build_episode(row)
                stages[ep["curriculum_stage"]].append(ep)
            except Exception as e:
                print(f"  [WARN] Skipping {row.get('case_id')}: {e}")

    all_eps = []
    for stage, eps in stages.items():
        random.shuffle(eps)
        out = os.path.join(output_dir, f"episodes_stage_{stage}.jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for ep in eps: f.write(json.dumps(ep, ensure_ascii=False)+"\n")
        print(f"  Stage {stage}: {len(eps):4d} episodes -> {out}")
        all_eps.extend(eps)

    random.shuffle(all_eps)
    with open(os.path.join(output_dir,"episodes_all.jsonl"), "w", encoding="utf-8") as f:
        for ep in all_eps: f.write(json.dumps(ep, ensure_ascii=False)+"\n")
    print(f"\nDone. Total: {len(all_eps)} episodes | stages: { {k:len(v) for k,v in stages.items()} }")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output", default="./episodes")
    args = p.parse_args()
    prepare(args.csv, args.output)
