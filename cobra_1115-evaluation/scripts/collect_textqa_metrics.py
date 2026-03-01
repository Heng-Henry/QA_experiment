#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple


TASK_MAIN_METRIC = {
    "longbench_hotpotqa": "hotpot_qa_f1",
    "longbench_2wikimqa": "hotpot_qa_f1",
    "longbench_narrativeqa": "hotpot_qa_f1",
    "longbench_qmsum": "rouge_l",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect text-QA sweep metrics from Quamba/MambaQuant outputs."
    )
    p.add_argument("--map", required=True, help="Path to textqa_sweep_*.tsv mapping file.")
    p.add_argument(
        "--project-root",
        default="/work/u9907741/project_0209",
        help="Project root path.",
    )
    p.add_argument("--out", required=True, help="Output TSV path.")
    return p.parse_args()


def _safe_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def parse_result_json(path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data and isinstance(data["results"], dict):
        results = data["results"]
    elif "lm_eval" in data and isinstance(data["lm_eval"], dict):
        # Quamba main.py output format.
        results = data["lm_eval"]
    else:
        results = {}
    flat = {}
    for task, task_metrics in results.items():
        flat[task] = {}
        for k, v in task_metrics.items():
            name = k.split(",")[0]
            flat[task][name] = v
    args = data.get("args", {})
    meta = {
        "model_source": str(args.get("model", "")),
        "result_file": str(path),
    }
    return flat, meta


def parse_lmeval_table_from_slurm_out(path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    metrics: Dict[str, Dict[str, float]] = {}
    cur_task = ""
    model_source = ""

    model_args_re = re.compile(r"^model_args:\s*(.+)$")
    for ln in lines:
        m = model_args_re.match(ln.strip())
        if m:
            model_source = m.group(1).strip()
            break

    for ln in lines:
        if not ln.startswith("|"):
            continue
        parts = [p.strip() for p in ln.split("|")]
        if len(parts) < 10:
            continue
        task = parts[1] or cur_task
        metric = parts[5]
        value = _safe_float(parts[6])
        stderr = _safe_float(parts[8])
        if not task or not metric or value is None:
            continue
        cur_task = task
        metrics.setdefault(task, {})
        metrics[task][metric] = value
        if stderr is not None:
            metrics[task][f"{metric}_stderr"] = stderr

    return metrics, {"model_source": model_source, "result_file": str(path)}


def find_quamba_result(project_root: Path, run_tag: str) -> Optional[Path]:
    log_dir = project_root / "runs" / "quamba_eval" / run_tag / "logs"
    if not log_dir.is_dir():
        return None
    files = sorted(log_dir.glob("*.json"))
    return files[0] if files else None


def find_mambaquant_result(project_root: Path, run_tag: str, jobid: str) -> Optional[Path]:
    run_json = project_root / "runs" / "mambaquant_eval" / run_tag / "results.json"
    if run_json.is_file():
        return run_json
    slurm_out = project_root / "runs" / "slurm" / f"mambaq_lb_eval.{jobid}.out"
    return slurm_out if slurm_out.is_file() else None


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    map_path = Path(args.map)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            framework = row["framework"].strip()
            mode = row["mode"].strip()
            task = row["task"].strip()
            jobid = row["jobid"].strip()
            run_tag = row["run_tag"].strip()
            metric_name = TASK_MAIN_METRIC.get(task, "")

            source: Optional[Path] = None
            parsed: Dict[str, Dict[str, float]] = {}
            meta: Dict[str, str] = {"model_source": "", "result_file": ""}
            status = "ok"
            note = ""

            if framework == "quamba":
                source = find_quamba_result(project_root, run_tag)
                if source and source.suffix == ".json":
                    parsed, meta = parse_result_json(source)
                else:
                    status = "missing_result_file"
            elif framework == "mambaquant":
                source = find_mambaquant_result(project_root, run_tag, jobid)
                if source and source.name == "results.json":
                    parsed, meta = parse_result_json(source)
                elif source and source.suffix == ".out":
                    parsed, meta = parse_lmeval_table_from_slurm_out(source)
                    note = "parsed_from_slurm_out"
                else:
                    status = "missing_result_file"
            else:
                status = "unknown_framework"

            score = None
            stderr = None
            if status == "ok":
                tmetrics = parsed.get(task, {})
                score = tmetrics.get(metric_name)
                stderr = tmetrics.get(f"{metric_name}_stderr")
                if score is None:
                    status = "missing_metric"

            rows_out.append(
                {
                    "framework": framework,
                    "mode": mode,
                    "task": task,
                    "jobid": jobid,
                    "run_tag": run_tag,
                    "status": status,
                    "main_metric": metric_name,
                    "score": "" if score is None else f"{score:.6f}",
                    "stderr": "" if stderr is None else f"{stderr:.6f}",
                    "model_source": meta.get("model_source", ""),
                    "result_file": meta.get("result_file", ""),
                    "note": note,
                }
            )

    fieldnames = [
        "framework",
        "mode",
        "task",
        "jobid",
        "run_tag",
        "status",
        "main_metric",
        "score",
        "stderr",
        "model_source",
        "result_file",
        "note",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows_out)

    print(f"[DONE] wrote {out_path}")


if __name__ == "__main__":
    main()
