# Collect render metrics of a benchmark run into per-scene and consolidated results.md.
#
# Usage: python utils/benchmarking/write_results.py runs/<run_id>
#
# Reads, for every scene folder inside the run directory:
#   <scene>/best_runtimeresults_v2.json   (written by render.py)
#   <scene>/best_iteration.txt            (written by train.py, optional)
#   <scene>/train_time                    (written by train.py, optional)
# and writes <scene>/results.md plus a consolidated <run_dir>/consolidated_results.md
# with one row per scene and an average row.

import json
import sys
from pathlib import Path

METRICS = ["PSNR", "SSIM", "MS-SSIM", "LPIPS-Alex", "Masked-PSNR", "Masked-SSIM"]


def read_scene(scene_dir: Path):
    results_file = scene_dir / "best_runtimeresults_v2.json"
    if not results_file.exists():
        return None
    raw = json.load(open(results_file))
    # structure: {model_path: {"best": {metric: value}}}
    metrics = next(iter(next(iter(raw.values())).values()))

    best_iter = None
    best_iter_file = scene_dir / "best_iteration.txt"
    if best_iter_file.exists():
        best_iter = best_iter_file.read_text().strip().split(":")[-1].strip()

    train_time = None
    train_time_file = scene_dir / "train_time"
    if train_time_file.exists():
        for line in train_time_file.read_text().splitlines():
            if "HH:MM:SS" in line:
                train_time = line.split("): ")[-1].strip()

    return {"metrics": metrics, "best_iter": best_iter, "train_time": train_time}


def fmt(v):
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_scene_md(scene_dir: Path, scene: str, data: dict):
    lines = [f"# Results: {scene}", ""]
    if data["best_iter"]:
        lines.append(f"- Best iteration: {data['best_iter']}")
    if data["train_time"]:
        lines.append(f"- Training time: {data['train_time']}")
    lines += ["",
              "| " + " | ".join(["Metric", "Value"]) + " |",
              "|---|---|"]
    for m in METRICS:
        if m in data["metrics"]:
            lines.append(f"| {m} | {fmt(data['metrics'][m])} |")
    lines += ["", "Per-view metrics: `best_runtimeperview_v2.json`", ""]
    (scene_dir / "results.md").write_text("\n".join(lines))


def parse_hms(s):
    """'HH:MM:SS' -> seconds, or None if unparseable."""
    try:
        h, m, sec = (int(x) for x in s.split(":"))
        return h * 3600 + m * 60 + sec
    except (ValueError, AttributeError):
        return None


def fmt_hms(total_seconds):
    """seconds -> 'HH:MM:SS'."""
    total_seconds = int(round(total_seconds))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_consolidated_md(run_dir: Path, scenes: dict):
    header = ["Scene"] + METRICS + ["Train time"]
    lines = [f"# Benchmark results: {run_dir.name}", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for scene, data in scenes.items():
        row = [scene] + [fmt(data["metrics"].get(m, "-")) for m in METRICS]
        row += [data["train_time"] or "-"]
        lines.append("| " + " | ".join(row) + " |")

    if len(scenes) > 1:
        avg = ["**Average**"]
        for m in METRICS:
            vals = [d["metrics"][m] for d in scenes.values() if m in d["metrics"]]
            avg.append(f"**{sum(vals) / len(vals):.4f}**" if vals else "-")
        secs = [t for t in (parse_hms(d["train_time"]) for d in scenes.values()) if t is not None]
        avg.append(f"**{fmt_hms(sum(secs) / len(secs))}**" if secs else "-")
        lines.append("| " + " | ".join(avg) + " |")

    lines.append("")
    (run_dir / "consolidated_results.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run_dir = Path(sys.argv[1])
    scenes = {}
    for scene_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        data = read_scene(scene_dir)
        if data is not None:
            scenes[scene_dir.name] = data
            write_scene_md(scene_dir, scene_dir.name, data)

    if scenes:
        write_consolidated_md(run_dir, scenes)
        print(f"results written for {len(scenes)} scene(s) -> {run_dir / 'consolidated_results.md'}")
    else:
        print(f"No render results found under {run_dir}")
