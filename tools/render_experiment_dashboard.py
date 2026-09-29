from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from html import escape
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.json"
OUTPUT = ROOT / "experiments" / "dashboard.html"

LANES = {
    "Data & evaluation": {0, 1, 8, 11, 15, 17, 38, 40, 57, 58, 62, 63, 64, 67},
    "Candidate ranking": {2, 4, 5, 6, 12, 13, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 42, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 61},
    "Temporal evidence": {3, 7, 10, 14, 16, 18, 19, 20, 43, 44, 59, 65, 68},
    "Runtime & release": {9, 21, 22, 24, 25, 27, 36, 60, 66},
    "Transfer": {23, 26},
}

DECISION_LABELS = {
    "accept": "accepted",
    "reject": "rejected",
    "reject_as_submission_candidate": "rejected",
    "inconclusive": "inconclusive",
}


def experiment_number(item: dict) -> int:
    return int(item["id"].split("-")[1])


def metric_text(metrics: dict) -> str:
    parts = []
    for name, value in metrics.items():
        label = name.replace("_", " ")
        if isinstance(value, float):
            rendered = f"{value:.4g}"
        else:
            rendered = str(value)
        parts.append(f"<dt>{escape(label)}</dt><dd>{escape(rendered)}</dd>")
    return "".join(parts)


def card(item: dict) -> str:
    decision = DECISION_LABELS[item["decision"]]
    sources = ", ".join(item["data"]["sources"])
    return f"""
      <article class="experiment {decision}" data-decision="{decision}" id="{item['id']}">
        <header><span>{item['id']}</span><strong>{escape(item['title'])}</strong></header>
        <p>{escape(item['conclusion'])}</p>
        <details>
          <summary>Protocol and metrics</summary>
          <p><b>Sources:</b> {escape(sources)}</p>
          <p><b>Split:</b> {escape(item['data']['split_unit'])}</p>
          <p><b>Leakage check:</b> {escape(item['data']['leakage_check'])}</p>
          <dl>{metric_text(item['metrics'])}</dl>
          <code>{escape(item['command'])}</code>
        </details>
      </article>"""


def summary_table(runs: list[dict]) -> str:
    by_id = {item["id"]: item for item in runs}
    g4 = by_id["EXP-005"]["metrics"]
    world = by_id["EXP-014"]["metrics"]
    runtime = by_id["EXP-021"]["metrics"]
    smoke = by_id["EXP-022"]["metrics"]
    transfer = by_id["EXP-023"]["metrics"]
    ros_short = by_id["EXP-025"]["metrics"]
    ros_long = by_id["EXP-027"]["metrics"]
    mlp = by_id["EXP-029"]["metrics"]
    spatial = by_id["EXP-031"]["metrics"]
    spatial_mlp = by_id["EXP-032"]["metrics"]
    gated = by_id["EXP-033"]["metrics"]
    abstention = by_id["EXP-034"]["metrics"]
    stronger_cv = by_id["EXP-035"]["metrics"]
    shared_context = by_id["EXP-036"]["metrics"]
    randomized_cv = by_id["EXP-037"]["metrics"]
    transplant = by_id["EXP-041"]["metrics"]
    extra_trees = by_id["EXP-046"]["metrics"]
    compact_hybrid = by_id["EXP-056"]["metrics"]
    shape_audit = by_id["EXP-057"]["metrics"]
    portable = by_id["EXP-060"]["metrics"]
    final = by_id["EXP-067"]["metrics"]
    portable_temporal = by_id["EXP-068"]["metrics"]
    world_reduction = 1 - (
        world["world_tracker_real_alarm_frames"]
        / world["range_tracker_real_alarm_frames"]
    )
    rows = [
        ("Strict scenario recall", f'{g4["scenario_recall_strict"]:.1%}', "measured", "EXP-005", "warn"),
        ("Strict negative specificity", f'{g4["scenario_negative_specificity_strict"]:.1%}', ">= 95%", "EXP-005", "pass"),
        ("Weak-normal alarm rate", f'{g4["assumed_normal_alarm_rate_strict"]:.3%}', "<= 1%", "EXP-005", "pass"),
        ("World-tracker alarm reduction", f'{world_reduction:.0%} (20 -> 15)', "> 0%", "EXP-014", "pass"),
        ("External geometry recall", f'{transfer["geometric_clearance_target_frames"]}/{transfer["evaluable_clearance_targets"]} ({transfer["geometric_zero_shot_recall"]:.1%})', "audit", "EXP-023", "pass"),
        ("External G4 recall", f'{transfer["ranked_clearance_target_frames"]}/{transfer["evaluable_clearance_targets"]} ({transfer["ranked_zero_shot_recall"]:.1%})', "audit", "EXP-023", "warn"),
        ("Offline full-mode p95", f'{runtime["end_to_end_worst_repeat_p95_ms"]:.2f} ms', "< 100 ms", "EXP-021", "pass"),
        ("ROS 10-frame worst p95", f'{max(ros_short["run_1_p95_ms"], ros_short["run_2_p95_ms"]):.2f} ms', "< 100 ms", "EXP-025", "pass"),
        ("ROS 100-frame reproducibility", "100/100 exact stable payloads", "exact", "EXP-027", "pass"),
        ("ROS 100-frame worst p95", f'{max(ros_long["run_1_p95_ms"], ros_long["run_2_p95_ms"]):.2f} ms', "< 100 ms", "EXP-027", "fail"),
        ("Linear vs tiny MLP", f'{mlp["linear_mean_positive_recall"]:.1%} vs {mlp["mean_positive_recall"]:.1%} recall; {mlp["linear_mean_real_frame_alarm_rate"]:.2%} vs {mlp["mean_real_frame_alarm_rate"]:.2%} alarms', "MLP dominates", "EXP-029", "fail"),
        ("Component + spatial CV recall", f'{spatial["component_mean_recall"]:.1%} -> {spatial["combined_mean_recall"]:.1%}', "improves", "EXP-031", "fail"),
        ("Spatial MLP recall / alarms", f'{spatial_mlp["mean_positive_recall"]:.1%} / {spatial_mlp["mean_real_frame_alarm_rate"]:.2%}', "beats component baseline", "EXP-032", "fail"),
        ("Domain-gated CV recall", f'{gated["geometry_mean_recall"]:.1%} -> {gated["gated_25_mean_recall"]:.1%}', "improves", "EXP-033", "fail"),
        ("OOD sparse-audit recall / alarms", f'{abstention["guarded_mean_recall"]:.1%} / {abstention["guarded_mean_real_alarm_rate"]:.2%}', "candidate-cache audit", "EXP-034", "pass"),
        ("Random-conv CV recall", f'{stronger_cv["original_spatial_mean_recall"]:.1%} -> {stronger_cv["random_conv_mean_recall"]:.1%}', "beats geometry", "EXP-035", "fail"),
        ("OOD alarm-frame p95", f'{shared_context["guarded_p95_ms"]:.2f} ms', "< 100 ms", "EXP-036", "pass"),
        ("Randomized CV recall", f'{randomized_cv["base_random_conv_recall"]:.1%} -> {randomized_cv["strong_randomization_recall"]:.1%}', "beats geometry", "EXP-037", "fail"),
        ("Real-transplant CV recall", f'{transplant["base_cv_recall"]:.1%} -> {transplant["transplant_recall"]:.1%}', "improves with zero alarms", "EXP-041", "pass"),
        ("Extra Trees recall / alarms", f'{extra_trees["mean_positive_recall"]:.1%} / {extra_trees["mean_real_frame_alarm_rate"]:.2%}', ">57.1% / <=0.71%", "EXP-046", "pass"),
        ("100-tree hybrid recall / alarms", f'{compact_hybrid["mean_positive_recall"]:.1%} / {compact_hybrid["mean_real_frame_alarm_rate"]:.2%}', ">57.1% / <=0.71%", "EXP-056", "pass"),
        ("Unseen-shape hybrid recall", f'{shape_audit["hybrid_ranked_recall"]:.1%}', ">57.1%", "EXP-057", "fail"),
        ("Portable hybrid p95 / size", f'{portable["hybrid_p95_ms"]:.1f} ms / {portable["portable_model_bytes"] / 1000:.1f} KB', "<100 ms", "EXP-060", "pass"),
        ("Regression suite", f'{portable["unit_tests"]}/44', "44/44", "EXP-060", "pass"),
        ("Selected full-bag alarm rate", f'{final["selected_normal_alarm_rate"]:.2%}', "lowest tested standalone", "EXP-067", "warn"),
        ("Selected exact-box specificity", f'{final["selected_exact_box_negative_specificity"]:.1%}', "100%", "EXP-067", "pass"),
        ("Selected runtime p95", f'{final["selected_latency_p95_ms"]:.2f} ms', "< 100 ms", "EXP-067", "pass"),
        ("Portable temporal episodes", f'{portable_temporal["per_frame_alarm_episodes"]} -> {portable_temporal["world_3of5_alarm_episodes"]}', "no event loss", "EXP-068", "fail"),
    ]
    body = "".join(
        f'<tr><td>{escape(metric)}</td><td>{escape(value)}</td>'
        f'<td>{escape(gate)}</td><td><a href="#{evidence}">{evidence}</a></td>'
        f'<td><span class="metric-status {status}">{status}</span></td></tr>'
        for metric, value, gate, evidence, status in rows
    )
    return f"""<section class="metric-summary">
    <h2>Decision metrics</h2>
    <p>Frozen headline metrics only. Audit rows are transfer evidence, not organizer ground truth.</p>
    <div class="table-scroll"><table>
      <thead><tr><th>Metric</th><th>Result</th><th>Gate</th><th>Evidence</th><th>Status</th></tr></thead>
      <tbody>{body}</tbody>
    </table></div>
  </section>"""


def regime_comparison_table(runs: list[dict]) -> str:
    metrics = {item["id"]: item["metrics"] for item in runs}["EXP-059"]
    return f"""<section class="metric-summary">
    <h2>Universal detector vs mapped-route expert</h2>
    <p>Recall and negative specificity below use the same 376 obstacle pairs. The mapped expert additionally receives a clean same-location reference, so it is a separate operating regime.</p>
    <div class="table-scroll"><table>
      <thead><tr><th>Mode</th><th>Extra input</th><th>Recall</th><th>Negative specificity</th><th>Evidence</th></tr></thead>
      <tbody>
        <tr><td>Universal linear</td><td>None</td><td>{metrics['linear_recall']:.1%}</td><td>78.1%</td><td><a href="#EXP-059">EXP-059</a></td></tr>
        <tr><td>Universal 100-tree hybrid</td><td>None</td><td>{metrics['hybrid_recall']:.1%}</td><td>{metrics['hybrid_negative_specificity']:.1%}</td><td><a href="#EXP-059">EXP-059</a></td></tr>
        <tr><td>Aligned route memory</td><td>Clean same-location reference</td><td>{metrics['route_memory_recall']:.1%}</td><td>{metrics['route_memory_negative_specificity']:.1%}</td><td><a href="#EXP-059">EXP-059</a></td></tr>
      </tbody>
    </table></div>
  </section>"""


def render(runs: list[dict], registry_hash: str) -> str:
    counts = Counter(DECISION_LABELS[item["decision"]] for item in runs)
    by_number = {experiment_number(item): item for item in runs}
    lanes = []
    for name, indexes in LANES.items():
        cells = []
        for index in sorted(indexes):
            item = by_number[index]
            decision = DECISION_LABELS[item["decision"]]
            cells.append(
                f'<a class="cell {decision}" href="#{item["id"]}" '
                f'title="{escape(item["title"])}">E{index:02d}</a>'
            )
        lanes.append(
            f'<div class="lane"><span>{escape(name)}</span><div>{"".join(cells)}</div></div>'
        )
    cards = "".join(card(item) for item in runs)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MetroGuard experiment map</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1017; --panel:#121a24; --text:#edf3fa;
      --muted:#9eb0c3; --line:#293747; --green:#40c98a; --red:#ff6b72;
      --yellow:#e8bd4c; --blue:#68a8ff; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,sans-serif; }}
    main {{ max-width:1120px; margin:auto; padding:28px 20px 60px; }}
    h1 {{ margin:0; font-size:28px; font-weight:650; }}
    .subtitle,.hash {{ color:var(--muted); }}
    .stats {{ display:flex; flex-wrap:wrap; gap:18px; margin:20px 0; }}
    .stats b {{ font-size:22px; display:block; }}
    .metric-summary {{ margin:20px 0; }}
    .metric-summary h2 {{ margin:0 0 4px; font-size:19px; }}
    .metric-summary p {{ margin:0 0 10px; color:var(--muted); }}
    .table-scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
    table {{ width:100%; border-collapse:collapse; min-width:720px; background:var(--panel); }}
    th,td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
    tbody tr:last-child td {{ border-bottom:0; }}
    td:nth-child(2),td:nth-child(3) {{ font-variant-numeric:tabular-nums; }}
    td a {{ color:var(--blue); text-decoration:none; }}
    .metric-status {{ display:inline-block; min-width:48px; padding:2px 7px; border-radius:999px;
      text-align:center; font-size:12px; text-transform:uppercase; border:1px solid var(--line); }}
    .metric-status.pass {{ color:var(--green); }}
    .metric-status.warn {{ color:var(--yellow); }}
    .metric-status.fail {{ color:var(--red); }}
    .map {{ border-block:1px solid var(--line); padding:14px 0; }}
    .lane {{ display:grid; grid-template-columns:155px 1fr; gap:12px; margin:8px 0; align-items:center; }}
    .lane>span {{ color:var(--muted); }}
    .lane>div {{ display:flex; flex-wrap:wrap; gap:6px; }}
    .cell {{ color:var(--text); text-decoration:none; border:1px solid var(--line);
      border-left-width:5px; padding:7px 10px; border-radius:5px; }}
    .accepted {{ border-left-color:var(--green)!important; }}
    .rejected {{ border-left-color:var(--red)!important; }}
    .inconclusive {{ border-left-color:var(--yellow)!important; }}
    nav {{ display:flex; gap:8px; flex-wrap:wrap; margin:20px 0 12px; }}
    button {{ color:var(--text); background:transparent; border:1px solid var(--line);
      padding:7px 11px; border-radius:5px; cursor:pointer; }}
    button[aria-pressed="true"] {{ border-color:var(--blue); }}
    .experiments {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }}
    .experiment {{ background:var(--panel); border:1px solid var(--line); border-left-width:5px;
      border-radius:6px; padding:13px; scroll-margin-top:12px; }}
    .experiment header {{ display:flex; gap:10px; align-items:baseline; }}
    .experiment header span {{ color:var(--blue); font-variant-numeric:tabular-nums; }}
    .experiment header strong {{ font-weight:600; }}
    .experiment p {{ margin:8px 0; }}
    details {{ color:var(--muted); }}
    summary {{ cursor:pointer; color:var(--text); }}
    dl {{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:3px 12px; }}
    dt,dd {{ margin:0; }}
    dd {{ color:var(--text); }}
    code {{ display:block; white-space:pre-wrap; overflow-wrap:anywhere; color:var(--text); }}
    .hidden {{ display:none; }}
    @media(max-width:720px) {{ .experiments {{ grid-template-columns:1fr; }}
      .lane {{ grid-template-columns:1fr; gap:4px; }} }}
  </style>
</head>
<body>
<main>
  <h1>MetroGuard experiment map</h1>
  <div class="subtitle">AI-native development · safety-native runtime</div>
  <div class="stats">
    <span><b>{len(runs)}</b>experiments</span>
    <span><b>{counts['accepted']}</b>accepted</span>
    <span><b>{counts['rejected']}</b>rejected</span>
    <span><b>{counts['inconclusive']}</b>inconclusive</span>
  </div>
  {summary_table(runs)}
  {regime_comparison_table(runs)}
  <section class="map">{"".join(lanes)}</section>
  <nav aria-label="Filter by decision">
    <button type="button" data-filter="all" aria-pressed="true">All</button>
    <button type="button" data-filter="accepted" aria-pressed="false">Accepted</button>
    <button type="button" data-filter="rejected" aria-pressed="false">Rejected</button>
    <button type="button" data-filter="inconclusive" aria-pressed="false">Inconclusive</button>
  </nav>
  <section class="experiments">{cards}</section>
  <p class="hash">registry sha256: {registry_hash}</p>
</main>
<script>
  const buttons = [...document.querySelectorAll('button[data-filter]')];
  const cards = [...document.querySelectorAll('.experiment')];
  buttons.forEach(button => button.addEventListener('click', () => {{
    const selected = button.dataset.filter;
    buttons.forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    cards.forEach(card => card.classList.toggle(
      'hidden', selected !== 'all' && card.dataset.decision !== selected
    ));
  }}));
</script>
</body>
</html>
"""


def expected() -> str:
    raw = REGISTRY.read_bytes()
    runs = json.loads(raw)
    known = set().union(*LANES.values())
    actual = {experiment_number(item) for item in runs}
    if known != actual:
        raise SystemExit(f"dashboard lanes mismatch: missing={actual-known}, stale={known-actual}")
    return render(runs, sha256(raw).hexdigest())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = expected()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            raise SystemExit("experiment dashboard is stale")
        print(f"OK: {OUTPUT.relative_to(ROOT)}")
        return
    OUTPUT.write_text(content, encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
