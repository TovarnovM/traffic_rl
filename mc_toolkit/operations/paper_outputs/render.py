"""Offline atlas from PLAN's normalized points; no scientific reductions.

Public entry points: render_atlas, export_tables, available_families,
render_conceptual, render_recorded_illustration. Every output directory must be
new. No input is modified. Imports of Matplotlib are lazy (Agg, like reporting).

Illustrations: illustrations.json is a list, or {"entries": [...], "deferred":
[...]}. Each entry has identity (cell/cohort/rule/road_length/density/fraction),
mode="final", selection_evidence (status="selected", n=90,
rule="frozen_median_line_then_environment"; DATA owns scientific verification),
and pair={"Graph": {"analysis_path": "...json", "arrays_path": "...npz"},
"H2": {...}}. Paths are relative to data_dir. NPZ uses allow_pickle=False;
local_speed/local_type are native (T,41,4). Optional map_steps is the recorded
step axis (otherwise native 1..T). analysis.pv.trace holds step, speed, lane,
cumulative_distance. Direct callers may instead pass pair entries containing
analysis+arrays, or trace+local_speed+local_type+map_steps. Selection is never
performed here. Snapshots use recorded steps 1/1000/2000, never nearest frames.
"""
from __future__ import annotations

import csv
import errno
import hashlib
import html
import json
import math
import re
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from .style import RC, canonical_method, method_style
except ImportError:
    from style import RC, canonical_method, method_style

POINT_COLUMNS = "family panel cohort rule method training_line model_id road_length density fraction metric window region lane vehicle_group phase x x_label value sd ci_low ci_high interval_kind n n_planned k k_planned support support_kind status units estimand".split()
FAMILIES = {
    "coverage": "Which planned conditions and training lines are available?",
    "mobility": "What is the absolute PV mobility?",
    "background": "What is the global background price?",
    "ratio": "How does paired background R compare with 1 and .95?",
    "contrasts": "What is the supplied direct effect (first minus comparator)?",
    "laps": "What are conditional lap times and their availability?",
    "control": "How often are interventions requested, applied or available?",
    "selection": "Which recorded validation candidates satisfy the screen?",
    "windows": "How does local mobility vary across windows and exposure?",
    "transitions": "Which recorded local transitions occur?",
    "state": "What are the recorded N/S/Z/A time occupancies?",
    "events": "How frequent and censored are stop/block episodes?",
    "mechanism": "What are the native opportunities, gaps and safety gates?",
    "spatial": "What is the connected occupancy covariance by physical lane?",
    "temporal": "What is the supplied normalized temporal covariance?",
    "onset": "What is observed around recorded onsets?",
    "profiles": "What is the accumulated offset exposure (not a movie)?",
    "tradeoff": "How do PV mobility and background/local cost coexist?",
    "illustrations": "What do the DATA-selected paired recorded episodes show?",
    "conceptual": "What is measured, sensed and controlled?",
    "literature": "T4 literature comparison (manual, not fabricated)",
}
NOTES = {
    "ratio": "R is the supplied mean of paired policy/B1 ratios. The .95 line is a descriptive point screen, not a safety guarantee.",
    "laps": "Mean lap time is conditional on completed laps. Availability n/N = episodes with completed lap / observed episodes; coverage n/n_planned is separate. Missing availability is not zero.",
    "windows": "Local exposure describes different vehicles around each PV; it is not an individual causal cost. Empty exposure is NA.",
    "state": "Recorded state frequency is time occupancy; frames are not independent MC replicates.",
    "events": "v=0 and v<=1 remain distinct. Censored terminal spells are not complete durations; see supplied metric/status and support.",
    "mechanism": "Native eligibility uses pre-step geometry; operational blocking can mix recorded phases. No stochastic-gate outcome is inferred.",
    "spatial": "Connected occupancy covariance, not Pearson correlation. Spatial pairs are not independent MC replicates.",
    "temporal": "Supplied normalized covariance; undefined zero-variance cases remain NA.",
    "onset": "Recorded onset alignment; events are not independent MC replicates. Follow-up support is supplied by DATA.",
    "profiles": "Accumulated offset profiles are exposure summaries, not recorded space-time maps.",
    "selection": "Recorded validation candidate curves, not training return. Infeasible selections have no replacement checkpoint.",
}
BASE = ("family", "panel", "cohort", "rule", "road_length", "metric", "units", "estimand", "phase", "x_label")
FACETS = ("fraction", "window", "region", "lane", "vehicle_group")
FRESH_PANELS = frozenset({"main", "E4", "W7", "E5_L1000", "E5_L2000"})


def _reference_views(rows):
    """Borrow unchanged fixed points for fresh comparisons; never reduce data.

    Only rule/cohort/estimand and candidate display facets change. Their source
    values remain explicit extras; blank fixed candidate IDs are references to
    each candidate view, never evidence for a different learned checkpoint.
    """
    keys = tuple(k for k in BASE + FACETS if k not in {"cohort", "rule", "estimand"}) + ("density", "x", "profile")
    candidate_keys = {base: _candidate_keys(group) for base, group in _groups(rows, BASE)}
    display = ("rule", "cohort", "estimand", "candidate_id", "model_id")
    def scope(row):
        cohort = row.get("cohort", "")
        if row.get("panel") == "W7" and _number(cohort) in (.5, 1, 2):
            cohort = "1"
        estimand = row.get("estimand", "")
        # DATA uses this separate label for its supplied global family means.
        if estimand == "equal frozen whole-line means; no training-line pairing":
            estimand = "mean of episode scalars"
        global_scalar = estimand == "mean of episode scalars" and not any(
            row.get(k) for k in ("window", "region", "lane", "vehicle_group", "phase"))
        return tuple("" if (k == "profile" and global_scalar) or
                     (k == "x" and row.get("x_label") not in {"density", "lattice density"})
                     else row.get(k, "") for k in keys) + (cohort, estimand)
    def fixed(row):
        return (row.get("panel") in FRESH_PANELS
                and canonical_method(row.get("method", "")) not in {"Graph", "Node", "Grid"}
                and row.get("training_line") in (None, "", "all"))
    references = defaultdict(list)
    occupied = set()
    for row in rows:
        if fixed(row):
            if row.get("rule") == "fixed":
                if row.get("panel") != "W7" or _number(row.get("cohort")) == 1:
                    references[scope(row)].append(row)
            else:
                occupied.add((scope(row), row.get("x", ""), canonical_method(row["method"]),
                              tuple(row.get(k, "") for k in display)))
    projected, matched, origins = [], set(), {}
    visited = set()
    for target in rows:
        if (target.get("panel") not in FRESH_PANELS or target.get("rule") not in {"S0", "C95"}
                or canonical_method(target.get("method", "")) not in {"Graph", "Node", "Grid"}):
            continue
        key = scope(target)
        # Visit a full curve once per view; retain candidate IDs for compatibility.
        view = (key, tuple(target.get(k, "") for k in display))
        if view in visited:
            continue
        visited.add(view)
        for reference in references.get(key, ()):
            if any(reference.get(k) and reference.get(k) != target.get(k) for k in ("candidate_id", "model_id")):
                continue
            facets = candidate_keys[tuple(target.get(k, "") for k in BASE)]
            destination = tuple(target.get(k, "") if k not in {"candidate_id", "model_id"} or k in facets
                                else reference.get(k, "") for k in display)
            identity = (key, reference.get("x", ""), canonical_method(reference["method"]), destination)
            if identity in occupied:
                if origins.get(identity) == id(reference):
                    matched.add(id(reference))
                continue
            matched.add(id(reference))
            occupied.add(identity)
            origins[identity] = id(reference)
            projected.append(dict(reference, **dict(zip(display, destination)),
                                  **{"source_" + k: reference.get(k, "") for k in display},
                                  reference_projection="fixed control; plot only"))
    return [r for r in rows if id(r) not in matched] + projected


def _candidate_keys(rows):
    # Candidate evolves along checkpoint-step curves; elsewhere it is a facet.
    if all(r.get("x_label") in {"checkpoint environment steps", "checkpoint steps", "env_steps"}
           or (r.get("family") == "selection" and r.get("x_label") in {"steps", "step"}) for r in rows):
        return ()
    scopes = _groups(rows, BASE + FACETS + ("method", "training_line"))
    return tuple(k for k in ("candidate_id", "model_id")
                 if any(len({r.get(k, "") for r in group}) > 1 for _, group in scopes))


def _density_keys(rows):
    return ("density",) if any(r.get("density") and r.get("density") != r.get("x") for r in rows) else ()


def _plt():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    return plt


def _iter_csv(path):
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        yield from csv.DictReader(stream)


WINDOW_OUTCOMES = frozenset({
    "pooled_speed", "pooled_stopped_fraction", "normalized_local_delay",
    "cell_time_density", "cell_time_flow", "time_mean_density", "time_mean_flow",
    "empty_fraction",
})
PV_METRICS = frozenset({"mean_speed_priority", "pv_speed", "pv_mean_speed", "mean_priority_speed"})
COST_METRICS = frozenset({"R", "paired_R", "paired_background_R", "mean_paired_R", "normalized_local_delay",
                          "pooled_delay", "mean_speed_background", "primary_normalized_local_delay",
                          "primary_pooled_speed", "pooled_speed"})
LOCAL_MARGINALS = [
    dict(window="fixed_front_2", region="*", lane="all", vehicle_group="*"),
    dict(window="fixed_front_2", region="full", lane="*", vehicle_group="*"),
    dict(window="*", region="full", lane="all", vehicle_group="background"),
]


def _local_scope(row, all_local_scopes):
    # Unscoped legacy summaries have no Cartesian axes to reduce.
    if all_local_scopes or row.get("family") not in {"windows", "transitions"} or not row.get("region"):
        return True
    return any(all(value == "*" or row.get(key) == value for key, value in scope.items())
               for scope in LOCAL_MARGINALS)


def _categorical_mean(row):
    metric = row.get("metric", "")
    return metric.endswith(("_type/mean", "_reason/mean", "_direction_set/mean"))


def _outcome(row, metrics):
    if _categorical_mean(row):
        return False
    if metrics is not None:
        return True
    if row.get("family") == "mechanism":
        return row.get("metric", "").endswith(("/frequency", "/mean"))
    return row.get("family") != "windows" or row.get("metric") in WINDOW_OUTCOMES


def _tradeoff_row(row, metrics):
    return ((row.get("family") == "mobility" and row.get("metric") in PV_METRICS)
            or ((row.get("family") == "ratio" or row.get("metric") in COST_METRICS)
                and (metrics is None or row.get("metric") in metrics)))


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows, columns=None):
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _unique(rows, name):
    return sorted({str(row.get(name, "")) for row in rows if row.get(name, "") != ""})


def _id(identity):
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=True)
    # Readable scope plus digest of the complete facet/metric identity.
    parts = [identity.get(k, "all") for k in ("family", "panel", "cohort", "rule", "road_length", "metric", "layout", "page")]
    slug = "--".join(re.sub(r"[^a-zA-Z0-9.-]+", "-", str(p or "all"))[:24] for p in parts)
    return slug + "--" + hashlib.sha256(encoded.encode()).hexdigest()[:12]


def _new_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _formats(formats):
    formats = tuple(dict.fromkeys(formats))
    if not formats or set(formats) - {"png", "pdf", "svg"}:
        raise ValueError("formats must contain png, pdf and/or svg")
    return formats


def _support(rows):
    labels = []
    for a, b, label in (("n", "n_planned", "n"), ("k", "k_planned", "k")):
        relevant = rows if a == 'n' else [r for r in rows if r.get(a) not in (None, '') or r.get(b) not in (None, '')]
        if not relevant:
            continue
        counts = sorted({f"{r.get(a) if r.get(a) not in (None, '') else 'NA'}/{r.get(b) if r.get(b) not in (None, '') else 'NA'}" for r in relevant})
        labels.append(label + "=" + (", ".join(counts) if len(counts) <= 4 else f"varies ({len(counts)} cells; see CSV)"))
    defined = [r for r in rows if r.get("n_defined") not in (None, "")
               and _number(r.get("n_defined")) < _number(r.get("n"))]
    if defined:
        counts = sorted({f"{r['n_defined']}/{r['n']}" for r in defined})
        labels.append("defined=" + (", ".join(counts) if len(counts) <= 4 else "varies; see CSV"))
    return "; ".join(labels)


def _caption(rows, manifest, family):
    mode = manifest["mode"]
    text = f"{mode}" + (" (preview)" if mode == "partial_exploratory" else "")
    text += "; " + _support(rows) + ". "
    kinds = _unique(rows, "interval_kind")
    text += "Intervals: " + (", ".join(kinds) if kinds else "none supplied") + ". "
    if any((r.get("interval_kind") == "crossed_approximate"
            or r.get("interval_kind", "").startswith("environment_only")
            or "95% CI" in r.get("interval_kind", ""))
           and all(math.isfinite(_number(r.get(k))) for k in ("value", "ci_low", "ci_high"))
           for r in rows):
        text += "95% descriptive CI, conditional on frozen selection; not simultaneous/FDR. "
    if any(math.isfinite(_number(r.get("sd"))) for r in rows):
        text += "SD is descriptive, not CI. "
    phases = _unique(rows, "phase")
    if phases:
        text += "Phase: " + ", ".join(phases) + ". "
    supports = []
    for (kind,), group in _groups(rows, ("support_kind",)):
        values = [_number(r.get("support")) for r in group]
        values = [v for v in values if math.isfinite(v)]
        if values:
            low, high = min(values), max(values)
            span = f"{low:g}" if low == high else f"{low:g}–{high:g}"
            supports.append((kind or "support").replace("_", " ") + "=" + span)
    if supports:
        text += "Support: " + (", ".join(supports) if len(supports) <= 3 else "varies; see CSV") + ". "
    if family == "ratio" and all(r.get("metric") in {"pass95_count", "fail95_count", "undefined_count"} for r in rows):
        text += "Counts of planned cells: pass95 means R >= .95; fail95 means R < .95; undefined means R unavailable. The .95 criterion is a descriptive point screen, not a safety guarantee."
    else:
        text += NOTES.get(family, "Missing values remain NA; no interpolation.")
    if any(not math.isfinite(_number(r.get("value"))) for r in rows):
        text += " Per-point support and full status/NA reasons are in CSV."
    if any(r.get("source_x_label") for r in rows):
        text += " Fixed controls: x is method category, not training history; source_x/source_x_label retain recorded coordinates."
    if any(r.get("reference_projection") for r in rows):
        text += " Fixed references: n is unchanged; provenance in CSV."
    contrasts = sorted({_method_label(r) for r in rows if r.get("panel") == "W7" and r.get("family") == "contrasts"})
    if contrasts:
        text += " " + "; ".join(contrasts) + ". Joint reward penalty scaling; not constrained PPO."
    return text


def _save(fig, output, identity, rows, manifest, formats, caption, **extra):
    figure_id = _id(identity)
    width, height = map(float, fig.get_size_inches())
    if width > 7.2:
        caption += f" Working atlas size: {width:g} × {height:g} in; resizing requires visual review."
    provenance = "Source mode: " + str(manifest.get("mode", "unspecified"))
    if manifest.get("source_manifest"):
        provenance += "; atlas source: " + str(manifest["source_manifest"].get("mode", "unspecified"))
    if width > 7.2:
        provenance += f"; working atlas {width:g} × {height:g} in; resize requires visual review."
    fig.text(.02, -.025, textwrap.fill(provenance, int(width * 15)), fontsize=8)
    files = []
    try:
        for ext in formats:
            name = figure_id + "." + ext
            fig.savefig(output / name, dpi=180, bbox_inches="tight")
            files.append(name)
    finally:
        _plt().close(fig)
    _csv(output / (figure_id + ".csv"), rows, list(dict.fromkeys(POINT_COLUMNS + [k for r in rows for k in r])))
    meta = dict(schema_version=1, id=figure_id, identity=identity,
                mode=manifest.get("mode"), source_manifest=manifest, caption=caption,
                question=FAMILIES.get(identity["family"], identity["family"]),
                render_filters=manifest.get("render_filters", {}),
                figure_size_inches=[width, height],
                size_policy="working atlas; resize requires review" if width > 7.2 else "journal-width target; visual review required",
                plotted_rows=len(rows), units=_unique(rows, "units"),
                phases=_unique(rows, "phase"), statuses=_unique(rows, "status"),
                support_kinds=_unique(rows, "support_kind"),
                fixed_reference_sources=[dict(zip(("source_rule", "source_cohort", "source_estimand"), key))
                                         for key, _ in _groups([r for r in rows if r.get("reference_projection")],
                                                              ("source_rule", "source_cohort", "source_estimand"))],
                files=files, points=figure_id + ".csv", **extra)
    _json(output / (figure_id + ".json"), meta)
    return dict(id=figure_id, family=identity["family"], files=files,
                points=figure_id + ".csv", metadata=figure_id + ".json", caption=caption)


def _groups(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    return sorted(groups.items())


def _axes(count):
    columns = min(2, count)
    fig, axes = _plt().subplots(math.ceil(count / columns), columns,
                               figsize=(3.4 if columns == 1 else 7.2, 2.7 * math.ceil(count / columns) + 1.2), squeeze=False)
    flat = axes.ravel()
    for ax in flat[count:]:
        ax.set_visible(False)
    return fig, flat[:count]


def _finish(fig, axes, title, caption, *, right=1):
    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label, handle)
    columns = 1 if fig.get_figwidth() < 5 else 3
    title = textwrap.fill(title, int(fig.get_figwidth() * 12))
    labels = [textwrap.fill(label, 28) for label in handles]
    heading = fig.text(.5, 1, title, fontsize=10, ha="center", va="top")
    legend = None
    if handles:
        legend = fig.legend(handles.values(), labels, loc="upper center",
                            ncol=min(columns, len(handles)), frameon=False,
                            borderaxespad=0)
    wrapped = textwrap.fill(caption, max(45, int(fig.get_figwidth() * 15)))
    footer = fig.text(.02, 0, wrapped, fontsize=8, va="bottom")
    # Measure actual text, including wrapped legend entries, in physical inches.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_height = heading.get_window_extent(renderer).height / fig.dpi
    legend_height = legend.get_window_extent(renderer).height / fig.dpi if legend else 0
    footer_height = footer.get_window_extent(renderer).height / fig.dpi
    top = title_height + legend_height + .28
    bottom = footer_height + .18
    grid_rows = axes[0].get_subplotspec().get_gridspec().nrows
    height = 2.7 * grid_rows + top + bottom
    fig.set_figheight(height)
    heading.set_position((.5, 1 - .06 / height))
    if legend:
        legend.set_bbox_to_anchor((.5, 1 - (title_height + .14) / height),
                                  transform=fig.transFigure)
    footer.set_y(.04 / height)
    # Constrained layout owns inter-row title/tick/axis-label spacing. The
    # measured header and caption have their own room outside its rectangle.
    fig.set_layout_engine("constrained", rect=(0, bottom / height, right,
                                               1 - (top + bottom) / height))


def _method_order(method):
    order = ("Graph", "Node", "Grid", "fixed controls", "H1", "H2", "H4", "H8", "B1", "B2", "noop", "original")
    method = canonical_method(method)
    return (order.index(method) if method in order else len(order), method)


def _method_label(row):
    method = row["method"]
    if row.get("panel") == "W7" and row.get("family") == "contrasts" and method == "Graph−Graph":
        return f"Graph penalty×{row.get('cohort')} minus ×1"
    return canonical_method(method)


def _checkpoint_display_row(row):
    if (row.get("x_label") in {"checkpoint environment steps", "checkpoint steps", "env_steps"}
            and row.get("training_line") in (None, "", "all") and row.get("x") in (None, "")):
        return dict(row, source_x=row.get("x"), source_x_label=row.get("x_label"),
                    x=canonical_method(row["method"]), x_label="fixed control (method category)")
    return row


def _metric_label(metric):
    return {"validation_V": "Validation PV speed V", "validation_R": "Validation background ratio R",
            "validation_feasible": "Recorded feasibility"}.get(metric, metric.replace("_", " ").replace("/", " / "))


def _series_arrays(rows, domain):
    """Align to supplied facet grid, preserving explicit and absent-series gaps."""
    by_x = {}
    for row in rows:
        x = row.get("x", "")
        if x in by_x:
            raise ValueError(f"ambiguous repeated x={x!r}; supply distinct facets/training lines")
        by_x[x] = row
    return np.array([_number(by_x.get(x, {}).get("value")) for x in domain]), by_x


def _line_panel(ax, rows, family, letter):
    domain = sorted({r.get("x", "") for r in rows}, key=lambda x: (not math.isfinite(_number(x)), _number(x) if math.isfinite(_number(x)) else str(x)))
    numeric = all(math.isfinite(_number(x)) for x in domain) and "category" not in rows[0].get("x_label", "")
    x = np.array([float(v) for v in domain]) if numeric else np.arange(len(domain))
    missing = set()
    for (method, training), series in sorted(_groups(rows, ("method", "training_line")),
                                             key=lambda item: (_method_order(item[0][0]), item[0][1])):
        y, by_x = _series_arrays(series, domain)
        label = _method_label(series[0]) + (f" / {training}" if training and training != "all" else "")
        style = method_style(method)
        if training and training != "all":
            lines = _unique([r for r in rows if r.get("method") == method and r.get("training_line") != "all"], "training_line")
            style["linestyle"] = ["-", "--", "-.", ":"][lines.index(training) % 4]
            style["linewidth"] = 1.2
        if not numeric or (family in {"state", "mechanism"} and rows[0].get("x_label") not in {"density", "lattice density", "steps", "step"}):
            style["linestyle"] = "none"
        ax.plot(x, y, label=label, **style)
        for pos, key in zip(x, domain):
            row = by_x.get(key)
            if row is None:
                continue
            value, low, high = (_number(row.get(k)) for k in ("value", "ci_low", "ci_high"))
            if all(math.isfinite(v) for v in (value, low, high)):
                # Draw supplied endpoints, even if the point falls outside them.
                ax.vlines(pos, low, high, color=style["color"], linewidth=1.2)
                ax.plot([pos, pos], [low, high], marker="_", color=style["color"], linestyle="none")
            if not math.isfinite(value):
                missing.add(pos)
            elif family == "laps" and row.get("support_kind") == "episodes_with_completed_lap" and row.get("support") not in (None, ""):
                ax.annotate(f"{row['support']}/{row.get('n') or 'NA'}", (pos, value),
                            xytext=(0, 5), textcoords="offset points", fontsize=8)
    # Matplotlib ignores x coordinates paired with NaN y when autoscaling.
    # Keep the complete supplied domain, including missing edge conditions.
    if len(x):
        pad = max(.05 * (x[-1] - x[0]), .5 if not numeric else .02)
        ax.set_xlim(x[0] - pad, x[-1] + pad)
    for pos in sorted(missing):
        ax.annotate("NA", (pos, .03), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=8, clip_on=True, annotation_clip=True)
    if rows[0].get("units") == "boolean":
        ax.set_yticks([0, 1])
        ax.set_ylim(-.12, 1.12)
    if numeric and rows[0].get("x_label") in {"density", "lattice density"}:
        limits = ax.get_xlim()
        marks = [v for v in (.10, .20, .30, .40) if limits[0] <= v <= limits[1]]
        if marks:
            ax.plot(marks, [0] * len(marks), transform=ax.get_xaxis_transform(),
                    marker="|", linestyle="none", color="#555555", markersize=7,
                    label="training densities", clip_on=False)
        ax.set_xlim(limits)
    if family == "ratio" and rows[0].get("metric") in {"R", "R_min"}:
        for value, label, line in ((1, "R=1", ":"), (.95, "screen .95", "--")):
            ax.axhline(value, color="#444444", linestyle=line, linewidth=1.2)
            ax.text(1, value, label, transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8)
    if family == "contrasts":
        ax.axhline(0, color="#555555", linewidth=1.2, linestyle=":")
    if not numeric:
        ax.set_xticks(x, domain, rotation=35, ha="right")
    ax.set_xlabel(textwrap.fill(rows[0].get("x_label") or "recorded category", 38))
    coded = "type" in rows[0]["metric"].lower() and "mean" in rows[0]["metric"].lower()
    ax.set_ylabel(textwrap.fill(("Native code summary (not a physical quantity): " if coded else "") + _metric_label(rows[0]["metric"]) + (f" [{rows[0]['units']}]" if rows[0].get("units") else ""), 28))
    ax.text(0, 1.04, letter, transform=ax.transAxes, fontweight="bold")


def _curve_pages(rows, output, manifest, formats, page_size):
    artifacts = []
    summary_keys = tuple(k for k in BASE if k != "estimand")
    summary_scopes = {tuple(r.get(k, "") for k in summary_keys) for r in rows
                      if r.get("panel") in FRESH_PANELS and r.get("training_line") == "all"
                      and canonical_method(r["method"]) in {"Graph", "Node", "Grid"}}
    for base, group in _groups(rows, BASE):
        identity = dict(zip(BASE, base))
        checkpoint = identity["x_label"] in {"checkpoint environment steps", "checkpoint steps", "env_steps"}
        # Density is a facet for lag/category/steps curves, never pooled away.
        facet_keys = FACETS + _density_keys(group) + _candidate_keys(group)
        individual = any(r.get("training_line") not in ("", "all") for r in group)
        if individual or checkpoint or identity["family"] == "contrasts":
            facet_keys += ("method",)
            if any(len(_unique(rr, "training_line")) > 4 for _, rr in _groups(group, facet_keys)):
                facet_keys += ("training_line",)
        variants = [("complete", group, facet_keys)]
        means = [r for r in group if r.get("training_line") == "all"
                 and canonical_method(r["method"]) in {"Graph", "Node", "Grid"}]
        if individual and not means and tuple(identity[k] for k in summary_keys) in summary_scopes:
            variants = [("individual-lines", group, facet_keys)]
        if means and identity["panel"] in FRESH_PANELS and (individual or identity["estimand"] == "equal frozen whole-line means; no training-line pairing"):
            cell_keys = FACETS + ("density", "candidate_id", "model_id")
            if identity["x_label"] in {"density", "lattice density"}:
                cell_keys += ("x",)
            mean_cells = {tuple(r.get(k, "") for k in cell_keys) for r in means}
            summary = [r for r in group if r.get("training_line") in ("", "all")
                       and tuple(r.get(k, "") for k in cell_keys) in mean_cells]
            variants = [("family-summary", summary, FACETS + _density_keys(summary) + _candidate_keys(summary))]
            if individual:
                variants.append(("individual-lines", [r for r in group if r.get("training_line") != "all"
                                                     or canonical_method(r["method"]) not in {"Graph", "Node", "Grid"}], facet_keys))
        methods = {canonical_method(r["method"]) for r in group}
        if not individual and len(methods) > 4:
            for name, chosen in (("architectures", {"Graph", "Node", "Grid", "H2"}),
                                 ("controls", {"B1", "B2", "noop", "original"})):
                subset = [r for r in group if canonical_method(r["method"]) in chosen]
                if subset:
                    variants.append((name, subset, facet_keys))
        for layout, subset, keys in variants:
            if checkpoint:
                # Only fixed records with absent steps receive categorical coordinates.
                subset = [_checkpoint_display_row(r) for r in subset]
                grouped = defaultdict(list)
                for r in subset:
                    facet = tuple("fixed controls" if k == "method" and r.get("source_x_label") else r.get(k, "") for k in keys)
                    grouped[facet].append(r)
                facets = list(grouped.items())
            else:
                facets = _groups(subset, keys)
            if "method" in keys:
                mi = keys.index("method")
                facets.sort(key=lambda item: (item[0][:mi], _method_order(item[0][mi]), item[0][mi + 1:]))
            common_limits = None
            if identity["units"] != "boolean":
                values = [_number(r.get(k)) for r in subset for k in ("value", "ci_low", "ci_high")]
                finite = [v for v in values if math.isfinite(v)]
                if identity["family"] == "ratio" and identity["metric"] in {"R", "R_min"}:
                    finite.extend([.95, 1])
                elif identity["family"] == "contrasts":
                    finite.append(0)
                if finite:
                    low, high = min(finite), max(finite)
                    pad = .05 * (high - low or max(abs(low), 1))
                    common_limits = (low - pad, high + pad)
            for start in range(0, len(facets), page_size):
                page = facets[start:start + page_size]
                used = [r for _, rr in page for r in rr]
                if not any(math.isfinite(_number(r.get("value"))) for r in used):
                    continue
                fig, axes = _axes(len(page))
                if identity["units"] == "boolean":
                    fig.set_figheight(1.8 * math.ceil(len(page) / min(2, len(page))) + 1.8)
                for i, (ax, (facet, rr)) in enumerate(zip(axes, page)):
                    _line_panel(ax, rr, identity["family"], chr(97 + i))
                    if common_limits is not None:
                        ax.set_ylim(*common_limits)
                    ax.set_title(textwrap.fill("; ".join(str(v) if k == "method" else f"{k}={v}"
                                                        for k, v in zip(keys, facet) if v), 34))
                caption = _caption(used, manifest, identity["family"])
                title = " · ".join(str(identity[k] or "all") for k in ("family", "panel", "cohort", "rule", "road_length"))
                _finish(fig, axes, title, caption)
                artifacts.append(_save(fig, output, dict(identity, layout=layout, page=start // page_size + 1,
                                                        facets=[list(f) for f, _ in page]), used, manifest, formats, caption))
    return artifacts


def _heatmap_pages(rows, output, manifest, formats, page_size):
    artifacts = []
    keys = BASE + ("window", "region", "lane", "vehicle_group")
    for base, group in _groups(rows, keys):
        if not all(math.isfinite(_number(r.get("density"))) and math.isfinite(_number(r.get("fraction")))
                   and _number(r.get("x")) == _number(r.get("density")) for r in group):
            continue
        if len(_unique(group, "density")) < 2 or len(_unique(group, "fraction")) < 2:
            continue
        facets = sorted(_groups(group, ("method", "training_line") + _candidate_keys(group)),
                        key=lambda item: (_method_order(item[0][0]), item[0][1:]))
        values = [_number(r.get("value")) for r in group]
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            continue
        identity = dict(zip(keys, base))
        from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
        diverging = identity["family"] == "contrasts"
        if diverging:
            bound = max(abs(min(finite)), abs(max(finite)), 1e-12)
            norm = TwoSlopeNorm(0, -bound, bound)
            cmap = LinearSegmentedColormap.from_list("effect", ["#315B87", "#F4F1EA", "#B65C45"])
        else:
            norm = Normalize(min(finite), max(finite) if max(finite) != min(finite) else min(finite) + 1)
            cmap = _plt().get_cmap("cividis").copy()
        cmap.set_bad("#f0f0f0")
        for start in range(0, len(facets), page_size):
            page = facets[start:start + page_size]
            used = [r for _, rr in page for r in rr]
            fig, axes = _axes(len(page))
            for i, (ax, (facet, rr)) in enumerate(zip(axes, page)):
                method, line, *candidate = facet
                densities = sorted(_unique(group, "density"), key=float)
                fractions = sorted(_unique(group, "fraction"), key=float)
                matrix = np.full((len(fractions), len(densities)), np.nan)
                seen = set()
                for row in rr:
                    cell = (fractions.index(row["fraction"]), densities.index(row["density"]))
                    if cell in seen:
                        raise ValueError("duplicate heatmap cell; renderer cannot aggregate")
                    seen.add(cell)
                    matrix[cell] = _number(row.get("value"))
                image = ax.imshow(np.ma.masked_invalid(matrix), origin="lower", interpolation="nearest", aspect="auto", cmap=cmap, norm=norm)
                for y, x in zip(*np.where(~np.isfinite(matrix))):
                    ax.text(x, y, "NA", ha="center", va="center", fontsize=8)
                ax.set_xticks(range(len(densities)), densities, rotation=45)
                ax.set_yticks(range(len(fractions)), fractions)
                ax.set(xlabel="lattice density", ylabel="AV fraction")
                ax.set_title(textwrap.fill(f"{chr(97 + i)}  {canonical_method(method)} {line} {' / '.join(candidate)}", 50), color=method_style(method)["color"])
                ax.grid(False)
            caption = _caption(used, manifest, identity["family"])
            _finish(fig, axes, identity["metric"] + " · density × fraction", caption, right=.90)
            color_ax = fig.add_axes((.92, .40, .015, .35))
            fig.colorbar(image, cax=color_ax, label=identity["units"] or identity["metric"])
            artifacts.append(_save(fig, output, dict(identity, layout="heatmap", page=start // page_size + 1,
                                                    facets=[list(f) for f, _ in page]), used, manifest, formats, caption))
    return artifacts


def _latex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "−": r"\ensuremath{-}", "×": r"\ensuremath{\times}",
                    "≤": r"\ensuremath{\leq}", "≥": r"\ensuremath{\geq}"}
    return "".join(replacements.get(c, c) for c in str(value))


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table_bundle(rows, output, name, title, columns, mode, page_rows=30, *, total_rows=None):
    # Stream even explicitly unabridged previews; never duplicate a full table.
    with (output / (name + ".csv")).open("w", newline="", encoding="utf-8") as machine, \
         (output / (name + ".md")).open("w", encoding="utf-8") as md, \
         (output / (name + ".tex")).open("w", encoding="utf-8") as tex:
        md.write(f"# {title}\n\nMode: {mode}. SD is descriptive, not CI. Blank values are NA; reasons are retained.\n\n")
        tex.write("% " + title + "; " + mode + "; SD descriptive, not CI\n")
        writer = None
        count = 0
        for count, row in enumerate(rows, 1):
            if writer is None:
                writer = csv.DictWriter(machine, list(row))
                writer.writeheader()
            writer.writerow(row)
            if (count - 1) % page_rows == 0:
                if count > 1:
                    tex.write(r"\hline\end{tabular}\par\medskip" + "\n")
                md.write(f"\nPage {(count - 1) // page_rows + 1}\n\n| " + " | ".join(map(_md, columns)) + " |\n| " + " | ".join("---" for _ in columns) + " |\n")
                tex.write(r"\begin{tabular}{" + ("p{" + str(round(.75 / len(columns), 3)) + r"\linewidth}") * len(columns) + "}\n" + r"\hline" + "\n" + " & ".join(map(_latex, columns)) + r" \\" + "\n")
            cells = [row.get(k, "") or "NA" for k in columns]
            md.write("| " + " | ".join(map(_md, cells)) + " |\n")
            tex.write(" & ".join(map(_latex, cells)) + r" \\" + "\n")
        if count:
            tex.write(r"\hline\end{tabular}" + "\n")
    total = count if total_rows is None else total_rows
    return dict(id=name, files=[name + ext for ext in (".csv", ".md", ".tex")], rows=count,
                total_rows=total, truncated=count < total, title=title)


def _compact_rows(rows):
    for row in rows:
        item = dict(row)
        item["condition"] = "; ".join(f"{k}={row[k]}" for k in ("density", "fraction", "training_line", "candidate_id", "model_id", "window", "region", "lane", "vehicle_group", "phase", "x_label", "x") if row.get(k))
        item["estimate"] = row.get("value") or "NA: " + (row.get("status") or "missing")
        if row.get("ci_low") and row.get("ci_high"):
            item["estimate"] += f" [{row['ci_low']}, {row['ci_high']}] {row.get('interval_kind', '')}"
        if row.get("sd"):
            item["estimate"] += f"; SD={row['sd']} (descriptive)"
        item["support_summary"] = _support([row]) + "; " + (row.get("support_kind") or "support") + "=" + (row.get("support") or "NA")
        yield item


def export_tables(data_dir, output_dir, *, families=None, metrics=None, table_row_limit=120, panels=None):
    """Lossless native CSV snapshots and bounded previews per scope.

    panels filters normalized panel names in point previews, never native CSVs.
    Native DATA publishes CSV atomically. Same-filesystem hardlinks preserve
    that inode even after DATA publishes a newer snapshot, without another
    large byte copy. Cross-filesystem exports copy the complete file instead.
    None explicitly requests streamed unabridged previews (one reread per scope).
    """
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    if table_row_limit is not None and (not isinstance(table_row_limit, int) or table_row_limit < 1):
        raise ValueError("table_row_limit must be positive or None")
    output = _new_dir(output_dir)
    tables = []
    for source in sorted(data.glob("*.csv")):
        target = output / source.name
        storage = "hardlink"
        try:
            target.hardlink_to(source)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, target)
            storage = "copy"
        tables.append(dict(id=source.stem, files=[source.name], storage=storage,
                           title="Complete native " + source.stem))
    for name in ("coverage", "selections"):
        preview, total, fields = [], 0, []
        for row in _iter_csv(data / (name + ".csv")):
            total += 1
            if not fields:
                fields = list(row)
            if table_row_limit is not None and len(preview) < table_row_limit:
                preview.append(row)
        if total:
            shown = total if table_row_limit is None else len(preview)
            for start in range(2, max(3, len(fields)), 5):
                rows = _iter_csv(data / (name + ".csv")) if table_row_limit is None else preview
                tables.append(_table_bundle(rows, output, name + f"-{start // 5 + 1}", f"{name}: first {shown}/{total} rows; complete {name}.csv", fields[:2] + fields[start:start + 5], manifest["mode"], total_rows=total))
    keys = ("family", "panel", "cohort", "rule", "road_length")
    counts, previews = defaultdict(int), defaultdict(list)
    panel_filter = set(panels) if panels is not None else None
    supplied_panels = set()
    for row in _iter_csv(data / "points.csv"):
        supplied_panels.add(row["panel"])
        if panel_filter is not None and row["panel"] not in panel_filter:
            continue
        if (families is not None and row["family"] not in families) or (metrics is not None and row["metric"] not in metrics):
            continue
        key = tuple(row.get(k, "") for k in keys)
        counts[key] += 1
        if table_row_limit is not None and len(previews[key]) < table_row_limit:
            previews[key].append(row)
    if panel_filter is not None and panel_filter - supplied_panels:
        raise ValueError(f"unknown panels: {sorted(panel_filter - supplied_panels)}")
    for key, total in sorted(counts.items()):
        identity = dict(zip(keys, key), layout="table")
        if table_row_limit is None:
            rows = (r for r in _iter_csv(data / "points.csv") if tuple(r.get(k, "") for k in keys) == key
                    and (metrics is None or r["metric"] in metrics))
            shown = total
        else:
            rows, shown = previews.pop(key), min(total, table_row_limit)
        tables.append(_table_bundle(_compact_rows(rows), output, _id(identity), " · ".join(v or "all" for v in key) + f"; first {shown}/{total} rows; complete points.csv",
                                    ["method", "condition", "metric", "estimate", "units", "support_summary", "status"], manifest["mode"], total_rows=total))
    _json(output / "tables.json", dict(mode=manifest["mode"], source_manifest=manifest, tables=tables,
                                      filters=dict(families=sorted(families) if families is not None else None, metrics=sorted(metrics) if metrics is not None else None, table_row_limit=table_row_limit, panels=sorted(panel_filter) if panel_filter is not None else None),
                                      deferred=["T4 literature comparison requires manual sources"]))
    return tables


def _tradeoff_pages(rows, output, manifest, formats, page_size, *, metrics=None):
    """Match already-summarized values by full condition; never fit or reduce."""
    pv_names = PV_METRICS
    cost_names = COST_METRICS
    pv = [r for r in rows if r.get("metric") in pv_names and r.get("family") == "mobility"]
    costs = [r for r in rows if (r.get("family") == "ratio" or r.get("metric") in cost_names) and (metrics is None or r.get("metric") in metrics)]
    match_keys = ("panel", "cohort", "rule", "road_length", "density", "fraction", "method", "training_line", "model_id", "candidate_id", "x", "x_label")
    cost_keys = tuple(dict.fromkeys(match_keys + BASE + FACETS))
    if any(len(group) > 1 for _, group in _groups(costs, cost_keys)):
        raise ValueError("ambiguous duplicate cost rows; renderer cannot aggregate tradeoff rows")
    lookup = defaultdict(list)
    for row in pv:
        lookup[tuple(row.get(k, "") for k in match_keys)].append(row)
    paired = []
    for cost in costs:
        matches = lookup.get(tuple(cost.get(k, "") for k in match_keys), [])
        # Distinct supplied PV estimands get separate views, never an implicit choice.
        for _, candidates in _groups(matches, ("metric", "units", "estimand", "phase")):
            if len(candidates) != 1:
                raise ValueError("ambiguous mobility match; renderer cannot aggregate tradeoff rows")
            mobility = candidates[0]
            if math.isfinite(_number(mobility.get("value"))) and math.isfinite(_number(cost.get("value"))):
                paired.append((mobility, cost))
    grouped = defaultdict(list)
    for mobility, cost in paired:
        grouped[tuple(cost.get(k, "") for k in BASE + ("window", "region", "vehicle_group", "lane")) + tuple(mobility.get(k, "") for k in ("metric", "units", "estimand", "phase"))].append((mobility, cost))
    artifacts = []
    for key, pairs in sorted(grouped.items()):
        identity = dict(zip(BASE + ("window", "region", "vehicle_group", "lane", "pv_metric", "pv_units", "pv_estimand", "pv_phase"), key))
        identity["family"] = "tradeoff"
        facets = defaultdict(list)
        facet_keys = ("fraction", "training_line") + _candidate_keys([cost for _, cost in pairs])
        for pair in pairs:
            facets[tuple(pair[1].get(k, "") for k in facet_keys)].append(pair)
        facets = sorted(facets.items())
        for start in range(0, len(facets), page_size):
            page = facets[start:start + page_size]
            fig, axes = _axes(len(page))
            used = []
            for i, (ax, (facet, entries)) in enumerate(zip(axes, page)):
                for mobility, cost in entries:
                    style = method_style(cost["method"])
                    ax.plot(_number(cost["value"]), _number(mobility["value"]), linestyle="none",
                            marker=style["marker"], color=style["color"], markerfacecolor=style["markerfacecolor"],
                            label=canonical_method(cost["method"]))
                    ax.annotate(cost.get("density", ""), (_number(cost["value"]), _number(mobility["value"])),
                                xytext=(3, 3), textcoords="offset points", fontsize=8)
                    used.extend([mobility, cost])
                ax.set(xlabel=_metric_label(identity["metric"]) + (f" [{identity['units']}]" if identity["units"] else ""),
                       ylabel=_metric_label(entries[0][0]["metric"]) + " [" + entries[0][0].get("units", "") + "]",
                       title=textwrap.fill(f"{chr(97 + i)}  f={facet[0]} · line={facet[1] or 'fixed'} · {' / '.join(facet[2:])}", 50))
                ax.locator_params(axis="x", nbins=4)
            caption = _caption(used, manifest, "windows") + " Labels are density; points are matched supplied summaries, not causal or Pareto claims."
            _finish(fig, axes, "PV mobility versus background/local cost", caption)
            artifacts.append(_save(fig, output, dict(identity, layout="scatter", page=start // page_size + 1,
                                                    facets=[list(f) for f, _ in page]), used, manifest, formats, caption))
    return artifacts


def render_conceptual(output_dir, *, formats=("png", "pdf")):
    """Draw F1, explicitly conceptual, into a new output directory."""
    return _conceptual(output_dir, formats, {"mode": "conceptual"})


def _conceptual(output_dir, formats, manifest):
    output = _new_dir(output_dir)
    formats = _formats(formats)
    plt = _plt()
    from matplotlib.patches import Rectangle
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.5), gridspec_kw={"height_ratios": [1.7, 1.1, 1.2]})
        road, enc, phases = axes
        road.set(xlim=(-18, 40), ylim=(-1.6, 6), xlabel="conceptual longitudinal offset from PV [cells]", ylabel="physical lane")
        road.set_yticks(range(4))
        road.set_title("a  Physical road, measurement, actuation and per-active-AV sensing")
        for lane in range(4):
            road.axhline(lane - .45, color="#777777", linewidth=.6)
        road.add_patch(Rectangle((-10, -.45), 40, 3.9, facecolor="#147D92", alpha=.08, edgecolor="#147D92"))
        road.plot(0, 1, marker="*", markersize=14, color="#111111", label="PV")
        road.plot([8, -5, 24], [2, 0, 3], "s", color="#3850A0", label="AV (conceptual)")
        road.plot([-7, 15, 27], [3, 1, 0], "o", markerfacecolor="none", color="#666666", label="HDV (conceptual)")
        road.plot([-10, 30], [4, 4], color="#147D92", linewidth=1.4)
        road.text(-10, 4.13, "Measurement fixed_front_2: [-10,+30], all lanes; background excludes PV", fontsize=9)
        road.plot([-10, 30], [4.9, 4.9], color="#80551E", linewidth=1.4, linestyle="--")
        road.text(-10, 5.03, "Actuation window: configured -10/+30 (separate scope)", fontsize=9)
        road.annotate("Per-active-AV sensor neighbourhood\nsensor_distance=60; schematic, not to scale", xy=(8, 2), xytext=(8, -.95),
                      arrowprops=dict(arrowstyle="->", color="#3850A0"), ha="center", fontsize=9,
                      bbox=dict(facecolor="white", edgecolor="#cccccc"))
        road.legend(loc="upper left", bbox_to_anchor=(0, -.22), ncol=3, frameon=False)
        for ax in (enc, phases):
            ax.set(xlim=(0, 1), ylim=(0, 1))
            ax.axis("off")
        enc.set_title("b  Alternative encoders, one common lateral interface", loc="left")
        def box(ax, x, y, label, color="#444444"):
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    bbox=dict(boxstyle="round,pad=.4", facecolor="white", edgecolor=color))
        box(enc, .11, .5, "Observation\n+ legal mask")
        for y, method in ((.85, "Graph"), (.5, "Node"), (.15, "Grid")):
            box(enc, .39, y, method + " encoder", method_style(method)["color"])
            enc.annotate("", (.30, y), (.20, .5), arrowprops=dict(arrowstyle="->", color="#777777"))
            enc.annotate("", (.64, .5), (.48, y), arrowprops=dict(arrowstyle="->", color="#777777"))
        box(enc, .73, .5, "Common lateral\naction interface")
        box(enc, .94, .5, "Native\nsimulator")
        enc.annotate("", (.89, .5), (.83, .5), arrowprops=dict(arrowstyle="->"))
        phases.set_title("c  Recorded step phases (order, not causal evidence)", loc="left")
        box(phases, .10, .5, "Pre-step\ngeometry t−1")
        box(phases, .39, .8, "Native eligibility\nfrom pre-step geometry")
        box(phases, .39, .2, "AV lateral decisions\nfrom pre-action state")
        box(phases, .72, .5, "After all lateral\nbefore motion")
        box(phases, .94, .5, "Post-step t\nfinal speeds")
        for target in ((.26, .8), (.26, .2)):
            phases.annotate("", target, (.20, .5), arrowprops=dict(arrowstyle="->"))
        for source in ((.53, .8), (.53, .2)):
            phases.annotate("", (.61, .5), source, arrowprops=dict(arrowstyle="->"))
        phases.annotate("", (.88, .5), (.83, .5), arrowprops=dict(arrowstyle="->"))
        caption = "Conceptual only: vehicle positions are illustrative. Sensor neighbourhood belongs to each active AV, not the PV. Native eligibility uses pre-step geometry; an accepted AV change does not open same-step native eligibility."
        fig.text(.03, .01, textwrap.fill(caption, 150), fontsize=8)
        fig.tight_layout(rect=(0, .07, 1, 1), h_pad=2.5)
        return _save(fig, output, dict(family="conceptual", metric="F1-scopes-phases"), [],
                     dict(manifest, source="FIGURE_GUIDE; configured actuator -10/+30 and sensor_distance=60"), formats, caption)


def _recorded(entry):
    arrays = entry.get("arrays", entry)
    analysis = entry.get("analysis", {})
    trace = entry.get("trace", analysis.get("pv", {}).get("trace"))
    speed = np.asarray(arrays.get("local_speed"))
    types = np.asarray(arrays.get("local_type"))
    if speed.ndim != 3 or speed.shape[1:] != (41, 4):
        raise ValueError("recorded local_speed must have native shape (T,41,4)")
    if types.shape != speed.shape or not np.isin(types, [-2, -1, 0, 1]).all():
        raise ValueError("recorded local_type must match speed and native type codes")
    if not np.isin(speed, [-2, -1, 0, 1, 2, 3, 4, 5]).all():
        raise ValueError("background speed must be 0..5, empty=-1, PV=-2")
    if not np.array_equal(speed < 0, types < 0) or not np.array_equal(speed[speed < 0], types[speed < 0]):
        raise ValueError("local_speed/local_type sentinels disagree")
    steps = np.asarray(arrays.get("map_steps", np.arange(1, len(speed) + 1)))
    if steps.shape != (len(speed),) or not np.all(np.diff(steps) > 0):
        raise ValueError("map_steps must be a strictly increasing recorded step axis")
    if not all(t in steps for t in (1, 1000, 2000)):
        raise ValueError("recorded storyboard steps 1/1000/2000 unavailable")
    if not isinstance(trace, dict):
        raise ValueError("recorded PV trace is required")
    trace = {key: np.asarray(trace.get(key, [])) for key in ("step", "speed", "lane", "cumulative_distance")}
    if not len(trace["step"]) or any(v.shape != trace["step"].shape for v in trace.values()):
        raise ValueError("PV trace requires aligned step/speed/lane/cumulative_distance")
    if not np.all(np.diff(trace["step"]) > 0) or not np.isin(trace["lane"], range(4)).all():
        raise ValueError("PV trace needs increasing steps and physical lanes 0..3")
    if not np.all(np.isfinite(trace["speed"])) or np.any((trace["speed"] < 0) | (trace["speed"] > 6)):
        raise ValueError("PV speed must be 0..6")
    return speed, types, steps, trace


def _complete_cell(evidence):
    # Validate DATA's declaration, never repeat the frozen scientific selection.
    return (isinstance(evidence, dict) and evidence.get("status") == "selected"
            and _number(evidence.get("n")) == 90
            and evidence.get("rule") == "frozen_median_line_then_environment")


def render_recorded_illustration(pair, output_dir, *, identity, selection_evidence,
                                 mode="final", formats=("png", "pdf"), source_manifest=None):
    """Draw DATA-selected Graph/H2 maps, physical-lane traces and storyboard.

    Refuses partial cells and absent selection evidence. DATA owns verification
    of the frozen median rule. No spatial_hist conversion, selection or inference.
    CSV map rows store one exact 41-cell speed/type vector per step and lane.
    """
    if mode != "final" or not _complete_cell(selection_evidence):
        raise ValueError("representative illustration requires final cell and DATA selection evidence")
    if set(pair) != {"Graph", "H2"}:
        raise ValueError("illustration requires the supplied paired Graph and H2 episodes")
    recorded = {method: _recorded(pair[method]) for method in ("Graph", "H2")}
    output = _new_dir(output_dir)
    formats = _formats(formats)
    plt = _plt()
    manifest = dict(mode=mode, selection_evidence=selection_evidence, identity=identity, source_manifest=source_manifest,
                    episode_ids={method: pair[method].get("episode_id", pair[method].get("analysis", {}).get("episode")) for method in pair})
    base = dict(identity, family="illustrations")
    conditions = (f"L={identity['road_length']}, ρ={identity['density']}, f={identity['fraction']}"
                  if all(identity.get(k) not in (None, "") for k in ("road_length", "density", "fraction"))
                  else str(identity.get("cell", "")))
    caption = "final; " + _support([selection_evidence if isinstance(selection_evidence, dict) else identity]) + ". Frozen DATA-selected paired illustration; not population evidence. "
    caption += "Background speed 0..5 cells/step; empty −1 is white, PV −2 a black star; AV outlined, HDV unoutlined. All four physical lanes. "
    caption += "Maps: recorded post-step local window [-10,+30]; PV speed 0..6. See selection evidence and episode IDs in metadata."
    artifacts = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(4, 2, figsize=(11.8, 11), squeeze=False)
        used = []
        cmap = plt.get_cmap("cividis").copy()
        cmap.set_bad("white")
        for col, (method, (speed, types, steps, trace)) in enumerate(recorded.items()):
            for lane in range(4):
                ax = axes[lane, col]
                slab = speed[:, :, lane].T
                image = ax.imshow(np.ma.masked_less(slab, 0), origin="lower", aspect="auto", interpolation="nearest",
                                  cmap=cmap, vmin=0, vmax=5, extent=(-.5, len(steps) - .5, -10.5, 30.5))
                yy, xx = np.where(slab == -2)
                ax.scatter(xx, yy - 10, marker="*", c="black", s=8, linewidths=0)
                yy, xx = np.where(types[:, :, lane].T == 1)
                ax.scatter(xx, yy - 10, marker="s", facecolors="none", edgecolors="#333333", s=3, linewidths=.25)
                ticks = np.unique(np.linspace(0, len(steps) - 1, min(5, len(steps))).astype(int))
                ax.set_xticks(ticks, steps[ticks])
                ax.set(xlabel="recorded step (rows; no interpolation)", ylabel="offset [cells]",
                       title=f"{chr(97 + lane * 2 + col)}  {method} · physical lane {lane}")
                ax.grid(False)
                for t, step in enumerate(steps):
                    used.append(dict(method=method, phase="post-step", lane=lane, x=int(step), x_label="step",
                                     metric="recorded_local_speed_type", speed_cells=json.dumps(speed[t, :, lane].tolist()),
                                     type_cells=json.dumps(types[t, :, lane].tolist()), offsets="-10..30"))
        fig.suptitle("Recorded local x–t maps · " + conditions, fontsize=10)
        fig.text(.02, .01, textwrap.fill(caption, 155), fontsize=8)
        fig.tight_layout(rect=(0, .10, .94, .97))
        color_ax = fig.add_axes((.95, .20, .012, .55))
        fig.colorbar(image, cax=color_ax, label="background speed [cells/step]")
        artifacts.append(_save(fig, output, dict(base, metric="maps", layout="physical-lanes"), used, manifest, formats, caption))
        fig, axes = plt.subplots(3, 2, figsize=(11.8, 8), squeeze=False)
        used = []
        for col, (method, (_, _, _, trace)) in enumerate(recorded.items()):
            for row_index, (key, unit) in enumerate((("speed", "cells/step"), ("lane", "physical lane"), ("cumulative_distance", "cells"))):
                ax = axes[row_index, col]
                gaps = np.flatnonzero(np.diff(trace["step"]) > 1) + 1
                plot_x = np.insert(trace["step"].astype(float), gaps, np.nan)
                plot_y = np.insert(trace[key].astype(float), gaps, np.nan)
                ax.plot(plot_x, plot_y, **method_style(method), markevery=max(1, len(trace["step"]) // 25),
                        drawstyle="steps-post" if key == "lane" else "default", label=method)
                ax.set(xlabel="recorded step", ylabel=f"{key} [{unit}]", title=f"{chr(97 + row_index * 2 + col)}  {method}")
                if key == "speed":
                    ax.set_ylim(0, 6)
                elif key == "lane":
                    ax.set_yticks(range(4))
                for step, value in zip(trace["step"], trace[key]):
                    used.append(dict(method=method, phase="post-step", metric=key, units=unit, x=float(step), x_label="step", value=float(value)))
        _finish(fig, axes.ravel(), "Recorded PV traces · " + conditions, caption)
        artifacts.append(_save(fig, output, dict(base, metric="PV-trace", layout="speed-lane-distance"), used, manifest, formats, caption))
        fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), squeeze=False)
        used = []
        for row_index, (method, (speed, types, steps, _)) in enumerate(recorded.items()):
            for col, step in enumerate((1, 1000, 2000)):
                t = int(np.where(steps == step)[0][0])
                ax = axes[row_index, col]
                slab = speed[t].T
                image = ax.imshow(np.ma.masked_less(slab, 0), origin="lower", aspect="auto", interpolation="nearest", cmap=cmap,
                                  vmin=0, vmax=5, extent=(-10.5, 30.5, -.5, 3.5))
                yy, xx = np.where(slab == -2)
                ax.scatter(xx - 10, yy, marker="*", color="black", s=60)
                yy, xx = np.where(types[t].T == 1)
                ax.scatter(xx - 10, yy, marker="s", facecolors="none", edgecolors="black", s=20, linewidths=.7)
                ax.set(xlabel="offset [cells]", ylabel="physical lane", title=f"{chr(97 + row_index * 3 + col)}  {method} · step {step}")
                ax.set_yticks(range(4))
                ax.grid(False)
                for lane in range(4):
                    used.append(dict(method=method, phase="post-step", lane=lane, x=step, x_label="step", metric="recorded_snapshot",
                                     speed_cells=json.dumps(speed[t, :, lane].tolist()), type_cells=json.dumps(types[t, :, lane].tolist()), offsets="-10..30"))
        fig.suptitle("Recorded storyboard · steps 1 / 1000 / 2000 · " + conditions, fontsize=10)
        fig.text(.02, .01, textwrap.fill(caption, 155), fontsize=8)
        fig.tight_layout(rect=(0, .17, .94, .95))
        color_ax = fig.add_axes((.95, .25, .012, .45))
        fig.colorbar(image, cax=color_ax, label="background speed [cells/step]")
        artifacts.append(_save(fig, output, dict(base, metric="storyboard", layout="1-1000-2000"), used, manifest, formats, caption))
    _json(output / "artifacts.json", dict(artifacts=artifacts, source_manifest=manifest))
    return artifacts


def available_families(data_dir):
    """List supplied/deferred families before rendering; no runtime data access."""
    data = Path(data_dir)
    counts, defined_counts = defaultdict(int), defaultdict(int)
    panels, metrics = defaultdict(set), defaultdict(set)
    for row in _iter_csv(data / "points.csv"):
        family = row.get("family")
        counts[family] += 1
        defined_counts[family] += math.isfinite(_number(row.get("value")))
        if row.get("panel"):
            panels[family].add(row["panel"])
        if row.get("metric"):
            metrics[family].add(row["metric"])
    manifest = json.loads((data / "manifest.json").read_text())
    result = []
    for family, question in FAMILIES.items():
        defined = defined_counts[family]
        status = "available" if defined or family == "conceptual" else "deferred"
        reason = "" if status == "available" else "No defined supplied points; missing/NA rows remain in tables."
        if family == "illustrations":
            status = "pending_validation" if (data / "illustrations.json").exists() else "deferred"
            reason = "Requires final paired maps/traces and DATA frozen-selection evidence."
        elif family == "literature":
            reason = "Manual T4 literature sources required; never fabricated."
        elif family == "tradeoff":
            reason = "Requires unambiguous matched supplied PV and cost summaries."
        result.append(dict(family=family, question=question, status=status, rows=counts[family], defined=defined, reason=reason, supplied_panels=sorted(panels[family]), supplied_metrics=sorted(metrics[family]),
                           supported_scope="Supplied normalized summaries only; no reconstruction of missing diagnostics."))
    return dict(families=result, source_deferred=manifest.get("deferred_families", manifest.get("deferred", [])), source_manifest=manifest)


def _illustration_entries(data, output, formats, mode):
    path = data / "illustrations.json"
    if not path.exists():
        return [], [dict(family="illustrations", reason="No DATA-selected recorded examples supplied.")]
    payload = json.loads(path.read_text())
    entries = payload if isinstance(payload, list) else payload.get("entries", [])
    deferred = [] if isinstance(payload, list) else list(payload.get("deferred", []))
    artifacts = []
    for entry in entries:
        identity = entry.get("identity", {"cell": entry.get("cell", "unspecified")})
        if entry.get("mode") != "final" or not _complete_cell(entry.get("selection_evidence")):
            deferred.append(dict(family="illustrations", identity=identity, reason="Partial cell or absent frozen-selection evidence; no representative output."))
            continue
        pair = {}
        try:
            for method, record in entry["pair"].items():
                analysis = json.loads((data / record["analysis_path"]).read_text())
                with np.load(data / record["arrays_path"], allow_pickle=False) as arrays:
                    pair[method] = dict(analysis=analysis, arrays={k: arrays[k] for k in arrays.files}, episode_id=record.get("episode_id"))
            directory = _id(dict(identity, family="illustrations", metric="recorded"))
            rendered = render_recorded_illustration(pair, output / directory, identity=identity,
                                                    selection_evidence=entry["selection_evidence"], formats=formats,
                                                    source_manifest=json.loads((data / "manifest.json").read_text()))
            for artifact in rendered:
                artifact["files"] = [directory + "/" + f for f in artifact["files"]]
                for key in ("points", "metadata"):
                    artifact[key] = directory + "/" + artifact[key]
            artifacts.extend(rendered)
        except (ValueError, KeyError, OSError) as exc:
            deferred.append(dict(family="illustrations", identity=identity, reason=str(exc)))
    return artifacts, deferred


def _index(output, result):
    md = ["# Physica A atlas", "", f"Mode: **{result['mode']}**", "", "## Source snapshot", "", "```json",
          json.dumps(result["source_manifest"], ensure_ascii=False, indent=2), "```", "", "## Families", "",
          "| Family | State | Reason |", "| --- | --- | --- |"]
    body = ["<!doctype html><meta charset='utf-8'><title>Physica A atlas</title>",
            "<style>body{font:16px sans-serif;max-width:1250px;margin:2em auto;padding:1em}img{max-width:100%}td,th{padding:.4em;text-align:left;vertical-align:top}pre{white-space:pre-wrap}article{border-top:1px solid #ccc;padding:1em 0}</style>",
            "<h1>Physica A atlas</h1><p>Mode: <strong>" + html.escape(result["mode"]) + "</strong></p>",
            "<details><summary>Source snapshot and caveats</summary><pre>" + html.escape(json.dumps(result["source_manifest"], ensure_ascii=False, indent=2)) + "</pre></details>",
            "<h2>Available and deferred</h2><table><tr><th>Family</th><th>State</th><th>Reason</th></tr>"]
    filter_text = json.dumps(result["filters"], ensure_ascii=False)
    md += ["", "Render filters: " + filter_text, "", "[Complete native points](tables/points.csv)", ""]
    body.append("<p>Render filters: " + html.escape(filter_text) + '</p><p><a href="tables/points.csv">Complete native points</a></p>')
    for family in result["families"]:
        fields = [family["family"], family["status"], family["reason"]]
        md.append("| " + " | ".join(map(_md, fields)) + " |")
        body.append("<tr>" + "".join("<td>" + html.escape(x) + "</td>" for x in fields) + "</tr>")
    body.append("</table><h2>Independent scopes</h2><pre>" + html.escape(json.dumps(result["scopes"], indent=2)) + "</pre><h2>Figures</h2>")
    md += ["", "## Independent scopes", "", "```json", json.dumps(result["scopes"], indent=2), "```"]
    md += ["", "## Figures", ""]
    for artifact in result["artifacts"]:
        links = artifact["files"] + [artifact["points"], artifact["metadata"]]
        md += ["### " + artifact["id"], "", artifact["caption"], "", " · ".join(f"[{Path(p).suffix[1:]}]({p})" for p in links), ""]
        body.append("<article><h3>" + html.escape(artifact["id"]) + "</h3><p>" + html.escape(artifact["caption"]) + "</p>")
        body.append(" · ".join(f'<a href="{html.escape(p, quote=True)}">{html.escape(Path(p).suffix[1:])}</a>' for p in links))
        png = next((p for p in artifact["files"] if p.endswith(".png")), None)
        if png:
            body.append(f'<p><img loading="lazy" src="{html.escape(png, quote=True)}" alt="{html.escape(artifact["caption"], quote=True)}"></p>')
        body.append("</article>")
    md += ["## Tables", ""]
    body.append("<h2>Tables</h2><ul>")
    for table in result["tables"]:
        md.append("- " + table["title"] + ": " + " · ".join(f"[{Path(p).suffix[1:]}](tables/{p})" for p in table["files"]))
        body.append("<li>" + html.escape(table["title"]) + ": " + " · ".join(f'<a href="tables/{html.escape(p, quote=True)}">{Path(p).suffix[1:]}</a>' for p in table["files"]) + "</li>")
    body.append("</ul><h2>Deferred detail</h2><pre>" + html.escape(json.dumps(result["deferred"], ensure_ascii=False, indent=2)) + "</pre>")
    md += ["", "## Deferred detail", "", "```json", json.dumps(result["deferred"], ensure_ascii=False, indent=2), "```"]
    (output / "index.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output / "index.html").write_text("\n".join(body) + "\n", encoding="utf-8")


def render_atlas(data_dir, output_dir, *, formats=("png", "pdf"), families=None,
                 page_size=6, heatmaps=True, conceptual=True, metrics=None, table_row_limit=120,
                 all_local_scopes=False, panels=None):
    """Render the normalized snapshot to a NEW directory and return its manifest.

    families/metrics/panels filter numerical figures and article previews, never native CSVs.
    panels uses normalized names (main/E4/W7/E5_L1000/E5_L2000/validation).
    Default local windows/transitions use declared axis marginals; all_local_scopes=True
    restores every supplied scope. Explicit metrics override window outcome defaults.
    Tradeoffs filter cost metrics and retain the PV counterpart. Filters are reported.
    Default PNG+PDF; opt into SVG via formats. No overwrite or resume: use a new
    directory for each snapshot/render. page_size is 1..6 readable facets/page.
    No simulator, checkpoint loader, inference, bootstrap or live dataset access.
    """
    formats = _formats(formats)
    if not isinstance(page_size, int) or not 1 <= page_size <= 6:
        raise ValueError("page_size must be between 1 and 6")
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("mode") not in ("final", "complete", "completed_validation", "partial_exploratory"):
        raise ValueError("manifest requires schema_version=1 and explicit final/complete/completed_validation/partial_exploratory mode")
    chosen = set(families) if families is not None else set(FAMILIES)
    unknown = chosen - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    metric_filter = set(metrics) if metrics is not None else None
    panel_filter = set(panels) if panels is not None else None
    if table_row_limit is not None and (not isinstance(table_row_limit, int) or table_row_limit < 1):
        raise ValueError("table_row_limit must be positive or None")
    selected, tradeoff = [], []
    supplied_metrics, supplied_families = set(), set()
    panel_counts, panel_defined = defaultdict(int), defaultdict(int)
    for row in _iter_csv(data / "points.csv"):
        if set(POINT_COLUMNS) - set(row):
            raise ValueError("points.csv does not satisfy PLAN's normalized column contract")
        low, high = (_number(row.get(k)) for k in ("ci_low", "ci_high"))
        if math.isfinite(low) != math.isfinite(high) or (math.isfinite(low) and not row.get("interval_kind")):
            raise ValueError("supplied intervals require both endpoints and interval_kind")
        if math.isfinite(low) and low > high:
            raise ValueError("interval endpoints reversed")
        if manifest["mode"] == "partial_exploratory" and math.isfinite(low):
            raise ValueError("partial_exploratory cannot carry final inferential intervals")
        supplied_metrics.add(row["metric"])
        supplied_families.add(row["family"])
        panel_counts[row["panel"]] += 1
        panel_defined[row["panel"]] += math.isfinite(_number(row.get("value")))
        if panel_filter is not None and row["panel"] not in panel_filter:
            continue
        if not _local_scope(row, all_local_scopes) or _categorical_mean(row):
            continue
        if row["family"] in chosen and (metric_filter is None or row["metric"] in metric_filter):
            if row["family"] != "mechanism" or _outcome(row, metric_filter):
                selected.append(row)
        if "tradeoff" in chosen and _tradeoff_row(row, metric_filter):
            tradeoff.append(row)
    if metric_filter is not None and metric_filter - supplied_metrics:
        raise ValueError(f"unknown metrics: {sorted(metric_filter - supplied_metrics)}")
    if panel_filter is not None and panel_filter - set(panel_counts):
        raise ValueError(f"unknown panels: {sorted(panel_filter - set(panel_counts))}")
    filters = dict(panels=sorted(panel_filter) if panel_filter is not None else None, families=sorted(chosen), metrics=sorted(metric_filter) if metric_filter is not None else None,
                   table_row_limit=table_row_limit, all_local_scopes=all_local_scopes,
                   local_scope_policy="full Cartesian" if all_local_scopes else "axis marginals",
                   local_scope_families=["windows", "transitions"],
                   local_scope_union=None if all_local_scopes else LOCAL_MARGINALS,
                   unscoped_local_rows="retained",
                   default_window_outcomes=sorted(WINDOW_OUTCOMES) if metric_filter is None else None,
                   window_support="dashboards and complete CSV; explicit metrics enable standalone charts",
                   default_mechanism_outcomes=["*/frequency", "noncategorical */mean"] if metric_filter is None else None,
                   mechanism_audit="complete CSV; explicit metrics enable protocol/support/min/max/count charts",
                   excluded_categorical_means=["*_type/mean", "*_reason/mean", "*_direction_set/mean"])
    manifest = dict(manifest, render_filters=filters)
    output = _new_dir(output_dir)
    selected = _reference_views(selected)
    tradeoff = _reference_views(tradeoff)
    outcomes = [r for r in selected if _outcome(r, metric_filter)]
    plt = _plt()
    with plt.rc_context(RC):
        artifacts = _curve_pages(outcomes, output, manifest, formats, page_size)
        artifacts.extend(_dashboard_pages(selected, output, manifest, formats, page_size))
        if heatmaps:
            artifacts.extend(_heatmap_pages(outcomes, output, manifest, formats, page_size))
        if "tradeoff" in chosen:
            artifacts.extend(_tradeoff_pages(tradeoff, output, manifest, formats, page_size, metrics=metric_filter))
    if conceptual:
        artifact = _conceptual(output / "conceptual", formats, manifest)
        artifact["files"] = ["conceptual/" + p for p in artifact["files"]]
        for key in ("points", "metadata"):
            artifact[key] = "conceptual/" + artifact[key]
        artifacts.append(artifact)
    illustration_artifacts, deferred = _illustration_entries(data, output, formats, manifest["mode"]) if "illustrations" in chosen and (panel_filter is None or "E4" in panel_filter) else ([], [])
    artifacts.extend(illustration_artifacts)
    tables = export_tables(data, output / "tables", families=chosen, metrics=metric_filter, table_row_limit=table_row_limit, panels=panel_filter)
    inventory = available_families(data)
    counts = defaultdict(int)
    for artifact in artifacts:
        counts[artifact["family"]] += 1
    for family in inventory["families"]:
        name = family["family"]
        if counts[name]:
            family.update(status="rendered", artifacts=counts[name], reason="Full plotted points and metadata accompany every figure.")
        elif name not in chosen or (name == "conceptual" and not conceptual):
            family.update(status="deferred", reason="Not requested in this render; supplied rows retained in tables.")
        else:
            family["status"] = "deferred"
    for key, group in _groups(selected, BASE + FACETS):
        if not any(math.isfinite(_number(r.get("value"))) for r in group):
            deferred.append(dict(family=group[0]["family"], identity=dict(zip(BASE + FACETS, key)),
                                 reason="No defined values in slice", statuses=_unique(group, "status")))
    deferred.append(dict(family="literature", reason="Manual T4 literature comparison requires sources."))
    for name in sorted(supplied_families - set(FAMILIES)):
        deferred.append(dict(family=name, reason="Unknown family retained in complete tables; no rendering contract."))
    result = dict(schema_version=1, mode=manifest["mode"], source_manifest=manifest,
                  families=inventory["families"], artifacts=artifacts, tables=tables,
                  filters=filters,
                  deferred=deferred, source_deferred=inventory["source_deferred"],
                  scopes=[dict(panel=panel, status="available" if panel_defined[panel] else "deferred",
                               rows=panel_counts[panel])
                          for panel in ("main", "E4", "W7", "E5_L1000", "E5_L2000", "validation")])
    _json(output / "artifacts.json", result)
    _index(output, result)
    return result


def _dashboard_pages(rows, output, manifest, formats, page_size):
    """Place conditional outcomes and recorded availability/support side by side."""
    artifacts = []
    keys = ("family", "panel", "cohort", "rule", "road_length") + FACETS
    selected = [_checkpoint_display_row(r) for r in rows
                if r["family"] in {"laps", "control", "windows", "events", "mechanism", "selection"}]
    dashboard_groups = []
    for _, scope in _groups(selected, ("x_label",)):
        scope_keys = keys + ("x_label",) + _density_keys(scope) + _candidate_keys(scope)
        if any(r.get("training_line") not in ("", "all") for r in scope):
            scope_keys += ("method", "training_line")
        dashboard_groups.extend((dict(zip(scope_keys, key)), group) for key, group in _groups(scope, scope_keys))
    for identity, group in dashboard_groups:
        panels = _groups(group, ("metric", "units", "estimand", "phase", "x_label"))
        if len(panels) < 2:
            continue
        # A dashboard is an extra view, not a replacement for the complete facets.
        # Keep line identities separate even when metrics share the same page.
        for start in range(0, len(panels), page_size):
            page = panels[start:start + page_size]
            used = [r for _, rr in page for r in rr]
            if not any(math.isfinite(_number(r.get("value"))) for r in used):
                continue
            fig, axes = _axes(len(page))
            for i, (ax, (metric, rr)) in enumerate(zip(axes, page)):
                _line_panel(ax, rr, identity["family"], chr(97 + i))
                ax.set_title(textwrap.fill(_metric_label(metric[0]) + (" · " + metric[3] if metric[3] else ""), 38))
            caption = _caption(used, manifest, identity["family"])
            title = " · ".join(f"{k}={v}" for k, v in identity.items() if v)
            _finish(fig, axes, textwrap.fill(title, 150), caption)
            artifacts.append(_save(fig, output, dict(identity, metric="outcomes-support", layout="dashboard",
                                                    page=start // page_size + 1, metrics=[list(k) for k, _ in page]), used, manifest, formats, caption))
    return artifacts
