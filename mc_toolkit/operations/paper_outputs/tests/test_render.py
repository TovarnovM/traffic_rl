"""Hand-built normalized points only; no native runtime or inference required."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import render
from style import canonical_method, method_style


def point(**changes):
    row = dict.fromkeys(render.POINT_COLUMNS, "")
    row.update(family="mobility", panel="main", cohort="E3", rule="S0", method="Graph",
               training_line="all", road_length="1000", density="0.1", fraction="0.9",
               metric="mean_speed_priority", x="0.1", x_label="lattice density", value="2",
               n="5", n_planned="30", k="3", k_planned="3", support="5", support_kind="episodes",
               status="defined", units="cells/step", estimand="whole_family")
    row.update(changes)
    return row


def dataset(tmp_path, rows, *, mode="partial_exploratory", selections=None):
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps(dict(schema_version=1, mode=mode,
        snapshot={"id": "snapshot-123", "cutoff": "2026-11-09T17:44:37Z"},
        deferred_families=["recorded maps unavailable"], min_repeats=5)))
    with (data / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(dict.fromkeys(render.POINT_COLUMNS + [k for r in rows for k in r])))
        writer.writeheader()
        writer.writerows(rows)
    if selections:
        with (data / "selections.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(selections[0]))
            writer.writeheader()
            writer.writerows(selections)
    return data


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_latex_contrasts_use_portable_math_glyphs():
    assert render._latex('Graph−Node × 0.5 ≤ 1') == r'Graph\ensuremath{-}Node \ensuremath{\times} 0.5 \ensuremath{\leq} 1'


def test_summary_conditions_share_scale_and_contrasts_have_separate_panels(monkeypatch, tmp_path):
    captured = []

    def capture(fig, output, identity, rows, manifest, formats, caption):
        axes = [ax for ax in fig.axes if ax.axison and ax.get_visible()]
        captured.append((identity, [ax.get_ylim() for ax in axes],
                         [[line.get_label() for line in ax.lines
                           if line.get_label().startswith('Graph−')] for ax in axes]))
        render._plt().close(fig)
        return identity

    monkeypatch.setattr(render, '_save', capture)
    manifest = dict(mode='partial_exploratory')
    render._curve_pages([point(value='2'), point(fraction='0.95', value='0')],
                        tmp_path, manifest, ('png',), 6)
    assert len(captured[0][1]) == 2
    assert captured[0][1][0] == captured[0][1][1]
    captured.clear()
    methods = ['Graph−H2', 'Graph−Node', 'Graph−Grid']
    render._curve_pages([point(family='contrasts', metric='paired_PV_speed_difference',
                              method=method, value=str(i - 1)) for i, method in enumerate(methods)],
                        tmp_path, manifest, ('png',), 6)
    labels = [labels for _, _, panels in captured for labels in panels]
    assert len(labels) == 3
    assert sorted(labels) == sorted([[method] for method in methods])


def test_tradeoff_near_unity_ticks_remain_readable(monkeypatch, tmp_path):
    captured = []
    def capture(fig, output, identity, rows, manifest, formats, caption):
        axes = [ax for ax in fig.axes if ax.axison and ax.get_visible()]
        captured.extend((len(ax.get_xticks()), ax.get_ylabel()) for ax in axes)
        render._plt().close(fig)
        return identity
    monkeypatch.setattr(render, '_save', capture)
    rows = []
    for density, speed, ratio in [('0.1', '2', '1.0004'), ('0.2', '3', '1.0018')]:
        rows.extend([point(density=density, x=density, value=speed),
                     point(density=density, x=density, family='ratio', metric='R',
                           units='ratio', value=ratio)])
    render._tradeoff_pages(rows, tmp_path, dict(mode='partial_exploratory'), ('png',), 6)
    assert captured and all(count <= 6 and '_' not in label for count, label in captured)


def test_fixed_controls_have_no_invented_training_roster():
    assert render._support([point(method='B1', k='', k_planned='')]) == 'n=5/30'
    assert 'k=1/NA' in render._support([point(k='1', k_planned='')])


def test_native_table_snapshot_reuses_bytes_and_survives_atomic_refresh(tmp_path):
    data = dataset(tmp_path, [point()])
    original = (data / 'points.csv').read_bytes()
    out = tmp_path / 'tables'
    tables = render.export_tables(data, out)
    assert (out / 'points.csv').stat().st_ino == (data / 'points.csv').stat().st_ino
    assert next(t for t in tables if t['id'] == 'points')['storage'] == 'hardlink'
    replacement = data / 'replacement.csv'
    replacement.write_text('new snapshot\n')
    assert (data / 'points.csv').resolve().parent == data.resolve()
    replacement.replace(data / 'points.csv')
    assert (out / 'points.csv').read_bytes() == original


def test_atlas_preview_rows_sidecars_indexes_tables_and_no_overwrite(tmp_path):
    rows = [point(sd=".2"), point(density="0.2", x="0.2", value="", status="empty_exposure"),
            point(density="0.3", x="0.3", value="0")]
    data = dataset(tmp_path, rows, selections=[dict(model_id="Graph_a&b", rule="C95", status="infeasible", checkpoint="", k="2/3")])
    out = tmp_path / "atlas"
    result = render.render_atlas(data, out, conceptual=False)
    assert result["mode"] == "partial_exploratory"
    assert len(result["artifacts"]) == 1
    artifact = result["artifacts"][0]
    assert {Path(f).suffix for f in artifact["files"]} == {".png", ".pdf"}
    assert all((out / f).stat().st_size > 0 for f in artifact["files"])
    assert read_csv(out / artifact["points"]) == rows
    meta = json.loads((out / artifact["metadata"]).read_text())
    assert meta["plotted_rows"] == 3
    assert meta["source_manifest"]["snapshot"]["id"] == "snapshot-123"
    assert "empty_exposure" in meta["statuses"]
    for text in (meta["caption"], (out / "index.html").read_text(), (out / "index.md").read_text()):
        assert "partial_exploratory" in text and "n=5/30" in text and "k=3/3" in text
    assert "SD is descriptive" in meta["caption"]
    assert "preview" in meta["caption"]
    tex = "\n".join(p.read_text() for p in (out / "tables").glob("*.tex"))
    assert "infeasible" in tex and r"Graph\_a\&b" in tex
    assert "empty\\_exposure" in tex
    with pytest.raises(FileExistsError):
        render.render_atlas(data, out, conceptual=False)
    assert read_csv(out / artifact["points"]) == rows


def test_missing_is_gap_zero_is_point_and_labels_are_distinct():
    rows = [point(), point(x=".2", value="", status="undefined_denominator"), point(x=".3", value="0")]
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, rows, "mobility", "a")
    y = ax.lines[0].get_ydata()
    assert y[0] == 2 and np.isnan(y[1]) and y[2] == 0
    assert [text.get_text() for text in ax.texts].count("NA") == 1
    assert "undefined_denominator" in rows[1]["status"]
    aligned, _ = render._series_arrays([rows[0], rows[2]], ["0.1", ".2", ".3"])
    assert np.isnan(aligned[1])
    plt.close(fig)


def test_palette_candidates_and_native_aliases_are_not_conflated():
    assert canonical_method("left_if_safe") == "original"
    assert canonical_method("H1") == "H1"
    assert canonical_method("graph_ppo") == "Graph"
    assert canonical_method("B3") == "B3"
    assert method_style("Graph")["color"] == "#3850A0"
    assert method_style("Node")["linewidth"] == method_style("Grid")["linewidth"]
    assert method_style("original")["markerfacecolor"] == "none"
    assert method_style("H2")["markerfacecolor"] == "#80551E"


def test_supplied_intervals_and_ratio_references_only(tmp_path):
    data = dataset(tmp_path, [point(family="ratio", metric="R", value=".97", ci_low=".94", ci_high="1.01", interval_kind="crossed 95% CI")], mode="final")
    result = render.render_atlas(data, tmp_path / "final", conceptual=False, formats=("svg",))
    assert "crossed 95% CI" in result["artifacts"][0]["caption"]
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, read_csv(data / "points.csv"), "ratio", "a")
    assert any(np.all(np.asarray(line.get_ydata()) == 1) for line in ax.lines)
    assert any(np.all(np.asarray(line.get_ydata()) == .95) for line in ax.lines)
    plt.close(fig)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["mode"] = "partial_exploratory"
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="partial_exploratory"):
        render.render_atlas(data, tmp_path / "bad", conceptual=False)
    assert not (tmp_path / "bad").exists()


def test_heatmaps_and_cohort_length_training_identity(tmp_path):
    rows = [point(panel="W7", cohort=cohort, road_length=length, method=method, training_line="seed1",
                  density=density, x=density, fraction=fraction)
            for cohort, length in (("scale0.5", "1000"), ("scale2", "2000"))
            for method in ("Graph", "Node") for density in ("0.1", "0.2") for fraction in ("0.9", "1.0")]
    data = dataset(tmp_path, rows)
    result = render.render_atlas(data, tmp_path / "out", conceptual=False, formats=("png",))
    ids = [a["id"] for a in result["artifacts"]]
    assert len(ids) == len(set(ids))
    assert any("heatmap" in identifier for identifier in ids)
    for artifact in result["artifacts"]:
        meta = json.loads((tmp_path / "out" / artifact["metadata"]).read_text())
        plotted = read_csv(tmp_path / "out" / artifact["points"])
        assert len({r["road_length"] for r in plotted}) == 1
        assert len({r["cohort"] for r in plotted}) == 1
        if meta["identity"]["layout"] == "complete":
            assert "method" in render.FACETS + ("method",)
            assert all(len(facet) == len(render.FACETS) + 1 for facet in meta["identity"]["facets"])


def test_all_na_is_deferred_but_complete_table_remains(tmp_path):
    data = dataset(tmp_path, [point(family="windows", window="fixed_front_2", value="", status="empty_exposure")])
    inventory = render.available_families(data)
    assert next(f for f in inventory["families"] if f["family"] == "windows")["status"] == "deferred"
    result = render.render_atlas(data, tmp_path / "out", conceptual=False, formats=("png",))
    assert not result["artifacts"]
    assert result["tables"]
    assert any(f["family"] == "literature" and f["status"] == "deferred" for f in result["families"])


def test_lap_dashboard_contains_availability_and_support(tmp_path):
    rows = [point(family="laps", metric="conditional_lap_time", units="steps", value="230"),
            point(family="laps", metric="lap_availability_fraction", units="fraction of runs", value=".5"),
            point(family="laps", metric="completed_laps", units="laps", value="2")]
    data = dataset(tmp_path, rows)
    result = render.render_atlas(data, tmp_path / "out", conceptual=False, formats=("png",))
    dashboard = next(a for a in result["artifacts"] if "dashboard" in a["id"])
    assert {r["metric"] for r in read_csv(tmp_path / "out" / dashboard["points"])} == {r["metric"] for r in rows}
    assert "n/N" in dashboard["caption"]


def recorded_pair():
    speed = np.full((3, 41, 4), -1, dtype=np.int8)
    types = np.full_like(speed, -1)
    speed[:, 10, 1] = -2
    types[:, 10, 1] = -2
    speed[:, 15, 2] = 0
    types[:, 15, 2] = 0
    speed[:, 20, 3] = 5
    types[:, 20, 3] = 1
    trace = dict(step=[1, 1000, 2000], speed=[0, 3, 6], lane=[1, 1, 2], cumulative_distance=[0, 3000, 6000])
    return {method: dict(episode_id=method + "-road1", trace=trace, local_speed=speed.copy(), local_type=types.copy(),
                         map_steps=np.array([1, 1000, 2000])) for method in ("Graph", "H2")}


def test_recorded_sentinels_storyboard_and_partial_refusal(tmp_path):
    pair = recorded_pair()
    with pytest.raises(ValueError, match="final cell"):
        render.render_recorded_illustration(pair, tmp_path / "partial", identity={}, selection_evidence={"rule": "frozen"}, mode="partial_exploratory")
    assert not (tmp_path / "partial").exists()
    out = tmp_path / "recorded"
    artifacts = render.render_recorded_illustration(pair, out, identity={"cell": "rho20-f90", "road_length": "1000"},
                  selection_evidence={"rule": "frozen_median_line_then_environment", "status": "selected", "n": 90, "n_planned": 90, "k": 3, "k_planned": 3}, formats=("png",))
    assert len(artifacts) == 3
    storyboard = next(a for a in artifacts if "storyboard" in a["id"])
    rows = read_csv(out / storyboard["points"])
    assert {int(r["x"]) for r in rows} == {1, 1000, 2000}
    assert {int(r["lane"]) for r in rows} == {0, 1, 2, 3}
    assert any(-2 in json.loads(r["speed_cells"]) for r in rows)
    assert any(-1 in json.loads(r["speed_cells"]) for r in rows)
    assert any(0 in json.loads(r["speed_cells"]) for r in rows)
    assert "n=90/90" in storyboard["caption"]
    pair["Graph"]["local_speed"][0, 0, 0] = 6
    with pytest.raises(ValueError, match="background speed"):
        render.render_recorded_illustration(pair, tmp_path / "invalid", identity={}, selection_evidence={"rule": "frozen_median_line_then_environment", "status": "selected", "n": 90})


def test_partial_illustration_index_explicitly_deferred(tmp_path):
    data = dataset(tmp_path, [])
    (data / "illustrations.json").write_text(json.dumps([dict(identity={"cell": "rho20"}, mode="partial_exploratory", pair={})]))
    result = render.render_atlas(data, tmp_path / "out", conceptual=False, formats=("png",))
    assert any("Partial cell" in d.get("reason", "") for d in result["deferred"] if isinstance(d, dict))
    assert not result["artifacts"]


def test_conceptual_export_is_labelled_and_has_phase_warning(tmp_path):
    artifact = render.render_conceptual(tmp_path / "concept", formats=("svg",))
    assert "Conceptual only" in artifact["caption"]
    assert "pre-step geometry" in artifact["caption"]
    text = (tmp_path / "concept" / artifact["files"][0]).read_text()
    assert "sensor_distance=60" in text
    assert "physical lane" in text


def test_tradeoff_pairs_supplied_values_without_reducing(tmp_path):
    rows = [point(), point(family="ratio", metric="R", units="ratio", value=".98"),
            point(family="windows", metric="primary_normalized_local_delay", units="fraction", value=".1", window="fixed_front_2")]
    data = dataset(tmp_path, rows)
    result = render.render_atlas(data, tmp_path / "out", families=["tradeoff"], conceptual=False, formats=("png",))
    assert len(result["artifacts"]) == 2
    for artifact in result["artifacts"]:
        plotted = read_csv(tmp_path / "out" / artifact["points"])
        assert len(plotted) == 2
        assert plotted[0] == rows[0]
        assert plotted[1] in rows[1:]


def test_native_scope_family_rows_preserve_phase_support_censoring(tmp_path):
    rows = [point(family=family, metric=metric, x="0", x_label="lag" if family in {"spatial", "temporal"} else "recorded category",
                  density="0.2", phase="pre-step", support="72", support_kind="recorded pairs", status="right_censored")
            for family, metric in (("state", "N_S_Z_A"), ("mechanism", "native_eligible/frequency"),
                                   ("spatial", "connected_covariance"), ("temporal", "normalized_covariance"),
                                   ("events", "pv_stop_right_censored"), ("profiles", "accumulated_offset"),
                                   ("transitions", "N_transition"), ("onset", "stop_onset_speed"))]
    data = dataset(tmp_path, rows)
    result = render.render_atlas(data, tmp_path / "out", conceptual=False, formats=("png",))
    assert {a["family"] for a in result["artifacts"]} == {r["family"] for r in rows}
    for artifact in result["artifacts"]:
        meta = json.loads((tmp_path / "out" / artifact["metadata"]).read_text())
        assert meta["phases"] == ["pre-step"]
        assert meta["support_kinds"] == ["recorded pairs"]
        assert meta["statuses"] == ["right_censored"]
        assert read_csv(tmp_path / "out" / artifact["points"])[0] in rows


@pytest.mark.parametrize("contrast,left", [("Graph−Node", "Graph"), ("Graph−Grid", "Graph"),
                                           ("Node−H2", "Node"), ("Grid−H2", "Grid")])
def test_directional_contrasts_keep_label_and_left_style(contrast, left):
    assert canonical_method(contrast) == contrast
    assert method_style(contrast) == method_style(left)


def test_availability_is_successes_over_observed_and_training_marks_are_discrete():
    plt = render._plt()
    fig, ax = plt.subplots()
    row = point(family="laps", metric="lap_availability", value="0", support="0",
                support_kind="episodes_with_completed_lap")
    render._line_panel(ax, [row], "laps", "a")
    assert "0/5" in [t.get_text() for t in ax.texts]
    assert "5/30" not in [t.get_text() for t in ax.texts]
    assert "n=5/30" in render._caption([row], {"mode": "partial_exploratory"}, "laps")
    marks = next(line for line in ax.lines if line.get_label() == "training densities")
    assert marks.get_marker() == "|" and marks.get_linestyle() == "None"
    plt.close(fig)
    fig, ax = plt.subplots()
    render._line_panel(ax, [dict(row, support_kind="episodes")], "laps", "a")
    assert not any("/" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_candidate_identity_and_complete_cell_in_partial_atlas(tmp_path):
    rows = []
    for candidate in ("checkpoint1", "checkpoint2"):
        for density in ("0.1", "0.2"):
            for fraction in ("0.9", "1.0"):
                common = dict(candidate_id=candidate, model_id=candidate, training_line="line1",
                              density=density, x=density, fraction=fraction)
                rows += [point(**common), point(**common, family="ratio", metric="R", value=".98"),
                         point(**common, family="laps", metric="conditional_lap_time", value="", status="no_completed_laps"),
                         point(**common, family="laps", metric="lap_availability", value="0", support="0",
                               support_kind="episodes_with_completed_lap")]
    data = dataset(tmp_path, rows)
    records = {}
    for method, record in recorded_pair().items():
        (data / (method + ".json")).write_text(json.dumps({"pv": {"trace": record["trace"]}}))
        np.savez(data / (method + ".npz"), **{k: record[k] for k in ("local_speed", "local_type", "map_steps")})
        records[method] = dict(analysis_path=method + ".json", arrays_path=method + ".npz", episode_id=record["episode_id"])
    evidence = dict(status="selected", n=90, rule="frozen_median_line_then_environment",
                    line_means={"line1": 0, "line2": 1, "line3": 2})
    (data / "illustrations.json").write_text(json.dumps({"entries": [dict(identity={"cell": "complete-E4"},
        mode="final", selection_evidence=evidence, pair=records)]}))
    out = tmp_path / "out"
    result = render.render_atlas(data, out, conceptual=False, formats=("svg",), table_row_limit=2)
    assert sum(a["family"] == "illustrations" for a in result["artifacts"]) == 3
    assert (out / "tables" / "points.csv").read_bytes() == (data / "points.csv").read_bytes()
    for layout in ("complete", "heatmap", "scatter", "dashboard"):
        artifacts = [a for a in result["artifacts"] if layout in a["id"]]
        assert artifacts
        seen = set()
        for artifact in artifacts:
            plotted = read_csv(out / artifact["points"])
            seen.update(r["candidate_id"] for r in plotted)
            if layout == "scatter":
                assert all(plotted[i]["candidate_id"] == plotted[i+1]["candidate_id"] for i in range(0, len(plotted), 2))
        assert seen == {"checkpoint1", "checkpoint2"}


def test_checkpoint_steps_evolve_on_same_curve_and_filters_are_reported(tmp_path):
    rows = [point(family="selection", metric="V", training_line="line1", candidate_id=f"c{i}",
                  model_id=f"m{i}", x=str(i * 1000), x_label="checkpoint environment steps", density="")
            for i in (1, 2)]
    rows.append(point(metric="other"))
    data = dataset(tmp_path, rows, mode="completed_validation")
    out = tmp_path / "out"
    result = render.render_atlas(data, out, families=["selection"], metrics=["V"], conceptual=False, formats=("svg",))
    assert len(result["artifacts"]) == 1
    assert len(read_csv(out / result["artifacts"][0]["points"])) == 2
    assert result["filters"]["metrics"] == ["V"]
    assert "Render filters" in (out / "index.html").read_text()
    assert len(read_csv(out / "tables" / "points.csv")) == 3


def test_tradeoff_ambiguous_rows_fail_instead_of_disappearing(tmp_path):
    rows = [point(), point(), point(family="ratio", metric="R")]
    data = dataset(tmp_path, rows)
    with pytest.raises(ValueError, match="ambiguous mobility match"):
        render.render_atlas(data, tmp_path / "out", families=["tradeoff"], conceptual=False, formats=("svg",))


@pytest.mark.parametrize("evidence", [dict(status="selected", n=89, rule="frozen"),
                                       dict(status="deferred", n=90, rule="frozen"),
                                       dict(status="selected", n=90)])
def test_partial_cell_evidence_is_rejected(tmp_path, evidence):
    with pytest.raises(ValueError, match="final cell"):
        render.render_recorded_illustration(recorded_pair(), tmp_path / "out", identity={}, selection_evidence=evidence)
    assert not (tmp_path / "out").exists()


def test_supplied_interval_need_not_contain_point():
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, [point(value="2", ci_low="3", ci_high="4", interval_kind="supplied")], "mobility", "a")
    segment = ax.collections[0].get_segments()[0]
    assert segment[:, 1].tolist() == [3, 4]
    assert ax.lines[0].get_ydata().tolist() == [2]
    plt.close(fig)


def test_many_training_lines_split_without_loss(tmp_path):
    rows = [point(training_line=f"line{i}") for i in range(5)]
    data = dataset(tmp_path, rows)
    out = tmp_path / "out"
    result = render.render_atlas(data, out, conceptual=False, formats=("svg",))
    artifact = result["artifacts"][0]
    metadata = json.loads((out / artifact["metadata"]).read_text())
    assert len(metadata["identity"]["facets"]) == 5
    assert len(read_csv(out / artifact["points"])) == 5


def test_category_codes_are_discrete_and_no_training_marks():
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, [point(family="mechanism", x="0", x_label="native category")], "mechanism", "a")
    assert ax.lines[0].get_linestyle() == "None"
    assert not any(line.get_label() == "training densities" for line in ax.lines)
    plt.close(fig)


@pytest.mark.parametrize("family", ["windows", "transitions"])
def test_local_marginals_preserve_every_axis_without_cartesian_crossings(family):
    windows = [f"{geometry}_{i}" for geometry in ("fixed_front", "fixed_sym", "speed_front", "speed_sym") for i in (1, 2, 3)]
    rows = [point(family=family, window=window, region=region, lane=lane, vehicle_group=group)
            for window in windows for region in ("rear", "centre", "front", "full")
            for lane in ("0", "1", "2", "3", "all") for group in ("AV", "HDV", "background")]
    selected = [r for r in rows if render._local_scope(r, False)]
    assert len(rows) == 720 and len(selected) == 35
    for key in ("window", "region", "lane", "vehicle_group"):
        assert {r[key] for r in selected} == {r[key] for r in rows}
    assert all(r["window"] == "fixed_front_2" or
               (r["region"], r["lane"], r["vehicle_group"]) == ("full", "all", "background") for r in selected)
    assert not render._local_scope(point(family=family, window="fixed_front_2", region="rear", lane="0", vehicle_group="AV"), False)
    assert all(render._local_scope(r, True) for r in rows)
    assert all(render._local_scope(dict(r, value="", status="empty_exposure"), False) ==
               render._local_scope(r, False) for r in rows)
    assert all(render._local_scope(dict(r, family="state"), False) for r in rows)


def test_atlas_scope_policy_and_explicit_support_override(tmp_path, monkeypatch):
    common = dict(family="windows", window="fixed_front_2", region="full", lane="all", vehicle_group="background")
    rows = [point(**common, metric="pooled_speed"), point(**common, metric="N"),
            point(**dict(common, window="speed_sym_3", region="rear", lane="2"), metric="pooled_speed"),
            point(family="mechanism", metric="pre_pv_leader_type/mean"),
            point(family="mechanism", metric="pre_pv_leader_type/frequency", x_label="native category")]
    data = dataset(tmp_path, rows)
    captured = {}
    def capture(name):
        def collect(points, *args, **kwargs):
            captured[name] = list(points)
            return []
        return collect
    for name in ("curve", "dashboard", "heatmap"):
        monkeypatch.setattr(render, f"_{name}_pages", capture(name))
    out = tmp_path / "default"
    result = render.render_atlas(data, out, families=["windows", "mechanism"], conceptual=False)
    assert captured["curve"] == [rows[0], rows[4]]
    assert captured["dashboard"] == [rows[0], rows[1], rows[4]]
    assert result["filters"]["local_scope_union"] == render.LOCAL_MARGINALS
    assert result["filters"]["default_window_outcomes"] == sorted(render.WINDOW_OUTCOMES)
    assert "axis marginals" in (out / "index.md").read_text()
    assert (out / "tables/points.csv").read_bytes() == (data / "points.csv").read_bytes()
    result = render.render_atlas(data, tmp_path / "all", families=["windows"], conceptual=False,
                                 all_local_scopes=True, metrics=["pooled_speed", "N"])
    assert captured["curve"] == rows[:3]
    assert result["filters"]["all_local_scopes"] is True
    assert result["filters"]["default_window_outcomes"] is None


@pytest.mark.parametrize("owner", ["atlas", "inventory", "tables"])
def test_streaming_consumers_do_not_retain_rejected_rows(tmp_path, monkeypatch, owner):
    import weakref
    class TrackedRow(dict):
        pass
    data = dataset(tmp_path, [])
    refs = []
    def tracked(path):
        if path.name != "points.csv":
            return
        for i in range(60):
            # One selected row, then family-, metric-, and local-scope rejections.
            row = TrackedRow(point(family="windows", metric="pooled_speed", window="fixed_front_2",
                                   region="full", lane="all", vehicle_group="background"))
            if i:
                row.update(({"family": "mobility"}, {"metric": "N"},
                            {"window": "speed_sym_3", "region": "rear", "lane": "0"})[(i - 1) % 3])
            refs.append(weakref.ref(row))
            yield row
            live = sum(ref() is not None for ref in refs)
            # Table previews retain <=2 rows in each of two family scopes.
            assert live <= (6 if owner == "tables" else 3)
    monkeypatch.setattr(render, "_iter_csv", tracked)
    if owner == "inventory":
        inventory = render.available_families(data)
        assert sum(f["rows"] for f in inventory["families"]) == 60
    elif owner == "tables":
        tables = render.export_tables(data, tmp_path / "tables", table_row_limit=2)
        previews = [t for t in tables if "total_rows" in t]
        assert sum(t["total_rows"] for t in previews) == 60
        assert all(t["rows"] == 2 and t["truncated"] for t in previews)
    else:
        captured = []
        def curves(rows, *args):
            captured.extend(dict(r) for r in rows)
            return []
        monkeypatch.setattr(render, "_curve_pages", curves)
        monkeypatch.setattr(render, "_dashboard_pages", lambda *a: [])
        monkeypatch.setattr(render, "export_tables", lambda *a, **k: [])
        monkeypatch.setattr(render, "available_families", lambda *a: dict(families=[], source_deferred=[]))
        render.render_atlas(data, tmp_path / "atlas", families=["windows"], metrics=["pooled_speed"],
                            conceptual=False, heatmaps=False)
        assert len(captured) == 1


@pytest.mark.parametrize("limit", [2, None])
def test_table_previews_report_totals_and_preserve_full_native_csv(tmp_path, limit):
    rows = [point(x=str(i), value=str(i)) for i in range(7)]
    data = dataset(tmp_path, rows, selections=[dict(candidate_id=str(i), status="selected") for i in range(7)])
    out = tmp_path / "tables"
    tables = render.export_tables(data, out, table_row_limit=limit)
    assert (out / "points.csv").read_bytes() == (data / "points.csv").read_bytes()
    assert (out / "selections.csv").read_bytes() == (data / "selections.csv").read_bytes()
    for table in tables:
        if "total_rows" not in table:
            continue
        assert table["total_rows"] == 7
        assert table["rows"] == (7 if limit is None else 2)
        assert table["truncated"] == (limit is not None)
        assert len(read_csv(out / table["files"][0])) == table["rows"]
        assert f"{table['rows']}/7" in (out / table["files"][1]).read_text()


def test_journal_axes_dimensions_and_recorded_size_metadata(tmp_path):
    plt = render._plt()
    with plt.rc_context(render.RC):
        for count, width in ((1, 3.4), (6, 7.2)):
            fig, axes = render._axes(count)
            assert fig.get_figwidth() == width
            assert axes[0].xaxis.label.get_fontsize() >= 8
            artifact = render._save(fig, tmp_path, dict(family="mobility", metric=str(count)),
                                    [point()], {"mode": "partial_exploratory", "render_filters": {"all_local_scopes": False}},
                                    ("svg",), "Size check")
            meta = json.loads((tmp_path / artifact["metadata"]).read_text())
            assert meta["figure_size_inches"][0] == width
            assert meta["render_filters"] == {"all_local_scopes": False}
            assert "visual review required" in meta["size_policy"]


def test_csv_iterator_yields_before_reading_following_rows(tmp_path, monkeypatch):
    data = dataset(tmp_path, [point()])
    def reader(stream):
        yield point()
        raise AssertionError("read ahead")
    monkeypatch.setattr(render.csv, "DictReader", reader)
    rows = render._iter_csv(data / "points.csv")
    assert next(rows) == point()
    rows.close()


@pytest.mark.parametrize("metric", ["pre_pv_leader_type", "pre_pv_native_left_reason", "pre_pv_native_direction_set"])
def test_code_means_are_excluded_even_if_explicit_but_frequencies_remain(metric):
    assert not render._outcome(point(family="mechanism", metric=metric + "/mean"), {metric + "/mean"})
    assert render._outcome(point(family="mechanism", metric=metric + "/frequency"), None)


def test_filtered_out_rows_still_validate_intervals(tmp_path):
    data = dataset(tmp_path, [point(family="windows", window="speed_sym_3", region="rear", lane="0",
                                    metric="N", ci_low="1", ci_high="2", interval_kind="supplied")])
    with pytest.raises(ValueError, match="partial_exploratory"):
        render.render_atlas(data, tmp_path / "out", families=["mobility"], conceptual=False)
    assert not (tmp_path / "out").exists()


def test_multiple_missing_methods_have_one_na_marker_per_x():
    rows = [point(method=method, x=x, density=x, value="" if x == ".2" else "2",
                  status="missing_" + method if x == ".2" else "defined")
            for method in ("Graph", "Node", "Grid", "H1", "H2", "H4", "H8")
            for x in (".1", ".2", ".3")]
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, rows, "mobility", "a")
    assert [text.get_text() for text in ax.texts].count("NA") == 1
    curves = [line for line in ax.lines if line.get_label() != "training densities"]
    assert len(curves) == 7
    assert all(np.isnan(line.get_ydata()[1]) for line in curves)
    assert "full status/NA reasons" in render._caption(rows, {"mode": "partial_exploratory"}, "mobility")
    plt.close(fig)


def test_defined_support_and_w7_penalty_labels():
    row = point(family="contrasts", panel="W7", cohort="0.5", method="Graph−Graph",
                n_defined="3", k="", k_planned="")
    caption = render._caption([row], {"mode": "partial_exploratory"}, "contrasts")
    assert "n=5/30" in caption and "defined=3/5" in caption and "k=" not in caption
    assert "Graph penalty×0.5 minus ×1" in caption
    assert "Joint reward penalty scaling; not constrained PPO" in caption
    assert render._method_label(dict(row, cohort="2")) == "Graph penalty×2 minus ×1"
    plt = render._plt()
    fig, ax = plt.subplots()
    render._line_panel(ax, [row], "contrasts", "a")
    assert ax.lines[0].get_label() == "Graph penalty×0.5 minus ×1"
    assert ax.lines[0].get_color() == method_style("Graph")["color"]
    assert row["method"] == "Graph−Graph"
    plt.close(fig)


@pytest.mark.parametrize("family,metric,units", [
    ("selection", "validation_V", "cells/step"),
    ("ratio", "validation_R", "ratio"),
    ("selection", "validation_feasible", "boolean"),
])
def test_checkpoint_architectures_and_categorical_controls_share_page(tmp_path, monkeypatch, family, metric, units):
    common = dict(family=family, metric=metric, units=units, panel="validation",
                  x_label="checkpoint environment steps", density="", fraction="")
    rows = [point(**common, method=method, training_line=line, x=x,
                  model_id=method + line + x, value=value)
            for method in ("Grid", "Graph", "Node") for line in ("s1", "s2", "s3")
            for x, value in (("1000", "0"), ("2000", "1"))]
    rows += [point(**common, method=method, training_line="", x="",
                   model_id=method, value="1", k="", k_planned="")
             for method in ("H8", "H1", "H4", "H2")]
    captured = []
    def save(fig, output, identity, plotted, manifest, formats, caption, **extra):
        captured.append((identity, plotted, [ax.get_ylim() for ax in fig.axes],
                         [ax.get_xlabel() for ax in fig.axes], caption))
        if units == "boolean":
            assert all(ax.get_yticks().tolist() == [0, 1] for ax in fig.axes)
        render._plt().close(fig)
        return {"id": "captured"}
    monkeypatch.setattr(render, "_save", save)
    render._curve_pages(rows, tmp_path, {"mode": "completed_validation"}, ("svg",), 6)
    assert len(captured) == 1
    identity, plotted, limits, labels, caption = captured[0]
    assert [facet[-1] for facet in identity["facets"]] == ["Graph", "Node", "Grid", "fixed controls"]
    assert len(set(limits)) == 1
    assert labels[:3] == ["checkpoint environment steps"] * 3
    assert "method category" in labels[3]
    assert len(plotted) == len(rows)
    fixed = [row for row in plotted if row["training_line"] == ""]
    assert {row["x"] for row in fixed} == {"H1", "H2", "H4", "H8"}
    assert all(row["source_x"] == "" and row["source_x_label"] == "checkpoint environment steps" for row in fixed)
    assert all(row["x"] == "" for row in rows if row["training_line"] == "")
    assert "not training history" in caption and "completed_validation" in caption


def test_finish_reserves_text_height_without_large_blank_margins():
    plt = render._plt()
    with plt.rc_context(render.RC):
        fig, axes = render._axes(4)
        for ax, method in zip(axes, ("Graph", "Node", "Grid", "H2")):
            for i in range(3):
                ax.plot([0, 1], [0, 1], label=f"{method} / seed{i}")
            ax.set(xlabel="checkpoint environment steps", ylabel="Validation PV speed V")
        render._finish(fig, axes, "Validation V", "completed_validation; n=20/20. Recorded validation.")
        fig.canvas.draw()
        # tight_layout also reserves tick/axis text: 12.4% on pinned Matplotlib.
        # Keep room for those labels while rejecting the former huge margins.
        assert fig.subplotpars.bottom < .16
        assert fig.subplotpars.top > .75
        assert sum(ax.get_position().height for ax in axes[::2]) > .5
        assert all(text.get_fontsize() >= 8 for text in fig.texts)
        plt.close(fig)


def test_conceptual_atlas_image_carries_source_mode(tmp_path):
    data = dataset(tmp_path, [], mode="completed_validation")
    out = tmp_path / "atlas"
    result = render.render_atlas(data, out, families=["conceptual"], formats=("svg",))
    artifact = result["artifacts"][0]
    image = (out / artifact["files"][0]).read_text()
    assert "Source mode: completed_validation" in image
    assert "working atlas" in image
    metadata = json.loads((out / artifact["metadata"]).read_text())
    assert metadata["source_manifest"]["snapshot"]["id"] == "snapshot-123"


def test_fixed_references_reach_both_rules_once_with_original_support():
    controls = [point(method=method, rule="fixed", training_line="", k="", k_planned="",
                      value="", status="not_applicable", sd=".2", ci_low="", ci_high="")
                for method in ("H2", "B1", "B2")]
    learned = [point(rule=rule, method=method, training_line=line)
               for rule in ("S0", "C95") for method in ("Graph", "Node", "Grid")
               for line in ("s1", "s2", "all")]
    rows = learned + controls
    plotted = render._reference_views(rows)
    for rule in ("S0", "C95"):
        refs = [r for r in plotted if r.get("reference_projection") and r["rule"] == rule]
        assert len(refs) == 3
        for row in refs:
            source = next(r for r in controls if r["method"] == row["method"])
            assert row["source_rule"] == "fixed" and row["source_cohort"] == source["cohort"]
            assert all(row[key] == source[key] for key in ("value", "n", "n_planned", "k", "k_planned",
                                                          "sd", "ci_low", "ci_high", "status", "support"))
    assert not any(r["rule"] == "fixed" for r in plotted)
    assert all(r["rule"] == "fixed" for r in controls)
    assert "n is unchanged" in render._caption(plotted, {"mode": "partial_exploratory"}, "mobility")


def test_w7_references_keep_cohort_one_and_only_match_physical_scopes():
    fixed = point(panel="W7", cohort="1.0", rule="fixed", method="H2", training_line="")
    targets = [point(panel="W7", cohort=scale) for scale in ("0.5", "1.0", "2.0")]
    plotted = render._reference_views([fixed] + targets)
    refs = [r for r in plotted if r.get("reference_projection")]
    assert {r["cohort"] for r in refs} == {"0.5", "1.0", "2.0"}
    assert all(r["source_cohort"] == "1.0" and r["n"] == "5" for r in refs)
    other = dict(fixed, cohort="0.5")
    assert render._reference_views([other, targets[-1]]) == [other, targets[-1]]


@pytest.mark.parametrize("change", [
    {"road_length": "2000"}, {"fraction": ".5"}, {"panel": "E4"}, {"cohort": "other"},
    {"window": "speed_sym_3"}, {"region": "rear"}, {"lane": "1"}, {"vehicle_group": "AV"},
    {"phase": "pre-step"}, {"density": ".2"}, {"x": ".2"}, {"x_label": "lag"},
    {"units": "steps"}, {"metric": "other"}, {"profile": "other"}, {"estimand": "other"},
])
def test_fixed_projection_never_crosses_scope_or_axis(change):
    fixed = point(rule="fixed", method="H2", training_line="")
    target = point(**change)
    assert render._reference_views([fixed, target]) == [fixed, target]


def test_fixed_projection_does_not_cross_candidate_or_validation():
    fixed = point(rule="fixed", method="H2", training_line="", candidate_id="c1", model_id="m1")
    target = point(candidate_id="c2", model_id="m2")
    assert render._reference_views([fixed, target]) == [fixed, target]
    rows = [dict(r, panel="validation", x_label="checkpoint environment steps") for r in (fixed, target)]
    assert render._reference_views(rows) == rows


def test_true_family_summary_overlay_is_separate_from_individual_curves(tmp_path, monkeypatch):
    mean_estimand = "equal frozen whole-line means; no training-line pairing"
    rows = [point(method=method, training_line="all", estimand=mean_estimand, value="3")
            for method in ("Graph", "Node", "Grid")]
    rows += [point(method=method, training_line=line, estimand="mean of episode scalars", value="2")
             for method in ("Graph", "Node", "Grid") for line in ("s1", "s2")]
    rows += [point(method="H2", training_line="", rule="fixed", estimand="mean of episode scalars", value="1")]
    captured = []
    def save(fig, output, identity, plotted, *args, **kwargs):
        captured.append((identity, plotted))
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, "_save", save)
    render._curve_pages(render._reference_views(rows), tmp_path, {"mode": "partial_exploratory"}, ("svg",), 6)
    summary = [(i, r) for i, r in captured if i["layout"] == "family-summary"]
    individual = [(i, r) for i, r in captured if i["layout"] == "individual-lines"]
    assert len(summary) == len(individual) == 1
    identity, plotted = summary[0]
    assert len(identity["facets"]) == 1
    assert {r["method"] for r in plotted} == {"Graph", "Node", "Grid", "H2"}
    assert all(r["value"] == "3" and r["training_line"] == "all" for r in plotted if r["method"] != "H2")
    reference = next(r for r in plotted if r["method"] == "H2")
    assert reference["source_estimand"] == "mean of episode scalars" and reference["value"] == "1"
    assert {r["training_line"] for r in individual[0][1] if r["method"] != "H2"} == {"s1", "s2"}
    assert all(r["value"] == "2" for r in individual[0][1] if r["method"] != "H2")


@pytest.mark.parametrize("family,metric", [("windows", "pooled_speed"), ("ratio", "R")])
def test_individual_points_never_invent_a_family_summary(tmp_path, monkeypatch, family, metric):
    rows = [point(family=family, metric=metric, training_line=line) for line in ("s1", "s2")]
    captured = []
    def save(fig, output, identity, plotted, *args, **kwargs):
        captured.append((identity, plotted))
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, "_save", save)
    render._curve_pages(rows, tmp_path, {"mode": "partial_exploratory"}, ("svg",), 6)
    assert len(captured) == 1 and captured[0][0]["layout"] == "complete"
    assert captured[0][1] == rows


def test_normalized_panel_filter_preserves_native_csv_and_full_inventory(tmp_path, monkeypatch):
    rows = [point(panel=panel) for panel in ("main", "E4", "W7")]
    data = dataset(tmp_path, rows)
    captured = []
    monkeypatch.setattr(render, "_curve_pages", lambda points, *args: captured.extend(points) or [])
    monkeypatch.setattr(render, "_dashboard_pages", lambda *args: [])
    out = tmp_path / "atlas"
    result = render.render_atlas(data, out, panels=["E4"], conceptual=False, heatmaps=False)
    assert captured == [rows[1]]
    assert result["filters"]["panels"] == ["E4"]
    assert sum(s["rows"] for s in result["scopes"]) == 3
    inventory = next(f for f in result["families"] if f["family"] == "mobility")
    assert inventory["rows"] == 3 and inventory["supplied_panels"] == ["E4", "W7", "main"]
    assert (out / "tables/points.csv").read_bytes() == (data / "points.csv").read_bytes()
    tables = json.loads((out / "tables/tables.json").read_text())
    assert tables["filters"]["panels"] == ["E4"]
    previews = [t for t in tables["tables"] if "total_rows" in t]
    assert len(previews) == 1 and previews[0]["total_rows"] == 1
    assert "E4" in previews[0]["title"]


@pytest.mark.parametrize("owner", ["render_atlas", "export_tables"])
def test_unknown_normalized_panel_is_rejected(tmp_path, owner):
    data = dataset(tmp_path, [point(panel="E4")])
    with pytest.raises(ValueError, match="unknown panels.*E4_mechanisms"):
        getattr(render, owner)(data, tmp_path / "out", panels=["E4_mechanisms"])


def test_existing_control_in_comparison_is_not_duplicated():
    fixed = point(rule="fixed", method="H2", training_line="")
    existing = point(rule="S0", method="H2", training_line="", candidate_id="")
    plotted = render._reference_views([fixed, existing, point()])
    assert [r for r in plotted if r["rule"] == "S0" and r["method"] == "H2"] == [existing]
    assert fixed in plotted


@pytest.mark.parametrize("owner", ["render_atlas", "export_tables"])
def test_panel_rejections_are_streamed_before_retention(tmp_path, monkeypatch, owner):
    import weakref
    class TrackedRow(dict):
        pass
    data = dataset(tmp_path, [])
    refs = []
    def tracked(path):
        if path.name != "points.csv":
            return
        for i in range(60):
            row = TrackedRow(point(panel="E4" if i == 0 else "main"))
            refs.append(weakref.ref(row))
            yield row
            assert sum(ref() is not None for ref in refs) <= 3
    monkeypatch.setattr(render, "_iter_csv", tracked)
    if owner == "render_atlas":
        monkeypatch.setattr(render, "_curve_pages", lambda *a: [])
        monkeypatch.setattr(render, "_dashboard_pages", lambda *a: [])
        monkeypatch.setattr(render, "export_tables", lambda *a, **k: [])
        monkeypatch.setattr(render, "available_families", lambda *a: dict(families=[], source_deferred=[]))
        render.render_atlas(data, tmp_path / "atlas", panels=["E4"], conceptual=False, heatmaps=False)
    else:
        tables = render.export_tables(data, tmp_path / "tables", panels=["E4"], table_row_limit=None)
        assert [t["total_rows"] for t in tables if "total_rows" in t] == [1]


def test_scalar_profiles_share_fixed_reference_but_local_profiles_do_not():
    fixed = point(method="H2", rule="fixed", training_line="", profile="full12",
                  estimand="mean of episode scalars")
    target = point(profile="mixed", estimand="equal frozen whole-line means; no training-line pairing")
    rows = render._reference_views([fixed, target])
    reference = next(r for r in rows if r["method"] == "H2")
    assert reference["reference_projection"] and reference["profile"] == "full12"
    assert reference["n"] == fixed["n"] and reference["source_rule"] == "fixed"
    for changes in (dict(window="front"), dict(estimand="local pooled speed")):
        local = [dict(r, **changes) for r in (fixed, target)]
        assert render._reference_views(local) == local


def test_single_checkpoint_lines_do_not_facet_or_duplicate_fixed_controls(tmp_path, monkeypatch):
    fixed = point(panel="W7", cohort="1", method="H2", rule="fixed", training_line="",
                  profile="full12", estimand="mean of episode scalars")
    targets = [point(panel="W7", cohort="0.5", training_line=line,
                     candidate_id=line + "-candidate", model_id=line + "-model",
                     profile="mechanism", estimand="mean of episode scalars")
               for line in ("a", "b", "c")]
    assert render._candidate_keys(targets) == ()
    projected = render._reference_views([fixed, *targets])
    refs = [r for r in projected if r.get("reference_projection")]
    assert len(refs) == 1 and not refs[0].get("candidate_id") and not refs[0].get("model_id")
    assert refs[0]["source_cohort"] == "1" and refs[0]["n"] == "5"
    captured = []
    def save(fig, output, identity, plotted, *args, **kwargs):
        captured.append((identity, plotted))
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, "_save", save)
    render._curve_pages(projected, tmp_path, {"mode": "partial_exploratory"}, ("svg",), 6)
    assert len(captured) == 1 and len(captured[0][0]["facets"]) == 2
    assert len(captured[0][1]) == 4
    collision = dict(targets[0], candidate_id="second", model_id="second-model")
    multiple = render._reference_views([fixed, *targets, collision])
    refs = [r for r in multiple if r.get("reference_projection")]
    assert len(refs) == 4
    assert render._candidate_keys(multiple) == ("candidate_id", "model_id")
    with pytest.raises(ValueError, match="ambiguous repeated"):
        render._series_arrays([targets[0], targets[0]], ["0.1"])


@pytest.mark.parametrize("domain", [("0.1", "0.2", "0.3", "0.6"), ("A", "B", "C", "D")])
def test_sparse_domain_keeps_na_inside_axes_and_normal_saved_width(tmp_path, domain):
    import warnings
    plt = render._plt()
    with plt.rc_context(render.RC), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, axes = render._axes(2)
        rows = [point(x=x, value="2" if i == 3 else "") for i, x in enumerate(domain)]
        for ax in axes:
            render._line_panel(ax, rows, "control", "a")
        render._finish(fig, axes, "Sparse planned conditions", "Missing values remain NA.")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in axes:
            for text in ax.texts:
                if text.get_text() == "NA":
                    box = text.get_window_extent(renderer)
                    assert ax.bbox.contains(box.x0, box.y0)
                    assert ax.bbox.contains(box.x1, box.y1)
            assert np.isnan(ax.lines[0].get_ydata()[:3]).all()
        path = tmp_path / "sparse.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        assert 1100 < plt.imread(path).shape[1] < 1350
        plt.close(fig)


def test_six_panel_dense_caption_and_legend_do_not_overlap(tmp_path):
    import warnings
    plt = render._plt()
    with plt.rc_context(render.RC), warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        fig, axes = render._axes(6)
        rows = [point(method=method, training_line=f"E3{method.lower()}_s{seed}",
                      family="mechanism", metric="control_applied_nonstay/sentinel_count", x="0.6",
                      phase="pre_step_decision_context", support=str(1000 + seed), value="0")
                for method in ("Graph", "Node", "Grid") for seed in range(1, 4)]
        for i, ax in enumerate(axes):
            render._line_panel(ax, rows, "mechanism", chr(97 + i))
            ax.set_title("fraction=0.9; " + ("Graph", "Node", "Grid")[i % 3])
        caption = render._caption(rows, {"mode": "partial_exploratory"}, "mechanism")
        caption += " Native benchmark context and recorded availability remain distinct." * 4
        render._finish(fig, axes, "mechanism · E4 · 1.0 · S0 · 1000", caption)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        boxes = [ax.get_tightbbox(renderer) for ax in axes]
        for upper, lower in zip(boxes, boxes[2:]):
            assert upper.y0 > lower.y1
        for left, right in zip(boxes[::2], boxes[1::2]):
            assert left.x1 < right.x0
        legend = fig.legends[0].get_window_extent(renderer)
        figure_box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
        for text in fig.texts:
            box = text.get_window_extent(renderer)
            assert figure_box.contains(box.x0, box.y0)
            assert figure_box.contains(box.x1, box.y1)
        assert legend.y0 > max(box.y1 for box in boxes)
        footer = next(t for t in fig.texts if t.get_text().startswith("partial_exploratory"))
        assert footer.get_window_extent(renderer).y1 < min(box.y0 for box in boxes)
        assert fig.get_figwidth() == 7.2
        assert all(t.get_fontsize() >= 8 for t in fig.texts)
        def check_saved_layout(event):
            # savefig(tight) draws again with a cropped figure bbox. Figure
            # legend anchors must stay in the original figure coordinates.
            bounds = [ax.get_tightbbox(event.renderer) for ax in axes]
            assert fig.legends[0].get_window_extent(event.renderer).y0 > max(b.y1 for b in bounds)
        fig.canvas.mpl_connect("draw_event", check_saved_layout)
        fig.savefig(tmp_path / "dense.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def test_mechanism_defaults_select_names_not_values_and_keep_native_csv(tmp_path, monkeypatch):
    metrics = ["gap/frequency", "gap/mean", "vehicle_type/frequency", "vehicle_type/mean",
               "gap/count", "gap/sentinel_count", "gap/denominator", "gap/contexts",
               "gap/min", "gap/max", "protocol/version"]
    rows = [point(family="mechanism", metric=metric, value="0") for metric in metrics]
    data = dataset(tmp_path, rows)
    captured = []
    monkeypatch.setattr(render, "_curve_pages", lambda points, *args: captured.extend(points) or [])
    dashboards = []
    monkeypatch.setattr(render, "_dashboard_pages", lambda points, *args: dashboards.extend(points) or [])
    out = tmp_path / "default"
    result = render.render_atlas(data, out, families=["mechanism"], conceptual=False, heatmaps=False)
    assert {r["metric"] for r in captured} == {"gap/frequency", "gap/mean", "vehicle_type/frequency"}
    assert dashboards == captured
    assert result["filters"]["default_mechanism_outcomes"] == ["*/frequency", "noncategorical */mean"]
    assert (out / "tables/points.csv").stat().st_ino == (data / "points.csv").stat().st_ino
    captured.clear()
    render.render_atlas(data, tmp_path / "explicit", families=["mechanism"],
                        metrics=["gap/sentinel_count"], conceptual=False, heatmaps=False)
    assert [r["metric"] for r in captured] == ["gap/sentinel_count"]


def test_caption_summarizes_large_native_support_as_range():
    rows = [point(support=str(i), support_kind="applicable_contexts") for i in range(1000)]
    caption = render._caption(rows, {"mode": "partial_exploratory"}, "mechanism")
    assert "applicable contexts=0–999" in caption
    assert len(caption) < 500


@pytest.mark.parametrize('family,metric,x_label', [
    ('state', 'N_frequency', 'N'),
    ('mechanism', 'gap/frequency', 'native category'),
    ('events', 'stop/duration_steps/left=False/right=False', 'steps'),
    ('temporal', 'pv_speed/value', 'lag'),
])
def test_sparse_fixed_curve_keeps_tail_in_views_and_summary(tmp_path, monkeypatch, family, metric, x_label):
    context = dict(family=family, metric=metric, x_label=x_label, panel='E4', cohort='1',
                   window='front', region='full', lane='all', vehicle_group='background',
                   phase='post-step measured states', profile='mechanism')
    fixed = [point(**context, method='B1', rule='fixed', training_line='', x=x, value=value)
             for x, value in (('1', '.3'), ('2', '.7'))]
    targets = [point(**context, training_line=line, x='1', value='1') for line in ('all', 'a')]
    others = [dict(fixed[1], **change) for change in (
        {'fraction': '.5'}, {'phase': 'pre-step'}, {'cohort': '2'},
        {'density': '.2'}, {'profile': 'full12'}, {'candidate_id': 'other'},
    )]
    plotted = render._reference_views([*fixed, *targets, *others])
    refs = [r for r in plotted if r.get('reference_projection')]
    assert len(refs) == 2
    assert {(r['x'], r['value']) for r in refs} == {('1', '.3'), ('2', '.7')}
    for source, reference in zip(fixed, refs):
        assert all(reference[k] == v for k, v in source.items() if k != 'rule')
        assert reference['source_rule'] == 'fixed'
    assert all(r in plotted for r in others)
    assert not any(r is source for r in plotted for source in fixed)
    captured = []
    def save(fig, output, identity, rows, *args, **kwargs):
        captured.append((identity, rows))
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, '_save', save)
    render._curve_pages([*targets, *refs], tmp_path, {'mode': 'partial_exploratory'}, ('svg',), 6)
    summary = [rows for identity, rows in captured if identity['layout'] == 'family-summary']
    assert len(summary) == 1
    assert [r for r in summary[0] if r['method'] == 'B1'] == refs


def test_sparse_reference_collision_owns_only_its_bin():
    fixed = [point(method='B1', rule='fixed', training_line='', x_label='N', x=x)
             for x in ('1', '2')]
    existing = dict(fixed[0], rule='S0', value='.9')
    target = point(x_label='N', x='1')
    rows = render._reference_views([*fixed, existing, target, dict(target, x='3')])
    refs = [r for r in rows if r.get('reference_projection')]
    assert len(refs) == 1 and refs[0]['x'] == '2'
    assert fixed[0] in rows and fixed[1] not in rows and existing in rows


@pytest.mark.parametrize("metric", ["pass95_count", "fail95_count", "undefined_count", "R", "R_min"])
def test_ratio_references_and_limits_follow_metric(tmp_path, monkeypatch, metric):
    is_ratio = metric in {"R", "R_min"}
    rows = [point(family="ratio", metric=metric, units="ratio" if is_ratio else "cells",
                  x="planned grid", x_label="planned domain", value="10")]
    captured = []
    def save(fig, output, identity, plotted, manifest, formats, caption):
        ax = fig.axes[0]
        references = [line for line in ax.lines if line.get_transform() == ax.get_yaxis_transform()]
        assert [list(line.get_ydata()) for line in references] == ([[1, 1], [.95, .95]] if is_ratio else [])
        labels = {text.get_text() for text in ax.texts}
        assert ("R=1" in labels) == is_ratio
        assert ("screen .95" in labels) == is_ratio
        assert (ax.get_ylim()[0] < 1) == is_ratio
        assert plotted == rows
        if not is_ratio:
            assert "Counts of planned cells" in caption
            assert "R >= .95" in caption and "R < .95" in caption
            assert "undefined means R unavailable" in caption
            assert "The .95 line" not in caption
        captured.append(identity)
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, "_save", save)
    render._curve_pages(rows, tmp_path, {"mode": "final"}, ("png",), 6)
    assert len(captured) == 1


@pytest.mark.parametrize("kind,low,high,expected", [
    ("crossed_approximate", "1", "3", True),
    ("crossed_approximate", "", "", False),
    ("descriptive_sd", "", "", False),
    ("none", "", "", False),
    ("supplied", "1", "3", False),
])
def test_ci_caption_is_descriptive_and_conditional(kind, low, high, expected):
    row = point(interval_kind=kind, ci_low=low, ci_high=high, sd=".2")
    caption = render._caption([row], {"mode": "final"}, "contrasts")
    assert ("95% descriptive CI, conditional on frozen selection" in caption) == expected
    assert kind in caption
    assert "SD is descriptive, not CI." in caption


@pytest.mark.parametrize("identity,conditions", [
    ({"road_length": 1000, "density": .1, "fraction": .9}, "L=1000, ρ=0.1, f=0.9"),
    ({"cell": "rho20-f90", "road_length": "1000"}, "rho20-f90"),
])
def test_recorded_conditions_visible_in_all_titles(tmp_path, monkeypatch, identity, conditions):
    captured = []
    def save(fig, output, identity, rows, manifest, formats, caption):
        assert any(conditions in text.get_text() for text in fig.texts)
        captured.append(identity["metric"])
        render._plt().close(fig)
        return {}
    monkeypatch.setattr(render, "_save", save)
    render.render_recorded_illustration(recorded_pair(), tmp_path / "recorded", identity=identity,
        selection_evidence={"rule": "frozen_median_line_then_environment", "status": "selected", "n": 90},
        formats=("png",))
    assert captured == ["maps", "PV-trace", "storyboard"]
