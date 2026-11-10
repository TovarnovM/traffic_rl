"""Hand-built observer fixtures; no simulator, RNG, or Torch dependency."""

from types import SimpleNamespace

import numpy as np
import pytest

from physica_mc.observer import WINDOW_IDS, WindowObserver


def state(ids=(10,), lane=(0,), pos=(0,), vel=(0,), changed=None):
    return SimpleNamespace(
        vehicle_id=np.array(ids, dtype=np.int64, copy=True),
        lane=np.array(lane, dtype=np.int16, copy=True), pos=np.array(pos, dtype=np.int64, copy=True),
        vel=np.array(vel, dtype=np.int16, copy=True), alive=np.ones(len(ids), dtype=bool),
        changed_lane=np.zeros(len(ids), dtype=bool) if changed is None else np.array(changed, dtype=bool, copy=True),
    )


def observer(initial, **kwargs):
    args = dict(pv_id=10, av_ids=(), road_length=200, num_lanes=2, horizon=1)
    args.update(kwargs)
    return WindowObserver(initial, **args)


def advance(initial, speeds=None, lanes=None, order=None, length=200):
    speeds = initial.vel.copy() if speeds is None else np.asarray(speeds)
    lanes = initial.lane.copy() if lanes is None else np.asarray(lanes)
    result = state(initial.vehicle_id, lanes, (initial.pos + speeds) % length,
                   speeds, lanes != initial.lane)
    if order is not None:
        for name, value in vars(result).items():
            setattr(result, name, value[order].copy())
    return result


@pytest.mark.parametrize("pv_speed", [0, 1, 2, 5, 6])
def test_all_windows_inclusive_periodic_parts_and_groups(pv_speed):
    # Every longitudinal offset is occupied in the other lane, including centre.
    offsets = np.arange(-60, 61)
    final_pos = np.r_[2, (2 + offsets) % 200]
    speeds = np.r_[pv_speed, np.zeros(len(offsets), dtype=int)]
    initial = state(np.r_[10, np.arange(100, 221)], np.r_[0, np.ones(121, dtype=int)],
                    (final_pos - speeds) % 200, speeds)
    av_ids = list(range(100, 221, 2))
    obs = observer(initial, av_ids=av_ids)
    obs.observe(advance(initial))
    summary, arrays = obs.finish()
    assert list(WINDOW_IDS) == summary["window_ids"]
    for w, name in enumerate(WINDOW_IDS):
        kind, shape, scale = name.split("_")
        d = 5 if kind == "fixed" else max(pv_speed, 1)
        m = int(scale)
        low, high = (-2*m*d, 2*m*d) if shape == "sym" else (-m*d, 3*m*d)
        np.testing.assert_array_equal(arrays["window_bounds"][1, w], [low, high])
        hist = arrays["window_hist"][0, w]
        np.testing.assert_array_equal(hist.sum(axis=(1, 2, 3)), [-low, 1, high])
        assert int(hist.sum()) == high - low + 1
        assert hist[1, 1].sum() == 1  # central other-lane car survives PV exclusion
        assert hist[:, 0].sum() == 0
        for group, ids in enumerate((set(range(100, 221)) - set(av_ids), set(av_ids))):
            expected = sum(low <= offset <= high and int(vid) in ids
                           for offset, vid in zip(offsets, range(100, 221)))
            assert hist[:, :, group].sum() == expected
        assert summary["windows"]["N"][w][3][2][2] == high - low + 1
        assert summary["windows"]["A"][w][3][2][2] == 2 * (high - low + 1)
    widths = np.diff(arrays["window_bounds"][1], axis=-1).ravel() + 1
    np.testing.assert_array_equal(widths[::2], widths[1::2])
    if pv_speed == 5:
        np.testing.assert_array_equal(arrays["window_hist"][:, :6], arrays["window_hist"][:, 6:])
    assert arrays["window_hist"].dtype == np.uint16
    assert arrays["window_transitions"].dtype == np.uint16
    assert arrays["window_bounds"].dtype == np.int16
    assert arrays["pv_path"].dtype == np.int64
    assert arrays["spatial_hist"].dtype == np.uint64


def test_t0_exclusion_scalar_hand_calculation_and_variable_area():
    initial = state((10, 20), (0, 1), (0, 2), (5, 5))
    obs = observer(initial, horizon=2, av_ids=(20,))
    first = advance(initial, [0, 0])
    obs.observe(first)
    obs.observe(advance(first, [2, 2]))
    summary, arrays = obs.finish()
    assert arrays["initial_global_hist"][1, 1, 5] == 1
    assert arrays["global_hist"][0, 1, 1, 0] == 1
    # speed_sym_1 full/all lanes/background: areas 10 then 18.
    metrics = summary["windows"]
    idx = (6, 3, 2, 2)
    def at(name):
        value = metrics[name]
        for i in idx:
            value = value[i]
        return value
    assert (at("N"), at("S"), at("Z"), at("A")) == (2, 2, 1, 28)
    assert at("pooled_speed") == 1
    assert at("pooled_stopped_fraction") == .5
    assert at("time_mean_density") == pytest.approx((1/10 + 1/18)/2)
    assert at("cell_time_density") == pytest.approx(2/28)
    assert at("time_mean_flow") == pytest.approx(1/18)
    assert at("cell_time_flow") == pytest.approx(2/28)
    assert at("speed_deficit_sum") == 8
    assert at("normalized_local_delay") == .8
    assert summary["pv"]["mean_speed"] == 1
    assert summary["pv"]["moving_mean_speed"] == 2
    assert summary["pv"]["stopped_steps"] == 1
    assert summary["global_all"]["N"] == 4
    assert summary["global_all"]["S"] == 4


def test_empty_background_and_no_laps():
    initial = state(vel=(0,))
    obs = observer(initial)
    obs.observe(advance(initial))
    summary, arrays = obs.finish()
    g = summary["global_background"]
    assert g["N"][-1][-1] == 0
    assert g["A"][-1][-1] == 400
    assert g["pooled_speed"][-1][-1] is None
    assert g["pooled_stopped_fraction"][-1][-1] is None
    assert g["lane_change_rate"][-1][-1] is None
    assert g["time_mean_density"][-1][-1] == 0
    assert g["empty_fraction"][-1][-1] == 1
    assert summary["pv"]["moving_mean_speed"] is None
    assert summary["pv"]["mean_completed_lap_time"] is None
    assert summary["pv"]["no_completed_laps"]
    assert arrays["window_hist"].sum() == 0


def test_same_id_transitions_reordered_rows_and_post_membership():
    initial = state((10, 20, 30, 40), (0, 1, 1, 1), (0, 11, 4, 7), (0, 0, 2, 2))
    obs = observer(initial, av_ids=(20,))
    # PV advances; 20 enters fixed_sym_1 and accelerates, 30 stops, 40 stays moving.
    final = advance(initial, [2, 1, 0, 2], [0, 0, 1, 1], order=[2, 0, 3, 1])
    obs.observe(final)
    summary, arrays = obs.finish()
    tr = arrays["window_transitions"][0, 0]
    np.testing.assert_array_equal(tr[2, 0, 1], [1, 0, 1, 0])
    np.testing.assert_array_equal(tr[2, 1, 0], [0, 1, 0, 1])
    assert tr.sum() == 4
    assert summary["global_background"]["lane_changes"][0][1] == 1
    assert summary["global_background"]["lane_change_rate"][-1][-1] == pytest.approx(1/3)
    assert arrays["global_lane_changes"][0, 0, 1] == 1


def test_entering_window_at_unchanged_speed_is_not_a_transition():
    initial = state((10, 20), (0, 1), (0, 11), (2, 1))
    obs = observer(initial)
    obs.observe(advance(initial))
    _, arrays = obs.finish()
    assert arrays["initial_window_hist"][0].sum() == 0
    assert arrays["window_hist"][0, 0].sum() == 1
    assert arrays["window_transitions"].sum() == 0


def test_mechanism_sentinels_spatial_profile_and_passivity():
    initial = state((10, 20, 30), (0, 1, 1), (199, 199, 169), (1, 1, 1))
    original = {k: v.copy() for k, v in vars(initial).items()}
    obs = observer(initial, av_ids=(10, 20), profile="mechanism")
    final = advance(initial)
    saved = {k: v.copy() for k, v in vars(final).items()}
    for value in vars(initial).values():
        value.flags.writeable = False
    for value in vars(final).values():
        value.flags.writeable = False
    obs.observe(final)
    summary, arrays = obs.finish()
    for name, value in original.items():
        np.testing.assert_array_equal(getattr(initial, name), value)
    for name, value in saved.items():
        np.testing.assert_array_equal(getattr(final, name), value)
    assert arrays["local_occupancy"].shape == (1, 41, 2)
    assert arrays["local_occupancy"][0, 10, 0] == 0
    assert arrays["local_speed"][0, 10, 0] == -2
    assert arrays["local_type"][0, 10, 0] == -2
    assert arrays["local_occupancy"][0, 10, 1] == 1
    assert arrays["local_speed"][0, 10, 1] == 1
    assert arrays["local_type"][0, 10, 1] == 1
    assert arrays["local_speed"][0, 0, 0] == -1
    assert arrays["local_type"][0, 0, 0] == -1
    assert arrays["spatial_hist"][6, 1, 0, 1] == 1
    assert arrays["spatial_hist"][36, 1, 1, 1] == 1
    np.testing.assert_array_equal(arrays["spatial_edges"], np.arange(-36, 56))
    np.testing.assert_array_equal(arrays["pv_pos"], [199, 0])
    np.testing.assert_array_equal(arrays["pv_path"], [199, 200])
    assert summary["pv"]["completed_laps"] == 0  # origin crossing is not a lap
    arrays["pv_path"][:] = -99
    assert obs.finish()[1]["pv_path"][1] == 200


def test_laps_use_measurement_distance_and_integer_endpoint_times():
    initial = state(pos=(108,), vel=(6,))
    obs = observer(initial, road_length=109, horizon=37)
    current = initial
    for _ in range(37):
        current = advance(current, length=109)
        obs.observe(current)
    summary, arrays = obs.finish()
    pv = summary["pv"]
    assert pv["distance"] == 222
    assert pv["completed_laps"] == 2
    assert pv["lap_completion_steps"] == [19, 37]
    assert pv["completed_lap_times"] == [19, 18]
    assert pv["mean_completed_lap_time"] == 18.5
    assert pv["remaining_distance"] == 4
    assert arrays["pv_path"][-1] == 330


@pytest.mark.parametrize("kwargs", [
    {"road_length": 108}, {"road_length": 65536}, {"num_lanes": 0},
    {"num_lanes": 257}, {"horizon": 0}, {"horizon": True},
    {"horizon": 10**12}, {"horizon": 10**19}, {"profile": "off"},
    {"pv_id": 11}, {"pv_id": -1}, {"av_ids": (20,)}, {"av_ids": (10, 10)},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        observer(state(), **kwargs)


@pytest.mark.parametrize("name,value", [
    ("vehicle_id", np.array([10, 10])), ("vehicle_id", np.array([10, -1])),
    ("vel", np.array([7, 0])), ("vel", np.array([0, 6])),
    ("vel", np.array([0., 1.])), ("pos", np.array([0, 200])),
    ("lane", np.array([0, 2])), ("alive", np.array([True, False])),
    ("changed_lane", np.array([0, 0])), ("pos", np.array([[0, 2]])),
])
def test_invalid_snapshots(name, value):
    initial = state((10, 20), (0, 1), (0, 2), (0, 0))
    if name == "alive":
        value = np.array([False, True])  # missing PV
    setattr(initial, name, value)
    with pytest.raises(ValueError):
        observer(initial)


def test_count_and_geometry_overflow():
    initial = state(np.r_[10, np.arange(100, 65635)], np.zeros(65536, dtype=int),
                    np.zeros(65536, dtype=int), np.zeros(65536, dtype=int))
    with pytest.raises(ValueError, match="count"):
        observer(initial)
    initial = state((10, 20), (0, 0), (0, 0), (0, 0))
    with pytest.raises(ValueError, match="one cell"):
        observer(initial)
    initial = state()
    initial.length = np.array([2])
    with pytest.raises(ValueError, match="single-cell"):
        observer(initial)


def test_rejected_observation_does_not_advance_and_finish_contract():
    initial = state((10, 20), (0, 1), (0, 2), (1, 1))
    obs = observer(initial)
    with pytest.raises(ValueError, match="exactly"):
        obs.finish()
    bad = advance(initial)
    bad.pos[0] += 1
    with pytest.raises(ValueError, match="modular"):
        obs.observe(bad)
    bad = advance(initial)
    bad.vehicle_id[1] = 30
    with pytest.raises(ValueError, match="physical IDs"):
        obs.observe(bad)
    bad = advance(initial)
    bad.changed_lane[1] = True
    with pytest.raises(ValueError, match="changed_lane"):
        obs.observe(bad)
    obs.observe(advance(initial))
    assert obs.finish()[0]["measurement_steps"] == 1
    with pytest.raises(ValueError, match="horizon"):
        obs.observe(advance(initial))


def test_pooled_speed_weights_vehicles_and_retains_empty_steps():
    # fixed_sym_1: first two cars at speeds 0,2; then only the stopped car;
    # finally PV moves away from both. Instantaneous defined speeds are 1,0.
    initial = state((10, 20, 30), (0, 1, 1), (0, 191, 8), (0, 0, 0))
    obs = observer(initial, horizon=3)
    first = advance(initial, [0, 0, 2])
    second = advance(first, [0, 0, 2])
    third = advance(second, [6, 0, 5])
    for current in (first, second, third):
        obs.observe(current)
    summary, _ = obs.finish()
    w = summary["windows"]
    assert w["N"][0][3][-1][-1] == 3
    assert w["S"][0][3][-1][-1] == 2
    assert w["pooled_speed"][0][3][-1][-1] == pytest.approx(2/3)
    assert w["pooled_stopped_fraction"][0][3][-1][-1] == pytest.approx(2/3)
    assert w["defined_steps"][0][3][-1][-1] == 2
    assert w["empty_steps"][0][3][-1][-1] == 1
    assert w["empty_fraction"][0][3][-1][-1] == pytest.approx(1/3)


def test_snapshots_are_copied_and_dead_rows_are_ignored():
    initial = state((10, 20, 99), (0, 1, -1), (0, 2, -1), (0, 0, -1))
    initial.alive[2] = False
    obs = observer(initial)
    # A simulator may reuse its state arrays after constructing the observer.
    initial.vel[:2] = [1, 2]
    initial.pos[:2] = [1, 4]
    initial.lane[0] = 1
    initial.changed_lane[0] = True
    obs.observe(initial)
    summary, arrays = obs.finish()
    assert arrays["initial_global_hist"][1, 0, 0] == 1
    assert arrays["window_transitions"][0, 0, 2, 1, 0, 0] == 1
    assert summary["pv"]["lane_changes"] == 1
    assert summary["pv"]["lane_change_rate"] == 1
    assert summary["global_all"]["lane_changes"] == 1
    assert summary["realized_vehicle_count"] == 2


def test_spatial_diagnostic_extent_does_not_clip_full_windows():
    # At post-step PV position 6, these cars sit at -36, -30, +54, +55.
    initial = state((10, 20, 30, 40, 50), (0, 1, 1, 1, 1),
                    (0, 170, 176, 60, 61), (6, 0, 0, 0, 0))
    obs = observer(initial)
    obs.observe(advance(initial))
    summary, arrays = obs.finish()
    assert arrays["window_hist"][0, 10, 0].sum() == 2
    assert arrays["window_hist"][0, 11, 2].sum() == 1
    assert arrays["spatial_hist"].sum() == 3
    assert arrays["spatial_hist"][0].sum() == 1
    assert arrays["spatial_hist"][6].sum() == 1
    assert arrays["spatial_hist"][-1].sum() == 1
    assert summary["spatial_extent_covers_all_windows"]


def test_json_summary_has_no_nonfinite_or_numpy_scalars():
    import json

    obs = observer(state())
    obs.observe(state())
    summary, arrays = obs.finish()
    json.dumps(summary, allow_nan=False)
    assert all(array.dtype.kind != "O" for array in arrays.values())
    assert "local_speed" not in arrays
