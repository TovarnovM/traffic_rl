"""Passive, physical-ID-aligned NumPy observations; see OBSERVER_SCHEMA.md."""

from collections.abc import Sequence

import numpy as np


WINDOW_IDS = tuple(
    f"{kind}_{shape}_{m}"
    for kind in ("fixed", "speed")
    for m in (1, 2, 3)
    for shape in ("sym", "front")
)
MAX_ARRAY_BYTES = 1 << 30


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _bounds(speed):
    return np.asarray([
        pair
        for d in (5, max(int(speed), 1))
        for m in (1, 2, 3)
        for pair in ((-2 * m * d, 2 * m * d), (-m * d, 3 * m * d))
    ], dtype=np.int16)


def _scopes(hist, windows=False):
    """Append full region (if local), all lanes, and pooled background."""
    if windows:
        hist = np.concatenate((hist, hist.sum(axis=2, keepdims=True, dtype=np.uint64)), axis=2)
    hist = np.concatenate((hist, hist.sum(axis=-3, keepdims=True, dtype=np.uint64)), axis=-3)
    return np.concatenate((hist, hist.sum(axis=-2, keepdims=True, dtype=np.uint64)), axis=-2)


def _metrics(hist, area):
    """Reduce time only; integer numerators stay exact, undefined ratios are None."""
    n = hist.sum(axis=-1, dtype=np.uint64)
    s = (hist * np.arange(hist.shape[-1], dtype=np.uint64)).sum(axis=-1, dtype=np.uint64)
    z = hist[..., 0]
    area = np.broadcast_to(area, n.shape)
    steps = hist.shape[0]
    totals = {"N": n.sum(axis=0, dtype=np.uint64),
              "S": s.sum(axis=0, dtype=np.uint64),
              "Z": z.sum(axis=0, dtype=np.uint64),
              "A": area.sum(axis=0, dtype=np.uint64)}
    empty = (n == 0).sum(axis=0, dtype=np.uint64)

    def ratio(num, den):
        num, den = np.broadcast_arrays(num, den)
        out = np.empty(num.shape, dtype=object)
        out.fill(None)
        defined = den != 0
        out[defined] = num[defined].astype(np.float64) / den[defined]
        return out.tolist()

    N, S, Z, A = (totals[key] for key in ("N", "S", "Z", "A"))
    # Background speeds <=5; signed arithmetic also permits the all-vehicle scope.
    deficit = 5 * N.astype(np.int64) - S.astype(np.int64)
    result = {key: value.tolist() for key, value in totals.items()}
    result.update(
        vehicle_time=N.tolist(), cell_time=A.tolist(),
        defined_steps=(steps - empty).tolist(), empty_steps=empty.tolist(),
        pooled_speed=ratio(S, N), pooled_stopped_fraction=ratio(Z, N),
        cell_time_density=ratio(N, A), cell_time_flow=ratio(S, A),
        time_mean_density=ratio((n / area).sum(axis=0), steps),
        time_mean_flow=ratio((s / area).sum(axis=0), steps),
        empty_fraction=ratio(empty, steps),
        speed_deficit_sum=deficit.tolist(), speed_deficit=ratio(deficit, N),
        normalized_local_delay=ratio(deficit, 5 * N),
    )
    return result


class WindowObserver:
    """Observe a closed, single-cell-vehicle ring without retaining input arrays.

    ``horizon`` is an exact contract: finish requires that many successful calls.
    Invalid observations do not advance time. Returned arrays are detached copies.
    """

    def __init__(self, initial_state, *, pv_id: int, av_ids: Sequence[int],
                 road_length: int, num_lanes: int, horizon: int, profile: str = "full12"):
        if profile not in ("full12", "mechanism"):
            raise ValueError("profile must be full12 or mechanism")
        self.road_length = _integer(road_length, "road_length", 109, 65535)
        self.num_lanes = _integer(num_lanes, "num_lanes", 1, 256)
        self.horizon = _integer(horizon, "horizon", 1, np.iinfo(np.int64).max // 6)
        self.pv_id = _integer(pv_id, "pv_id", 0, np.iinfo(np.int64).max)
        av = [_integer(x, "av_id", 0, np.iinfo(np.int64).max) for x in av_ids]
        if len(av) != len(set(av)):
            raise ValueError("av_ids must be unique")
        self._av_ids = set(av) - {self.pv_id}
        self.profile = profile
        self._step = 0
        self._previous = self._snapshot(initial_state)
        ids, lane, pos, speed, _ = self._previous
        self._ids = ids
        if not self._av_ids.issubset(set(ids.tolist())):
            raise ValueError("av_ids must identify live initial vehicles")
        self._pv = int(np.flatnonzero(ids == self.pv_id)[0])
        self._group = np.asarray([int(int(x) in self._av_ids) for x in ids], dtype=np.intp)
        T, M = self.horizon, self.num_lanes
        # Bound all count/speed/area sums before uint64 or signed deficit arithmetic.
        if T * max(M * self.road_length, len(ids) * 6) > np.iinfo(np.int64).max:
            raise ValueError("horizon/geometry exceeds accumulator capacity")
        if int(pos[self._pv]) + 6 * T > np.iinfo(np.int64).max:
            raise ValueError("PV path exceeds int64 capacity")
        specs = {
            "window_hist": ((T, 12, 3, M, 2, 6), np.uint16),
            "initial_window_hist": ((12, 3, M, 2, 6), np.uint16),
            "global_hist": ((T, M, 2, 6), np.uint16),
            "initial_global_hist": ((M, 2, 6), np.uint16),
            "window_bounds": ((T + 1, 12, 2), np.int16),
            "window_transitions": ((T, 12, 3, M, 2, 4), np.uint16),
            "global_lane_changes": ((T, M, 2), np.uint16),
            "step": ((T + 1,), np.int64),
            "pv_speed": ((T + 1,), np.uint8),
            "pv_lane": ((T + 1,), np.uint8),
            "pv_pos": ((T + 1,), np.int64),
            "pv_path": ((T + 1,), np.int64),
            "spatial_hist": ((91, M, 2, 6), np.uint64),
            "spatial_edges": ((92,), np.int16),
        }
        if profile == "mechanism":
            specs.update({f"local_{name}": ((T, 41, M), dtype)
                          for name, dtype in (("occupancy", np.uint8),
                                              ("speed", np.int8), ("type", np.int8))})
        total_bytes = sum(int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
                          for shape, dtype in specs.values())
        if total_bytes > MAX_ARRAY_BYTES:
            raise ValueError("observer arrays exceed 1 GiB byte limit")
        self._arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype) in specs.items()}
        self._arrays["step"][:] = np.arange(T + 1, dtype=np.int64)
        self._arrays["spatial_edges"][:] = np.arange(-36, 56, dtype=np.int16)
        bounds, wh, gh, _ = self._histograms(self._previous)
        self._arrays["window_bounds"][0] = bounds
        self._arrays["initial_window_hist"][:] = wh
        self._arrays["initial_global_hist"][:] = gh
        for name, value in (("pv_speed", speed[self._pv]), ("pv_lane", lane[self._pv]),
                            ("pv_pos", pos[self._pv]), ("pv_path", pos[self._pv])):
            self._arrays[name][0] = value

    def _snapshot(self, state):
        fields = []
        shape = None
        for name in ("vehicle_id", "lane", "pos", "vel", "alive", "changed_lane"):
            arr = getattr(state, name, None)
            if not isinstance(arr, np.ndarray) or arr.ndim != 1:
                raise ValueError(f"{name} must be a 1D NumPy array")
            if shape is not None and arr.shape != shape:
                raise ValueError("state arrays must have matching shapes")
            shape = arr.shape
            if name in ("alive", "changed_lane"):
                if arr.dtype != np.bool_:
                    raise ValueError(f"{name} must have boolean dtype")
            elif arr.dtype.kind not in "iu":
                raise ValueError(f"{name} must have integer dtype")
            fields.append(arr)
        ids, lane, pos, speed, alive, changed = fields
        values = [a[alive] for a in (ids, lane, pos, speed, changed)]
        ids, lane, pos, speed, changed = values
        if len(ids) > 65535:
            raise ValueError("live count exceeds uint16 capacity")
        if np.any(ids > np.iinfo(np.int64).max) or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError("live physical IDs must be unique nonnegative int64 values")
        if np.count_nonzero(ids == self.pv_id) != 1:
            raise ValueError("PV must be alive exactly once")
        if np.any(lane < 0) or np.any(lane >= self.num_lanes):
            raise ValueError("lane out of range")
        if np.any(pos < 0) or np.any(pos >= self.road_length):
            raise ValueError("position out of range")
        if np.any(speed < 0) or np.any(speed > np.where(ids == self.pv_id, 6, 5)):
            raise ValueError("unsupported velocity: background 0..5, PV 0..6")
        if hasattr(state, "length"):
            length = state.length
            if not isinstance(length, np.ndarray) or length.shape != shape or np.any(length[alive] != 1):
                raise ValueError("observer requires single-cell vehicles")
        cells = lane.astype(np.int64) * self.road_length + pos.astype(np.int64)
        if len(np.unique(cells)) != len(cells):
            raise ValueError("multiple live vehicles occupy one cell")
        order = np.argsort(ids)
        return tuple(a[order].astype(np.bool_ if i == 4 else np.int64, copy=True)
                     for i, a in enumerate(values))

    def _histograms(self, snapshot):
        ids, lane, pos, speed, _ = snapshot
        bounds = _bounds(speed[self._pv])
        relative = (pos - pos[self._pv] + self.road_length // 2) % self.road_length - self.road_length // 2
        bg = ids != self.pv_id
        gh = np.zeros((self.num_lanes, 2, 6), dtype=np.uint16)
        np.add.at(gh, (lane[bg], self._group[bg], speed[bg]), 1)
        wh = np.zeros((12, 3, self.num_lanes, 2, 6), dtype=np.uint16)
        region = np.sign(relative).astype(np.intp) + 1
        for w, (low, high) in enumerate(bounds):
            keep = bg & (relative >= low) & (relative <= high)
            np.add.at(wh[w], (region[keep], lane[keep], self._group[keep], speed[keep]), 1)
        return bounds, wh, gh, relative

    def observe(self, post_state):
        if self._step >= self.horizon:
            raise ValueError("observer horizon exceeded")
        current = self._snapshot(post_state)
        ids, lane, pos, speed, changed = current
        if not np.array_equal(ids, self._ids):
            raise ValueError("closed ring requires unchanged live physical IDs")
        _, old_lane, old_pos, old_speed, _ = self._previous
        if not np.array_equal(pos, (old_pos + speed) % self.road_length):
            raise ValueError("post-step positions do not match modular velocity movement")
        if not np.array_equal(changed, lane != old_lane):
            raise ValueError("changed_lane disagrees with physical-ID lane movement")
        bounds, wh, gh, relative = self._histograms(current)
        t, a = self._step, self._arrays
        bg = ids != self.pv_id
        region = np.sign(relative).astype(np.intp) + 1
        transitions = np.stack(((old_speed == 0) & (speed > 0),
                                (old_speed > 0) & (speed == 0),
                                speed > old_speed, speed < old_speed), axis=1)
        for w, (low, high) in enumerate(bounds):
            for k in range(4):
                keep = bg & (relative >= low) & (relative <= high) & transitions[:, k]
                np.add.at(a["window_transitions"][t, w],
                          (region[keep], lane[keep], self._group[keep], np.full(keep.sum(), k)), 1)
        keep = bg & changed
        np.add.at(a["global_lane_changes"][t], (lane[keep], self._group[keep]), 1)
        keep = bg & (relative >= -36) & (relative <= 54)
        np.add.at(a["spatial_hist"], (relative[keep] + 36, lane[keep], self._group[keep], speed[keep]), 1)
        if self.profile == "mechanism":
            a["local_speed"][t].fill(-1)
            a["local_type"][t].fill(-1)
            keep = bg & (relative >= -10) & (relative <= 30)
            cells = (relative[keep] + 10, lane[keep])
            a["local_occupancy"][t][cells] = 1
            a["local_speed"][t][cells] = speed[keep]
            a["local_type"][t][cells] = self._group[keep]
            # PV is excluded from background occupancy, but is not an empty cell.
            a["local_speed"][t, 10, lane[self._pv]] = -2
            a["local_type"][t, 10, lane[self._pv]] = -2
        a["window_hist"][t] = wh
        a["global_hist"][t] = gh
        a["window_bounds"][t + 1] = bounds
        a["pv_speed"][t + 1] = speed[self._pv]
        a["pv_lane"][t + 1] = lane[self._pv]
        a["pv_pos"][t + 1] = pos[self._pv]
        a["pv_path"][t + 1] = int(a["pv_path"][t]) + int(speed[self._pv])
        self._previous = current
        self._step += 1

    def finish(self) -> tuple[dict, dict[str, np.ndarray]]:
        if self._step != self.horizon:
            raise ValueError("finish requires exactly horizon measured steps")
        a, T, M = self._arrays, self.horizon, self.num_lanes
        b = a["window_bounds"][1:].astype(np.int64)
        widths = np.stack((-b[..., 0], np.ones(b.shape[:2], dtype=np.int64),
                           b[..., 1], b[..., 1] - b[..., 0] + 1), axis=-1)
        lane_factor = np.asarray([1] * M + [M], dtype=np.uint64)
        area = widths.astype(np.uint64)[..., None, None] * lane_factor[None, None, None, :, None]
        windows = _metrics(_scopes(a["window_hist"], windows=True), area)
        global_area = self.road_length * lane_factor[None, :, None]
        global_metrics = _metrics(_scopes(a["global_hist"]), global_area)
        changes = a["global_lane_changes"].sum(axis=0, dtype=np.uint64)
        changes = np.concatenate((changes, changes.sum(axis=0, keepdims=True)), axis=0)
        changes = np.concatenate((changes, changes.sum(axis=1, keepdims=True)), axis=1)
        global_metrics["lane_changes"] = changes.tolist()
        exposure = np.asarray(global_metrics["N"], dtype=np.uint64)
        global_metrics["lane_change_rate"] = [
            [int(c) / int(n) if n else None for c, n in zip(cr, nr)]
            for cr, nr in zip(changes, exposure)
        ]
        speed = a["pv_speed"][1:]
        distance = int(a["pv_path"][-1]) - int(a["pv_path"][0])
        cumulative = a["pv_path"] - a["pv_path"][0]
        crossings = (np.flatnonzero(np.diff(cumulative // self.road_length)) + 1).tolist()
        lap_times = np.diff([0] + crossings).tolist()
        stopped = int(np.count_nonzero(speed == 0))
        lane_changes = int(np.count_nonzero(np.diff(a["pv_lane"].astype(np.int16))))
        pv = dict(mean_speed=distance / T, distance=distance,
                  normalized_time_loss=1 - distance / (6 * T), stopped_steps=stopped,
                  stopped_fraction=stopped / T, moving_steps=T - stopped,
                  moving_mean_speed=distance / (T - stopped) if T > stopped else None,
                  lane_changes=lane_changes, lane_change_rate=lane_changes / T,
                  completed_laps=len(crossings), lap_completion_steps=crossings,
                  completed_lap_times=lap_times,
                  mean_completed_lap_time=sum(lap_times) / len(lap_times) if lap_times else None,
                  remaining_distance=distance % self.road_length, no_completed_laps=not crossings)
        # All-vehicle global scalar scope includes the PV's seventh speed bin.
        all_hist = np.zeros((T, 7), dtype=np.uint64)
        all_hist[:, :6] = a["global_hist"].sum(axis=(1, 2), dtype=np.uint64)
        all_hist[np.arange(T), speed] += 1
        all_metrics = _metrics(all_hist, np.full(T, M * self.road_length, dtype=np.uint64))
        for key in ("speed_deficit_sum", "speed_deficit", "normalized_local_delay"):
            del all_metrics[key]  # Vref=5 belongs to background, not PV.
        all_changes = int(changes[-1, -1]) + lane_changes
        all_metrics.update(lane_changes=all_changes, lane_change_rate=all_changes / all_metrics["N"])
        summary = dict(schema_version=1, profile=self.profile, measurement_steps=T,
                       road_length=self.road_length, num_lanes=M, pv_id=self.pv_id,
                       av_ids=sorted(self._av_ids), window_ids=list(WINDOW_IDS),
                       primary_window="fixed_front_2", regions=["rear", "centre", "front", "full"],
                       lanes=list(range(M)) + ["all"], groups=["HDV", "AV", "background"],
                       windows=windows, global_background=global_metrics, global_all=all_metrics, pv=pv,
                       speed_floor_steps=stopped, speed_floor_fraction=stopped / T,
                       spatial_cell_time_per_bin_lane=T,
                       spatial_extent=[-36, 54], spatial_extent_covers_all_windows=True,
                       realized_vehicle_count=len(self._ids),
                       realized_density=len(self._ids) / (M * self.road_length),
                       realized_background_av_fraction=len(self._av_ids) / (len(self._ids) - 1)
                       if len(self._ids) > 1 else None)
        return summary, {name: value.copy() for name, value in a.items()}
