from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import numpy as np

from snfs_traffic.core import SimulationParams, TrafficState
from snfs_traffic.scenarios import AV_VEH_TYPE, BUS_VEH_TYPE
from snfs_traffic.topology import RingTopology


def _require_pygame():
    import os

    if not os.environ.get("DISPLAY") and "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    try:
        import pygame
        import pygame.surfarray
    except ImportError as exc:
        raise RuntimeError(
            "RoadRenderer requires visualization dependencies. Install with: python -m pip install -e '.[viz]'"
        ) from exc
    if not pygame.get_init():
        pygame.init()
    if not pygame.display.get_init():
        pygame.display.init()
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1))
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame


@dataclass(frozen=True, slots=True)
class VehicleRenderState:
    vehicle_id: int
    pos: float
    lane: float
    length: int
    vel: int
    veh_type: int
    behavior_id: int
    controlled: bool
    changed_lane: bool


@dataclass(frozen=True, slots=True)
class RoadRenderConfig:
    width: int = 1280
    height: int = 320
    cell_w: int = 28
    cell_h: int = 54
    y0: int = 44
    interpolation_frames: int = 8
    camera: str = "follow"
    follow: str | None = "first-controlled"
    follow_vehicle_id: int | None = None
    fallback_to_first_alive: bool = True
    fixed_start_cell: float = 0.0
    focus_x_ratio: float = 0.45
    draw_grid: bool = True
    draw_cell_numbers: bool = True
    draw_lane_lines: bool = True
    draw_step: bool = True
    draw_interpolation_progress: bool = True
    draw_focus_velocity: bool = True
    draw_speed: bool = True
    draw_vehicle_ids: bool = False
    draw_changed_lane_marker: bool = True
    body_mode: str = "head"
    font_size: int = 18

    def __post_init__(self) -> None:
        for n in ("width", "height", "cell_w", "cell_h", "interpolation_frames", "font_size"):
            v = getattr(self, n)
            if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                raise ValueError(f"{n} must be a positive int")
        if not (0.0 < self.focus_x_ratio < 1.0):
            raise ValueError("focus_x_ratio must be in (0.0, 1.0)")
        if self.camera not in {"follow", "fixed"}:
            raise ValueError("camera must be one of {'follow','fixed'}")
        if self.follow not in {"first-controlled", "first-alive", None}:
            raise ValueError("follow must be one of {'first-controlled','first-alive',None}")
        if self.body_mode not in {"head", "state_length"}:
            raise ValueError("body_mode must be one of {'head','state_length'}")


def snapshot_from_state(state: TrafficState) -> dict[int, VehicleRenderState]:
    out: dict[int, VehicleRenderState] = {}
    for i in range(state.n_vehicles):
        if not bool(state.alive[i]):
            continue
        vid = int(state.vehicle_id[i])
        out[vid] = VehicleRenderState(
            vehicle_id=vid,
            pos=float(state.pos[i]),
            lane=float(state.lane[i]),
            length=int(state.length[i]),
            vel=int(state.vel[i]),
            veh_type=int(state.veh_type[i]),
            behavior_id=int(state.behavior_id[i]),
            controlled=bool(state.controlled[i]),
            changed_lane=bool(state.changed_lane[i]),
        )
    return out


def select_follow_vehicle_id(snapshot: dict[int, VehicleRenderState], *, follow_vehicle_id: int | None, follow: str | None, fallback_to_first_alive: bool) -> int | None:
    if not snapshot:
        return None
    if follow_vehicle_id is not None:
        if follow_vehicle_id not in snapshot:
            raise ValueError(f"follow_vehicle_id={follow_vehicle_id} is not present in current snapshot")
        return follow_vehicle_id
    if follow == "first-controlled":
        for vid, veh in snapshot.items():
            if veh.controlled:
                return vid
        if fallback_to_first_alive:
            return next(iter(snapshot))
        return None
    if follow == "first-alive":
        return next(iter(snapshot))
    return None


def interpolate_snapshots(prev: dict[int, VehicleRenderState], curr: dict[int, VehicleRenderState], *, road_length: int, alpha: float) -> dict[int, VehicleRenderState]:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0.0,1.0]")
    out: dict[int, VehicleRenderState] = {}
    for vid, new in curr.items():
        old = prev.get(vid)
        if old is None:
            out[vid] = new
            continue
        dx = (new.pos - old.pos) % road_length
        x = (old.pos + dx * alpha) % road_length
        lane = old.lane + (new.lane - old.lane) * alpha
        out[vid] = replace(new, pos=float(x), lane=float(lane))
    return out


def _iter_periodic_screen_x(*, world_x_px: float, offset_x_px: float, road_px_len: float, width_px: int, margin_px: float) -> Iterator[float]:
    base = world_x_px + offset_x_px
    min_x = -margin_px
    max_x = width_px + margin_px
    if road_px_len <= 0:
        if min_x <= base <= max_x:
            yield base
        return
    k0 = int(np.floor((min_x - base) / road_px_len))
    k1 = int(np.ceil((max_x - base) / road_px_len))
    for k in range(k0, k1 + 1):
        x = base + k * road_px_len
        if min_x <= x <= max_x:
            yield x


class RoadRenderer:
    def __init__(self, *, params: SimulationParams, topology: RingTopology, config: RoadRenderConfig | None = None, **config_overrides) -> None:
        if topology.num_lanes != params.num_lanes or topology.length != params.road_length or topology.boundary != "periodic":
            raise ValueError("params and topology must describe the same periodic ring")
        self.params = params
        self.topology = topology
        cfg = config or RoadRenderConfig()
        if config_overrides:
            cfg = replace(cfg, **config_overrides)
        self.config = cfg
        self._pygame = _require_pygame()
        self._surface = self._pygame.Surface((cfg.width, cfg.height))
        self._font = self._pygame.font.SysFont(None, cfg.font_size)
        self._prev_snapshot: dict[int, VehicleRenderState] | None = None
        self._follow_vehicle_id: int | None = None

    def reset(self, state: TrafficState, *, follow_vehicle_id: int | None = None) -> None:
        snap = snapshot_from_state(state)
        self._prev_snapshot = snap
        self._follow_vehicle_id = select_follow_vehicle_id(
            snap,
            follow_vehicle_id=follow_vehicle_id if follow_vehicle_id is not None else self.config.follow_vehicle_id,
            follow=self.config.follow,
            fallback_to_first_alive=self.config.fallback_to_first_alive,
        )

    def render_step(self, state: TrafficState, *, step: int | None = None) -> list[np.ndarray]:
        if self._prev_snapshot is None:
            self.reset(state)
            return []
        curr = snapshot_from_state(state)
        if self._follow_vehicle_id is not None and self._follow_vehicle_id not in curr:
            if self.config.fallback_to_first_alive:
                self._follow_vehicle_id = select_follow_vehicle_id(curr, follow_vehicle_id=None, follow="first-alive", fallback_to_first_alive=True)
            else:
                raise ValueError("Focus vehicle disappeared and fallback_to_first_alive is disabled")
        frames = []
        for k in range(1, self.config.interpolation_frames + 1):
            alpha = k / self.config.interpolation_frames
            inter = interpolate_snapshots(self._prev_snapshot, curr, road_length=self.params.road_length, alpha=alpha)
            frames.append(self._render(inter, step=step, alpha=alpha))
        self._prev_snapshot = curr
        return frames

    def _render(self, snapshot: dict[int, VehicleRenderState], *, step: int | None, alpha: float) -> np.ndarray:
        c = self.config
        pg = self._pygame
        self._surface.fill((20, 22, 28))
        road_h = c.cell_h * self.params.num_lanes
        pg.draw.rect(self._surface, (35, 38, 45), pg.Rect(0, c.y0, c.width, road_h))
        focus = snapshot.get(self._follow_vehicle_id) if self._follow_vehicle_id is not None else None
        offset = (-c.fixed_start_cell * c.cell_w) if c.camera == "fixed" else (0.0 if focus is None else (c.focus_x_ratio * c.width - focus.pos * c.cell_w))
        road_px = self.params.road_length * c.cell_w
        if c.draw_grid:
            for cell in range(self.params.road_length):
                wx = cell * c.cell_w
                for sx in _iter_periodic_screen_x(world_x_px=wx, offset_x_px=offset, road_px_len=road_px, width_px=c.width, margin_px=1):
                    pg.draw.line(self._surface, (55, 58, 68), (sx, c.y0), (sx, c.y0 + road_h), 1)
                    if c.draw_cell_numbers and cell % 5 == 0:
                        txt = self._font.render(str(cell), True, (140, 145, 160))
                        self._surface.blit(txt, (sx + 2, c.y0 - c.font_size - 2))
        if c.draw_lane_lines:
            for li in range(self.params.num_lanes + 1):
                y = c.y0 + li * c.cell_h
                pg.draw.line(self._surface, (95, 100, 112), (0, y), (c.width, y), 2 if li in {0, self.params.num_lanes} else 1)

        for vid, v in snapshot.items():
            lane_y = c.y0 + (self.params.num_lanes - 1 - v.lane) * c.cell_h
            body_cells = 1 if c.body_mode == "head" else max(1, v.length)
            col = (70, 160, 240) if v.veh_type == AV_VEH_TYPE else ((220, 150, 70) if v.veh_type == BUS_VEH_TYPE else (185, 185, 190))
            if v.controlled:
                col = (130, 220, 130)
            for b in range(body_cells):
                wx = (v.pos - b) * c.cell_w
                for sx in _iter_periodic_screen_x(world_x_px=wx, offset_x_px=offset, road_px_len=road_px, width_px=c.width, margin_px=c.cell_w):
                    rect = pg.Rect(int(sx), int(lane_y), c.cell_w, c.cell_h)
                    pg.draw.rect(self._surface, col, rect)
                    pg.draw.rect(self._surface, (15, 15, 15), rect, 1)
                    if v.changed_lane and c.draw_changed_lane_marker:
                        pg.draw.circle(self._surface, (255, 80, 120), (rect.right - 5, rect.top + 5), 4)
                    if c.draw_speed:
                        pip_n = max(0, min(v.vel, 6))
                        for p in range(pip_n):
                            px = rect.x + 4 + (p % 3) * 7
                            py = rect.y + rect.h - 6 - (p // 3) * 7
                            pg.draw.circle(self._surface, (10, 10, 10), (px, py), 2)
                    if c.draw_vehicle_ids:
                        t = self._font.render(str(vid), True, (10, 10, 10))
                        self._surface.blit(t, (rect.x + 2, rect.y + 2))
        if focus is not None:
            txt = self._font.render(f"focus id={focus.vehicle_id} v={focus.vel}", True, (240, 240, 240))
            if c.draw_focus_velocity:
                self._surface.blit(txt, (8, 8))
        if c.draw_step and step is not None:
            self._surface.blit(self._font.render(f"step={step}", True, (240, 240, 240)), (8, c.height - c.font_size - 8))
        if c.draw_interpolation_progress:
            w = max(1, int((c.width - 20) * alpha))
            pg.draw.rect(self._surface, (80, 200, 160), pg.Rect(10, c.height - 6, w, 4))
        arr = pg.surfarray.array3d(self._surface)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8, copy=False)
