from __future__ import annotations

import argparse

import numpy as np

from snfs_traffic.backends import get_backend
from snfs_traffic.core import SimulationParams, validate_runtime_invariants
from snfs_traffic.scenarios import VehicleMix, make_uniform_random_state
from snfs_traffic.topology import RingTopology
from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer, VideoWriter


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="episode.mp4")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num-lanes", type=int, default=3)
    p.add_argument("--road-length", type=int, default=120)
    p.add_argument("--density", type=float, default=0.20)
    p.add_argument("--av-fraction", type=float, default=0.0)
    p.add_argument("--controlled-fraction", type=float, default=0.03)
    p.add_argument("--bus-fraction", type=float, default=0.0)
    p.add_argument("--bus-length", type=int, default=3)
    p.add_argument("--backend", choices=["auto", "reference", "optimized"], default="auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rng-seed", type=int, default=123)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--cell-w", type=int, default=28)
    p.add_argument("--cell-h", type=int, default=54)
    p.add_argument("--interpolation-frames", type=int, default=8)
    p.add_argument("--fps", type=int, default=48)
    p.add_argument("--camera", choices=["follow", "fixed"], default="follow")
    p.add_argument("--follow", choices=["first-controlled", "first-alive", "none"], default="first-controlled")
    p.add_argument("--body-mode", choices=["head", "state_length"], default="head")
    p.add_argument("--draw-cell-numbers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--draw-speed", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    params = SimulationParams(num_lanes=args.num_lanes, road_length=args.road_length)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    mix = VehicleMix(
        av_fraction=args.av_fraction,
        controlled_fraction=args.controlled_fraction,
        bus_fraction=args.bus_fraction,
        bus_length=args.bus_length,
    )
    state = make_uniform_random_state(
        num_lanes=params.num_lanes,
        road_length=params.road_length,
        density=args.density,
        seed=args.seed,
        vehicle_mix=mix,
    )
    backend = get_backend(args.backend)
    rng = np.random.default_rng(args.rng_seed)
    follow = None if args.follow == "none" else args.follow
    renderer = RoadRenderer(
        params=params,
        topology=topology,
        config=RoadRenderConfig(
            width=args.width,
            height=args.height,
            cell_w=args.cell_w,
            cell_h=args.cell_h,
            interpolation_frames=args.interpolation_frames,
            camera=args.camera,
            follow=follow,
            body_mode=args.body_mode,
            draw_cell_numbers=args.draw_cell_numbers,
            draw_speed=args.draw_speed,
        ),
    )
    renderer.reset(state)
    with VideoWriter(args.output, fps=args.fps) as video:
        for step in range(args.steps):
            nxt = backend.step(state, params, topology, rng)
            validate_runtime_invariants(nxt, params, topology)
            video.write_many(renderer.render_step(nxt, step=step + 1))
            state = nxt
    print(
        f"Wrote {args.output} with backend={args.backend}, "
        f"steps={args.steps}, fps={args.fps}"
    )


if __name__ == "__main__":
    main()
