from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snfs_traffic.core import SimulationParams, validate_runtime_invariants
from snfs_traffic.scenarios import VehicleMix
from snfs_traffic.simulator import ScenarioConfig, TrafficSimulator
from snfs_traffic.topology import RingTopology
from snfs_traffic.visualization import RoadRenderConfig, RoadRenderer, VideoWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate a 4-lane periodic ring road and write an MP4 video."
    )
    parser.add_argument("--output", default="ring_4lane.mp4", help="Output MP4 path.")
    parser.add_argument("--steps", type=int, default=300, help="Simulation steps to render.")
    parser.add_argument("--road-length", type=int, default=1000, help="Ring length in cells.")
    parser.add_argument("--density", type=float, default=0.12, help="Vehicle density in [0, 1].")
    parser.add_argument(
        "--backend",
        choices=["auto", "reference", "optimized"],
        default="auto",
        help="Simulation backend. 'auto' uses optimized when Numba kernels are available.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Scenario seed.")
    parser.add_argument("--rng-seed", type=int, default=123, help="Dynamics RNG seed.")

    # Vehicle mix. Defaults keep all vehicles unit-length, which is the safest/fastest path.
    parser.add_argument("--av-fraction", type=float, default=0.0)
    parser.add_argument(
        "--controlled-fraction",
        type=float,
        default=0.03,
        help="Fraction of vehicles marked controlled/green in the visualization."
             " With no explicit action provider they still move through the normal simulator step.",
    )
    parser.add_argument("--bus-fraction", type=float, default=0.0)
    parser.add_argument("--bus-length", type=int, default=3)

    # Video/rendering.
    parser.add_argument("--fps", type=int, default=48)
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        default=8,
        help="Rendered frames per CA step. Video duration ~= steps * interpolation_frames / fps.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--cell-w", type=int, default=28)
    parser.add_argument("--cell-h", type=int, default=54)
    parser.add_argument(
        "--camera",
        choices=["follow", "fixed"],
        default="follow",
        help="For road_length=1000, follow is usually more readable than showing a fixed small segment.",
    )
    parser.add_argument(
        "--follow",
        choices=["first-controlled", "first-alive", "none"],
        default="first-controlled",
        help="Vehicle to follow when --camera=follow.",
    )
    parser.add_argument("--fixed-start-cell", type=float, default=0.0)
    parser.add_argument("--draw-cell-numbers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--draw-speed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-vehicle-ids", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--body-mode",
        choices=["head", "state_length"],
        default="head",
        help="Use state_length if you enable buses and want to draw full bodies.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if not 0.0 <= args.density <= 1.0:
        raise ValueError("--density must be in [0, 1]")
    for name in ("av_fraction", "controlled_fraction", "bus_fraction"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.av_fraction + args.controlled_fraction + args.bus_fraction > 1.0:
        raise ValueError("vehicle fractions must sum to <= 1")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    params = SimulationParams(num_lanes=4, road_length=args.road_length)
    topology = RingTopology(num_lanes=params.num_lanes, length=params.road_length)
    mix = VehicleMix(
        av_fraction=args.av_fraction,
        controlled_fraction=args.controlled_fraction,
        bus_fraction=args.bus_fraction,
        bus_length=args.bus_length,
    )

    sim = TrafficSimulator(
        params=params,
        topology=topology,
        backend=args.backend,
        rng_seed=args.rng_seed,
        scenario=ScenarioConfig(
            density=args.density,
            seed=args.seed,
            vehicle_mix=mix,
        ),
        validate=True,
        require_all_controlled_actions=False,
    )

    state = sim.reset(seed=args.seed, rng_seed=args.rng_seed)
    validate_runtime_invariants(state, params, topology)

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
            fixed_start_cell=args.fixed_start_cell,
            body_mode=args.body_mode,
            draw_cell_numbers=args.draw_cell_numbers,
            draw_speed=args.draw_speed,
            draw_vehicle_ids=args.draw_vehicle_ids,
        ),
    )
    renderer.reset(state)

    n_vehicles = int(np.count_nonzero(state.alive))
    n_controlled = int(np.count_nonzero(state.controlled & state.alive))
    print(
        "start: "
        f"lanes={params.num_lanes}, road_length={params.road_length}, "
        f"vehicles={n_vehicles}, controlled={n_controlled}, "
        f"density={args.density:.3f}, backend={sim.backend_name}"
    )

    with VideoWriter(output, fps=args.fps) as video:
        # A few static frames make the first state visible before movement starts.
        video.write_many(renderer.render_step(state, step=0))
        for step in range(1, args.steps + 1):
            state = sim.step()
            validate_runtime_invariants(state, params, topology)
            video.write_many(renderer.render_step(state, step=step))

            if step == 1 or step % 50 == 0 or step == args.steps:
                print(f"rendered step {step}/{args.steps}")

    print(
        f"wrote {output} "
        f"({args.steps} steps, {args.interpolation_frames} frames/step, fps={args.fps})"
    )


if __name__ == "__main__":
    main()



"""
python research/ring_4lane_video.py \
  --output tmp/videos/ring_4lane.mp4 \
  --steps 300 \
  --road-length 500 \
  --density 0.1 \
  --backend auto --draw-vehicle-ids

"""