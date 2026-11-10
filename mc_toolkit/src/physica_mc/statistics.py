"""Independent E2 crossed bootstrap for the Physica A Monte Carlo toolkit.

This module implements the frozen E2 statistical analysis plan directly; it is
not a generic i.i.d. CI helper.  It resamples whole training lines and whole
environment blocks in a crossed design, keeps fixed controllers (H/B1) fixed,
shares one environment draw per density across every fraction, line, method,
rule and baseline alias, and draws independent environment streams for distinct
densities.  Selection is an input and is never recomputed here.

The exact source SAP is recorded in :data:`SOURCE_SAP`:

* NumPy 2.1.2, ``Generator(PCG64(seed))``, integer indices ``int64``;
* seed = first 8 bytes, little-endian, of ``SHA256(UTF-8(str(master) + "|" + identity))``;
* identities ``env|rho=<two decimal places>`` and
  ``train|rule=<S0 or S_0.95>|architecture=<canonical mode ID>``;
* 49 999 resamples, percentile 95 %, linear quantiles;
* intervals are approximate, pointwise, descriptive, conditional on the frozen
  validation selection, and are neither simultaneous nor multiplicity-controlled.

Data are already reduced scalar contrasts: no raw per-episode time series is
accepted or allocated.  Bootstrap indices are produced in bounded chunks, and a
chunk boundary never changes a draw sequence.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType

import numpy as np


BOOTSTRAP_B = 49999
CONFIDENCE = 0.95
QUANTILE_METHOD = "linear"
MASTER_SEED = 2026102401
EXPECTED_NUMPY_VERSION = "2.1.2"
BIT_GENERATOR = "PCG64"
INDEX_DTYPE = "int64"
ENV_IDENTITY_TEMPLATE = "env|rho={density:.2f}"
TRAINING_IDENTITY_TEMPLATE = "train|rule={rule}|architecture={architecture}"
ARCHITECTURES = ("graph_ppo", "node_mlp_ppo", "grid_cnn_ppo")
RULES = ("S0", "S_0.93", "S_0.95", "S_0.97", "S_0.99")
THRESHOLDS = (0.93, 0.95, 0.97, 0.99)

INTERVAL_LABEL = (
    "approximate pointwise descriptive interval conditional on the frozen "
    "validation selection; not simultaneous and not multiplicity-controlled"
)
ENVIRONMENT_ONLY_LABEL = (
    "environment-only interval conditional on the single training line; "
    "between-training variation is not estimated; approximate pointwise "
    "descriptive interval conditional on the frozen validation selection; "
    "not simultaneous and not multiplicity-controlled"
)
UNDEFINED_LABEL = "contrast undefined; no interval is reported"
MISSING_LABEL = (
    "full-grid estimate/CI unavailable because the planned grid is incomplete; "
    "no available-case interval is substituted"
)

SOURCE_SAP = MappingProxyType({
    "resamples": BOOTSTRAP_B,
    "confidence": CONFIDENCE,
    "quantile_method": QUANTILE_METHOD,
    "numpy_version": EXPECTED_NUMPY_VERSION,
    "bit_generator": BIT_GENERATOR,
    "integer_index_dtype": INDEX_DTYPE,
    "master_seed": MASTER_SEED,
    "seed_derivation": "sha256(utf8(str(master) + '|' + identity)) first 8 bytes little-endian",
    "environment_identity": ENV_IDENTITY_TEMPLATE,
    "training_identity": TRAINING_IDENTITY_TEMPLATE,
    "environment_draw": "one 30-index stream per density, shared across fractions/lines/methods/rules",
    "training_draw_S0": "independent whole-line streams per architecture; shared across cells and contrasts",
    "training_draw_S_tau": "independent k validation-feasible whole-line streams per architecture; H fixed",
    "selection_resampled": False,
    "simultaneous_or_multiplicity_controlled": False,
})

Cell = tuple[float, float]


def rule_label(tau: float) -> str:
    """Canonical rule label, e.g. ``0.95 -> "S_0.95"``."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or tau not in THRESHOLDS:
        raise ValueError("tau must be one of THRESHOLDS")
    return f"S_{tau:.2f}"


def _finite(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{what} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite")
    return result


def _nonnegative(value, what):
    result = _finite(value, what)
    if result < 0:
        raise ValueError(f"{what} must be nonnegative")
    return result


def _positive_int(value, what):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{what} must be a positive integer")
    return int(value)


def stream_seed(master_seed: int, identity: str) -> int:
    """Keyed PCG64 seed from the frozen UTF-8 SHA-256 derivation."""
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    if not isinstance(identity, str) or not identity:
        raise ValueError("stream identity must be a non-empty string")
    digest = hashlib.sha256(f"{int(master_seed)}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def environment_identity(density: float) -> str:
    """Canonical environment stream identity with exactly two decimal places.

    A density that cannot be represented exactly at two decimals is rejected
    instead of silently sharing or renaming another density's stream.
    """
    density = _finite(density, "density")
    if not 0 < density < 1:
        raise ValueError("density must be in (0, 1)")
    text = f"{density:.2f}"
    if float(text) != density:
        raise ValueError("density must have an exact two-decimal canonical form")
    return ENV_IDENTITY_TEMPLATE.format(density=density)


def training_identity(rule: str, architecture: str) -> str:
    """Canonical training stream identity; rule/architecture are frozen enums."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"architecture must be one of {ARCHITECTURES}")
    return TRAINING_IDENTITY_TEMPLATE.format(rule=rule, architecture=architecture)


def _generator(identity: str, master_seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(stream_seed(master_seed, identity)))


def environment_index_chunks(
    density: float, n_env: int, *,
    master_seed: int = MASTER_SEED, B: int = BOOTSTRAP_B, chunk_size: int = 1024,
) -> Iterator[np.ndarray]:
    """Yield ``(chunk, n_env)`` int64 environment indices, with replacement.

    One replicate consumes exactly ``n_env`` draws, so the concatenated stream
    is independent of ``chunk_size``.  The stream belongs to the density
    identity alone and is therefore reused by every fraction/line/method/rule.
    """
    identity = environment_identity(density)
    n_env = _positive_int(n_env, "n_env")
    B = _positive_int(B, "B")
    chunk_size = _positive_int(chunk_size, "chunk_size")
    rng = _generator(identity, master_seed)
    for start in range(0, B, chunk_size):
        count = min(chunk_size, B - start)
        yield rng.integers(0, n_env, size=(count, n_env), dtype=np.int64)


def training_index_chunks(
    rule: str, architecture: str, n_lines: int, *,
    master_seed: int = MASTER_SEED, B: int = BOOTSTRAP_B, chunk_size: int = 1024,
) -> Iterator[np.ndarray]:
    """Yield ``(chunk, n_lines)`` int64 whole-line indices, with replacement.

    ``n_lines`` is the full per-architecture line set for S0 (3) or the k
    validation-feasible lines for S_tau.  Line identities are assumed to be in
    lexical order; :class:`LineSide` enforces that ordering.  The stream is
    keyed only by rule and architecture, so the same draws are reused across
    cells and contrasts (Graph draws are shared by Graph-Node and Graph-Grid).
    """
    identity = training_identity(rule, architecture)
    n_lines = _positive_int(n_lines, "n_lines")
    B = _positive_int(B, "B")
    chunk_size = _positive_int(chunk_size, "chunk_size")
    rng = _generator(identity, master_seed)
    for start in range(0, B, chunk_size):
        count = min(chunk_size, B - start)
        yield rng.integers(0, n_lines, size=(count, n_lines), dtype=np.int64)


def mean_sample_sd(values: Iterable[float]) -> tuple[float | None, float | None]:
    """Binary64 mean and sample SD (``n - 1``); both None for ``n == 0``, SD for ``n == 1``."""
    data = tuple(_finite(value, "value") for value in values)
    if not data:
        return None, None
    mean = math.fsum(data) / len(data)
    if len(data) < 2:
        return mean, None
    variance = math.fsum((value - mean) ** 2 for value in data) / (len(data) - 1)
    return mean, math.sqrt(variance)


def paired_ratios(
    numerators: Sequence[float], denominators: Sequence[float],
) -> tuple[float | None, ...]:
    """Exact per-road quotients; ``None`` where the denominator is zero.

    q_bg must be finite and nonnegative: a negative or non-finite denominator is
    an input/invariant failure, not an undefined constraint.  A zero denominator
    is the only ``None`` case, and no epsilon or zero substitution is used.
    """
    numerators = tuple(numerators)
    denominators = tuple(denominators)
    if len(numerators) != len(denominators):
        raise ValueError("numerator and denominator lengths differ")
    quotients = []
    for index, (numerator, denominator) in enumerate(zip(numerators, denominators)):
        numerator = _nonnegative(numerator, f"numerator[{index}]")
        denominator = _nonnegative(denominator, f"denominator[{index}]")
        quotients.append(None if denominator == 0.0 else numerator / denominator)
    return tuple(quotients)


def mean_paired_ratio(
    numerators: Sequence[float], denominators: Sequence[float],
) -> float | None:
    """Mean of per-road paired ratios; a cell with any zero B1 flow is undefined.

    This is deliberately not a ratio of means: the order is divide-then-average.
    """
    quotients = paired_ratios(numerators, denominators)
    if not quotients or any(value is None for value in quotients):
        return None
    return math.fsum(quotients) / len(quotients)


@dataclass(frozen=True)
class CellSummary:
    """One side's scalar cell: ``n`` planned rows, NaN rows are missing."""

    mean: float | None
    sample_sd: float | None
    n: int
    n_defined: int
    complete: bool


def summarize_cell(values) -> CellSummary:
    """Descriptive summary of one cell; never falls back to available cases."""
    array = np.asarray(values, dtype=np.float64)
    if np.isinf(array).any():
        raise ValueError("outcome values must not be infinite")
    flat = array.ravel()
    n = int(flat.size)
    finite = flat[np.isfinite(flat)]
    n_defined = int(finite.size)
    complete = n > 0 and n_defined == n
    if not complete:
        return CellSummary(None, None, n, n_defined, False)
    mean, sample_sd = mean_sample_sd(float(value) for value in finite)
    return CellSummary(mean, sample_sd, n, n_defined, True)


@dataclass(frozen=True)
class GridSpec:
    """Explicit planned grid: densities x fractions x ``n_env`` road blocks."""

    densities: tuple[float, ...]
    fractions: tuple[float, ...]
    n_env: int

    def __post_init__(self):
        densities = tuple(_finite(value, "density") for value in self.densities)
        fractions = tuple(_finite(value, "fraction") for value in self.fractions)
        if not densities or not fractions:
            raise ValueError("grid must declare at least one density and fraction")
        if any(not 0 < value < 1 for value in densities):
            raise ValueError("densities must be in (0, 1)")
        if any(not 0 < value <= 1 for value in fractions):
            raise ValueError("fractions must be in (0, 1]")
        densities = tuple(sorted(densities))
        fractions = tuple(sorted(fractions))
        if len(set(densities)) != len(densities) or len(set(fractions)) != len(fractions):
            raise ValueError("grid densities/fractions must be unique")
        object.__setattr__(self, "densities", densities)
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(self, "n_env", _positive_int(self.n_env, "n_env"))

    @property
    def cells(self) -> tuple[Cell, ...]:
        return tuple((density, fraction) for density in self.densities for fraction in self.fractions)


def equal_weight_grid_mean(cell_values: Mapping[Cell, float | None], grid: GridSpec) -> float | None:
    """Equal-weight mean over **all** planned cells, in canonical cell order.

    Returns None if any planned cell is absent or None; this helper never
    averages an available subset.  It is the reporting primitive for
    per-condition tables and for the bootstrap point estimate.
    """
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec")
    if not isinstance(cell_values, Mapping):
        raise ValueError("cell values must be a mapping keyed by cell")
    planned = set(grid.cells)
    try:
        extra = set(cell_values) - planned
    except TypeError as exc:
        raise ValueError("cell keys must be (density, fraction) tuples") from exc
    if extra:
        raise ValueError(f"cell values contain unplanned cells: {sorted(extra)!r}")
    values = []
    for cell in grid.cells:
        if cell not in cell_values or cell_values[cell] is None:
            return None
        values.append(_finite(cell_values[cell], f"cell value {cell!r}"))
    return math.fsum(values) / len(values)


@dataclass(frozen=True)
class LineSide:
    """Scalar outcomes of one architecture/rule over whole training lines.

    ``values`` maps each planned ``(density, fraction)`` cell to an array of
    shape ``(len(line_ids), n_env)``.  Line IDs are sorted lexically before
    integer sampling.  An absent key or a NaN entry means unavailable data (not
    an available-case fallback); an infinite value is invalid input.
    """

    architecture: str
    rule: str
    line_ids: tuple[str, ...]
    values: Mapping[Cell, np.ndarray]

    def __post_init__(self):
        training_identity(self.rule, self.architecture)
        line_ids = tuple(self.line_ids)
        if len(line_ids) > 3:
            raise ValueError("E2 has at most three training lines per architecture")
        if any(not isinstance(line_id, str) or not line_id for line_id in line_ids):
            raise ValueError("line IDs must be non-empty strings")
        if len(set(line_ids)) != len(line_ids):
            raise ValueError("line IDs must be unique")
        if not isinstance(self.values, Mapping):
            raise ValueError("line side values must be a mapping keyed by cell")
        object.__setattr__(self, "line_ids", line_ids)


@dataclass(frozen=True)
class FixedSide:
    """A fixed controller (H or B1); it is not cloned into independent lines.

    ``values`` maps each cell to an array of shape ``(n_env,)``.  Because there
    is no line identity, no training stream is consumed and the same environment
    draws are reused for this comparator everywhere (including fraction aliases
    of one physical B1 block).
    """

    label: str
    values: Mapping[Cell, np.ndarray]

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("fixed side needs a non-empty label")
        if not isinstance(self.values, Mapping):
            raise ValueError("fixed side values must be a mapping keyed by cell")


@dataclass(frozen=True)
class Contrast:
    """One frozen estimand, e.g. Graph S0 - Node S0 or Graph S_0.95 - H_0.95."""

    label: str
    left: LineSide
    right: LineSide | FixedSide

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("contrast needs a non-empty label")
        if not isinstance(self.left, LineSide):
            raise ValueError("contrast left side must be a line side")
        if not isinstance(self.right, (LineSide, FixedSide)):
            raise ValueError("contrast right side must be a line or fixed side")


@dataclass(frozen=True)
class CellContrastSummary:
    """Observed per-condition contrast (left minus right) and its coverage."""

    cell: Cell
    estimate: float | None
    complete: bool
    left: CellSummary
    right: CellSummary


@dataclass(frozen=True)
class BootstrapResult:
    """Descriptive contrast at one equal-weight grid plus pointwise percentile CI.

    ``point`` is the observed equal-weight grid mean (canonical cell order,
    binary64 ``math.fsum``); ``ci_low``/``ci_high`` are linear percentiles of the
    crossed bootstrap distribution.  When the planned grid is incomplete,
    ``point`` and both interval bounds are None and no available-case bootstrap
    is run; complete individual cells remain readable in ``cells``.  A contrast
    with zero training lines on either side is ``undefined``.  ``replicates`` is
    the ``(B,)`` bootstrap distribution, or None when no resampling was run.
    """

    label: str
    status: str  # complete | missing_planned_cells | undefined_no_training_lines
    reason: str
    point: float | None
    ci_low: float | None
    ci_high: float | None
    n_cells_planned: int
    n_cells_complete: int
    n_lines_left: int
    n_lines_right: int | None
    line_ids_left: tuple[str, ...]
    line_ids_right: tuple[str, ...] | None
    n_env: int
    confidence: float
    quantile_method: str
    interval_kind: str
    interval_label: str
    master_seed: int
    resamples: int
    numpy_version: str
    cells: Mapping[Cell, CellContrastSummary]
    replicates: np.ndarray | None


@dataclass(frozen=True)
class _PreparedSide:
    kind: str  # "line" or "fixed"
    identity: str
    n_lines: int | None
    architecture: str
    rule: str
    arrays: dict[Cell, np.ndarray | None]
    line_ids: tuple[str, ...] = ()


def _checked_values(values: Mapping, *, n_rows: int | None, n_env: int, cells: tuple[Cell, ...]):
    if not isinstance(values, Mapping):
        raise ValueError("side values must be a mapping keyed by (density, fraction)")
    planned = set(cells)
    try:
        extra = set(values) - planned
    except TypeError as exc:
        raise ValueError("cell keys must be (density, fraction) tuples") from exc
    if extra:
        raise ValueError(f"values contain unplanned cells: {sorted(extra)!r}")
    expected_shape = (n_env,) if n_rows is None else (n_rows, n_env)
    checked: dict[Cell, np.ndarray | None] = {}
    for cell in cells:
        if cell not in values:
            checked[cell] = None
            continue
        try:
            array = np.asarray(values[cell], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cell {cell!r} must be a numeric array") from exc
        if array.shape != expected_shape:
            raise ValueError(f"cell {cell!r} has shape {array.shape}, expected {expected_shape}")
        if np.isinf(array).any():
            raise ValueError(f"cell {cell!r} contains an infinite outcome; NaN marks missing")
        checked[cell] = array
    return checked


def _prepare_side(side: LineSide | FixedSide, grid: GridSpec) -> _PreparedSide:
    if isinstance(side, LineSide):
        n_lines = len(side.line_ids)
        arrays = _checked_values(side.values, n_rows=n_lines, n_env=grid.n_env, cells=grid.cells)
        order = sorted(range(n_lines), key=lambda index: side.line_ids[index])
        if order != list(range(n_lines)):
            arrays = {
                cell: None if array is None else array[order]
                for cell, array in arrays.items()
            }
        line_ids = tuple(side.line_ids[index] for index in order)
        identity = training_identity(side.rule, side.architecture)
        return _PreparedSide("line", identity, n_lines, side.architecture, side.rule, arrays, line_ids)
    arrays = _checked_values(side.values, n_rows=None, n_env=grid.n_env, cells=grid.cells)
    return _PreparedSide("fixed", side.label, None, "", "", arrays)


def _side_summary(array: np.ndarray | None) -> CellSummary:
    if array is None:
        return CellSummary(None, None, 0, 0, False)
    return summarize_cell(array)


def _cell_summary(cell: Cell, left: _PreparedSide, right: _PreparedSide) -> CellContrastSummary:
    left_summary = _side_summary(left.arrays[cell])
    right_summary = _side_summary(right.arrays[cell])
    complete = left_summary.complete and right_summary.complete
    estimate = None
    if complete:
        estimate = float(left_summary.mean - right_summary.mean)
    return CellContrastSummary(cell, estimate, complete, left_summary, right_summary)


def _replicate_mean(side: _PreparedSide, cell: Cell, env_idx: np.ndarray, line_idx: Mapping[str, np.ndarray]) -> np.ndarray:
    array = side.arrays[cell]
    if side.kind == "fixed":
        return array[env_idx].mean(axis=1)
    rows = line_idx[side.identity]
    drawn = array[rows[:, :, None], env_idx[:, None, :]]
    return drawn.mean(axis=(1, 2))


def _interval_kind(left: _PreparedSide, right: _PreparedSide) -> str:
    if left.n_lines is not None and left.n_lines < 2:
        return "environment_only_conditional_on_single_training_line"
    if right.kind == "line" and right.n_lines is not None and right.n_lines < 2:
        return "environment_only_conditional_on_single_training_line"
    return "crossed_approximate"


def _result(
    *, label: str, status: str, reason: str, left: _PreparedSide, right: _PreparedSide,
    grid: GridSpec, cells: Mapping[Cell, CellContrastSummary], point, ci_low, ci_high,
    confidence: float, master_seed: int, B: int, replicates: np.ndarray | None,
    interval_kind: str, interval_label: str,
) -> BootstrapResult:
    return BootstrapResult(
        label=label,
        status=status,
        reason=reason,
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_cells_planned=len(grid.cells),
        n_cells_complete=sum(cell.complete for cell in cells.values()),
        n_lines_left=left.n_lines if left.n_lines is not None else 0,
        n_lines_right=right.n_lines,
        line_ids_left=tuple(left.line_ids),
        line_ids_right=tuple(right.line_ids) if right.kind == "line" else None,
        n_env=grid.n_env,
        confidence=confidence,
        quantile_method=QUANTILE_METHOD,
        interval_kind=interval_kind,
        interval_label=interval_label,
        master_seed=int(master_seed),
        resamples=int(B),
        numpy_version=np.__version__,
        cells=dict(cells),
        replicates=replicates,
    )


def crossed_bootstrap(
    contrasts: Iterable[Contrast],
    *,
    grid: GridSpec,
    master_seed: int = MASTER_SEED,
    B: int = BOOTSTRAP_B,
    confidence: float = CONFIDENCE,
    chunk_size: int = 512,
) -> dict[str, BootstrapResult]:
    """Run the frozen crossed bootstrap for a set of contrasts on one grid.

    Environment indices are drawn once per density and chunk, then reused for
    every contrast/fraction/line/rule and for fixed comparators.  Whole-line
    indices are drawn once per ``(rule, architecture)`` identity and chunk and
    reused across cells and contrasts; this is what makes Graph draws common to
    Graph-Node and Graph-Grid.  Fixed H/B1 sides consume no line stream, so they
    are never cloned into k pseudo-independent controllers.

    The full-grid estimate/CI exists only when every planned cell of every
    contrast side is present and finite (no NaN).  Otherwise the result is
    ``missing_planned_cells`` with None estimate/CI, while complete cells remain
    visible.  A zero-line side is ``undefined_no_training_lines``.  Selection is
    not resampled, and equal-weight grid means use canonical cell order.
    """
    if not isinstance(grid, GridSpec):
        raise ValueError("grid must be a GridSpec")
    if isinstance(master_seed, (bool, np.bool_)) or not isinstance(master_seed, (int, np.integer)) or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    B = _positive_int(B, "B")
    chunk_size = _positive_int(chunk_size, "chunk_size")
    confidence = _finite(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    contrasts = tuple(contrasts)
    if not contrasts:
        raise ValueError("at least one contrast is required")
    labels = [contrast.label for contrast in contrasts]
    if len(set(labels)) != len(labels):
        raise ValueError("contrast labels must be unique")

    prepared: dict[str, tuple[_PreparedSide, _PreparedSide]] = {}
    for contrast in contrasts:
        left = _prepare_side(contrast.left, grid)
        right = _prepare_side(contrast.right, grid)
        prepared[contrast.label] = (left, right)

    cells_by_label: dict[str, dict[Cell, CellContrastSummary]] = {}
    for contrast in contrasts:
        left, right = prepared[contrast.label]
        cells_by_label[contrast.label] = {
            cell: _cell_summary(cell, left, right) for cell in grid.cells
        }

    results: dict[str, BootstrapResult] = {}
    computable: list[str] = []
    undefined: set[str] = set()
    for contrast in contrasts:
        left, right = prepared[contrast.label]
        zero_lines = (left.n_lines == 0) or (right.kind == "line" and right.n_lines == 0)
        if zero_lines:
            undefined.add(contrast.label)
            continue
        if not all(cell.complete for cell in cells_by_label[contrast.label].values()):
            continue
        computable.append(contrast.label)

    if computable:
        streams_needed: dict[str, tuple[str, str, int]] = {}
        for label in computable:
            for side in prepared[label]:
                if side.kind != "line":
                    continue
                previous = streams_needed.get(side.identity)
                if previous is not None and previous[2] != side.n_lines:
                    raise ValueError(
                        f"training identity {side.identity!r} used with inconsistent line counts"
                    )
                streams_needed[side.identity] = (side.rule, side.architecture, side.n_lines)
        env_streams = {
            density: environment_index_chunks(
                density, grid.n_env, master_seed=int(master_seed), B=B, chunk_size=chunk_size,
            )
            for density in grid.densities
        }
        train_streams = {
            identity: training_index_chunks(
                rule, architecture, n_lines,
                master_seed=int(master_seed), B=B, chunk_size=chunk_size,
            )
            for identity, (rule, architecture, n_lines) in streams_needed.items()
        }
        replicates = {label: np.empty(B, dtype=np.float64) for label in computable}
        for start in range(0, B, chunk_size):
            count = min(chunk_size, B - start)
            env_chunks = {density: next(stream) for density, stream in env_streams.items()}
            line_chunks = {identity: next(stream) for identity, stream in train_streams.items()}
            for label in computable:
                left, right = prepared[label]
                total = np.zeros(count, dtype=np.float64)
                for cell in grid.cells:
                    env_idx = env_chunks[cell[0]]
                    total += (
                        _replicate_mean(left, cell, env_idx, line_chunks)
                        - _replicate_mean(right, cell, env_idx, line_chunks)
                    )
                replicates[label][start:start + count] = total / len(grid.cells)
        for label in computable:
            values = replicates[label]
            quantiles = np.quantile(
                values,
                [(1.0 - confidence) / 2.0, 1.0 - (1.0 - confidence) / 2.0],
                method=QUANTILE_METHOD,
            )
            left, right = prepared[label]
            point = equal_weight_grid_mean(
                {cell: cells_by_label[label][cell].estimate for cell in grid.cells}, grid,
            )
            interval_kind = _interval_kind(left, right)
            # k=1 must carry its own limitation even though the shared
            # pointwise/frozen-selection/multiplicity caveats still apply.
            interval_label = (
                ENVIRONMENT_ONLY_LABEL
                if interval_kind == "environment_only_conditional_on_single_training_line"
                else INTERVAL_LABEL
            )
            results[label] = _result(
                label=label, status="complete", reason="",
                left=left, right=right, grid=grid, cells=cells_by_label[label],
                point=point, ci_low=float(quantiles[0]), ci_high=float(quantiles[1]),
                confidence=confidence, master_seed=int(master_seed), B=B,
                replicates=values,
                interval_kind=interval_kind, interval_label=interval_label,
            )

    for label in undefined:
        left, right = prepared[label]
        results[label] = _result(
            label=label, status="undefined_no_training_lines",
            reason="k=0 training lines; this contrast is undefined and has no comparator interval",
            left=left, right=right, grid=grid, cells=cells_by_label[label],
            point=None, ci_low=None, ci_high=None,
            confidence=confidence, master_seed=int(master_seed), B=B,
            replicates=None, interval_kind="undefined", interval_label=UNDEFINED_LABEL,
        )

    for contrast in contrasts:
        label = contrast.label
        if label in results:
            continue
        left, right = prepared[label]
        incomplete = sum(not cell.complete for cell in cells_by_label[label].values())
        results[label] = _result(
            label=label, status="missing_planned_cells",
            reason=(
                f"{incomplete} of {len(grid.cells)} planned cells lack complete finite scalar data; "
                "full-grid estimate/CI undefined (no available-case bootstrap)"
            ),
            left=left, right=right, grid=grid, cells=cells_by_label[label],
            point=None, ci_low=None, ci_high=None,
            confidence=confidence, master_seed=int(master_seed), B=B,
            replicates=None, interval_kind=_interval_kind(left, right),
            interval_label=MISSING_LABEL,
        )
    return results
