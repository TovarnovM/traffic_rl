"""Redraw current Physica A publication figures and export table-only sources."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from inputs import INPUTS, RENDER, verify_inputs
from build_global_assets import build as global_build, require
from build_local_assets import build as local_build
from export_tables import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Explicit NEW directory; existing paths refused')
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f'output already exists: {output}')
    verify_inputs(INPUTS)
    require(RENDER.is_dir(), f'missing release shared renderer: {RENDER}')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    definitions = json.loads((INPUTS / 'source_generators.json').read_text())
    for name in ('render.py', 'style.py'):
        require(hashlib.sha256((RENDER / name).read_bytes()).hexdigest() == definitions[name]['sha256'],
                f'shared publication definition changed: {name}')
    output.mkdir()
    global_build(INPUTS, output / 'global')
    local_build(INPUTS, output / 'local')
    sys.path.insert(0, str(RENDER))
    import render
    path = next((INPUTS / 'validation').glob('*1.0*candidate*validation-V*.json'))
    meta = json.loads(path.read_text())
    with path.with_suffix('.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == meta['plotted_rows'] == 121 and meta['mode'] == 'completed_validation',
            'wrong main validation input')
    # The included final sidecar already has categorical fixed-control x values.
    # Restore its recorded source coordinates so the original renderer applies
    # the same display-only projection once, inside the single validation page.
    rows = [dict(r, x=r['source_x'], x_label=r['source_x_label'])
            if r.get('source_x_label') else r for r in rows]
    validation = output / 'validation'
    validation.mkdir()
    with render._plt().rc_context(render.RC):
        artifacts = render._curve_pages(rows, validation, meta['source_manifest'], ('png', 'pdf'), 6)
    require(len(artifacts) == 1 and artifacts[0]['id'] == meta['id'], 'main validation identity changed')
    figures = output / 'figures'
    figures.mkdir()
    mapping = json.loads((INPUTS / 'figure_mapping.json').read_text())
    require(len(mapping) == 37, 'expected 37 numerical figures')
    for entry in mapping:
        for suffix in ('.pdf', '.png', '.csv', '.json'):
            source = output / entry['generator_group'] / Path(entry['generated_name']).with_suffix(suffix)
            destination = figures / Path(entry['publication_name']).with_suffix(suffix)
            shutil.copyfile(source, destination)
    shutil.copyfile(INPUTS / 'masters/method_schematic.pdf', figures / 'method_schematic.pdf')
    export(INPUTS, output / 'tables')
    (output / 'COMPLETE.json').write_text(json.dumps(dict(
        numerical_figures=37, vector_masters_copied=1, publication_figures=38,
        tables=10, additional_tables='42-domain blocks',
        schematic='Original vector PDF master copied, not numerically redrawn. '
                  'The generic conceptual renderer is not asserted to match this master.',
        source='Included derived rows; no inference, raw archives, caches or private worktrees.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
