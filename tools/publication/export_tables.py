"""Export the ten current article/SI tables and the 42-domain blocks.

Only formatting and exact CSV projection; no inference or statistical reductions.
Outputs are table-only TeX fragments and exact CSVs, without an article/class/bib.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

from build_global_assets import COUNTS, latex, require, select, table
from inputs import INPUTS, verify_inputs


def read_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def export(source, output):
    verify_inputs(source)
    require(not output.exists(), f'output already exists: {output}')
    output.mkdir()
    data = source / 'tables'
    for path in data.glob('*.csv'):
        shutil.copyfile(path, output / path.name)
    for path in (source / 'historical').iterdir():
        shutil.copyfile(path, output / ('historical_' + path.name))
    table(output / 'revision_effects.tex', read_rows(data / 'revision_effects.csv'),
          ['rule', 'method', 'value', 'ci_low', 'ci_high', 'k', 'support'])
    domains = read_rows(data / 'revision_all42domains.csv')
    require(len(domains) == 42, 'expected 42 domain effects')
    parts = []
    for panel, cohort, rule in sorted({(r['panel'], r['cohort'], r['rule']) for r in domains}):
        subset = select(domains, panel=panel, cohort=cohort, rule=rule)
        path = output / 'revision_all42domains.tex'
        table(path, subset, ['method', 'x', 'value', 'ci_low', 'ci_high', 'k', 'support'])
        parts.append(r'\par\noindent ' + latex(panel + ' / scale ' + cohort + ' / ' + rule)
                     + '\n' + r'\par\nopagebreak[4]' + '\n' + path.read_text())
    (output / 'revision_all42domains.tex').write_text('\n\n'.join(parts))
    audit = read_rows(data / 'revision_background_audit.csv')
    wide = []
    for rule, method, line in dict.fromkeys((r['rule'], r['method'], r['training_line']) for r in audit):
        group = select(audit, rule=rule, method=method, training_line=line)
        values = {}
        for metric in COUNTS:
            found = select(group, metric=metric)
            require(len(found) == 1, 'duplicate/missing audit count')
            values[metric] = found[0]['value']
        wide.append(dict(rule=rule, method=method, training_line=line, **values))
    table(output / 'revision_background_audit.tex', wide, ['rule', 'method', 'training_line', *COUNTS])
    pairs = read_rows(data / 'revision_recorded_pairs.csv')
    require(len(pairs) == 12, 'expected 12 frozen pairs')
    template = (data / 'revision_recorded_pairs.tex').read_text()
    start = template.index('\n0.10')
    end = template.index(r'\hline\end{tabular}', start)
    lines = [f'{float(r["rho"]):.2f} & {float(r["f"]):.2f} & {latex(r["training_line"])} & '
             + r['replicate'] + f' & {float(r["paired_pv_effect"]):.4f}' + r' \\' for r in pairs]
    (output / 'revision_recorded_pairs.tex').write_text(template[:start] + '\n' + '\n'.join(lines) + '\n' + template[end:])
    shutil.copyfile(data / 'checkpoint_selection_rows.tex', output / 'checkpoint_selection_rows.tex')

    # Historical SI Table 5: format stored decisions, never rerun multiplicity tests.
    historical = read_rows(source / 'historical/multiplicity_summary.csv')
    names = {'graph_ppo': 'Graph', 'node_mlp_ppo': 'NodeMLP', 'grid_cnn_ppo': 'GridCNN',
             'baseline2_matched': 'B2', 'left_if_safe': 'left-if-safe'}
    template = (data / 'revision_supplement_table_04.tex').read_text()
    start = template.index(r'\midrule') + len(r'\midrule')
    end = template.index(r'\bottomrule', start)
    body = []
    for row in historical:
        contrast = names[row['treatment_mode']] + '--' + names[row['comparison_mode']]
        counts = ['/'.join(row[f'{kind}_{sign}'] for sign in ('positive', 'negative', 'nonrejected'))
                  for kind in ('raw', 'bh', 'by')]
        body.append(contrast + ' &' + row['family_m'] + '&' + '&'.join(counts) + r'\\')
    require(len(body) == 8, 'expected eight historical contrasts')
    historical_tex = template[:start] + '\n' + '\n'.join(body) + '\n' + template[end:]
    tables = []
    for path in sorted(data.glob('*_table_*.tex')):
        text = historical_tex if path.name == 'revision_supplement_table_04.tex' else path.read_text()
        text = re.sub(r'\\input\{tables/([^}]+)\}', lambda m: (output / m[1]).read_text(), text)
        (output / path.name).write_text(text)
        tables.append(path.name)
    tables.insert(8, 'revision_recorded_pairs.tex')
    require(len(tables) == 10, 'expected five main and five SI tables')
    (output / 'TABLES.json').write_text(json.dumps(dict(
        tables=tables, unnumbered='revision_all42domains.tex',
        historical='revision_supplement_table_04.tex is SI Table 5; the recorded-pairs table precedes it',
        format='Table-only TeX, original captions/labels/macros; no journal class, article prose or bibliography. '
               'The host TeX document supplies booktabs, column P, PV, eqref and any original equation references.'), indent=2) + '\n')
    shutil.copyfile(data / 'extraction_manifest.json', output / 'extraction_manifest.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    export(INPUTS, args.output.resolve())


if __name__ == '__main__':
    main()
