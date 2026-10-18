"""Compare frozen-L1 L3 R0/R2 validation checkpoints across residual training seeds.

No test access. Reconstruct best.pt selection from per-run history.jsonl;
compare the *same* fixed validation windows (seed 900001) and the same L1.
"""

import argparse
import json
import statistics
from pathlib import Path


def best_record(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError(f'Empty history: {path}')
    if len({r['variant'] for r in records}) != 1:
        raise ValueError(f'Multiple variants in {path}')
    return min(records, key=lambda r: r['refined_dz_mae_mm'])


def compare(root, seeds):
    rows = []
    for seed in seeds:
        paths = {arm: root / f'l3_{arm}_s{seed}' for arm in ('r0', 'r2')}
        for arm, folder in paths.items():
            if not (folder / 'best.pt').is_file():
                raise FileNotFoundError(f'Missing selected checkpoint: {folder / "best.pt"}')
        r0, r2 = [best_record(paths[arm] / 'history.jsonl') for arm in ('r0', 'r2')]
        if r0['variant'] != 'r0' or r2['variant'] != 'r2':
            raise ValueError(f'Wrong variant found for seed {seed}')
        for field in ('primary_pixels', 'full_valid_pixels', 'base_dz_mae_mm',
                      'full_base_dz_mae_mm'):
            if abs(r0[field] - r2[field]) > 1e-7:
                raise ValueError(f'Unpaired validation support or baseline for seed {seed}: {field}')
        if set(r0['by_ct']) != set(r2['by_ct']):
            raise ValueError(f'CT mismatch for seed {seed}')
        per_ct = {}
        for ct in r0['by_ct']:
            c0, c2 = r0['by_ct'][ct], r2['by_ct'][ct]
            if c0['pixels'] != c2['pixels']:
                raise ValueError(f'Pixel support mismatch for seed {seed}, case {ct}')
            per_ct[ct] = {
                'pixels': c0['pixels'],
                'r0_mae_mm': c0['refined_dz_mae_mm'],
                'r2_mae_mm': c2['refined_dz_mae_mm'],
                'r2_minus_r0_mm': c2['refined_dz_mae_mm'] - c0['refined_dz_mae_mm']}
        rows.append({
            'seed': seed, 'r0_step': r0['step'], 'r2_step': r2['step'],
            'baseline_l1_mm': r0['base_dz_mae_mm'],
            'r0_mae_mm': r0['refined_dz_mae_mm'],
            'r2_mae_mm': r2['refined_dz_mae_mm'],
            'r2_minus_r0_mm': r2['refined_dz_mae_mm'] - r0['refined_dz_mae_mm'],
            'r2_improvement_pct': 100 * (r0['refined_dz_mae_mm'] - r2['refined_dz_mae_mm']) / r0['refined_dz_mae_mm'],
            'r0_full_mae_mm': r0['full_refined_dz_mae_mm'],
            'r2_full_mae_mm': r2['full_refined_dz_mae_mm'],
            'by_ct': per_ct})
    improvements = [r['r2_improvement_pct'] for r in rows]
    summary = {
        'primary_metric': 'VAL |dz| MAE (<=3 mm), lower is better',
        'seeds': seeds, 'n_independent_validation_ct': len(rows[0]['by_ct']),
        'mean_r2_improvement_pct': statistics.mean(improvements),
        'sample_sd_r2_improvement_pct': statistics.stdev(improvements) if len(rows) > 1 else None,
        'r2_better_seeds': sum(r['r2_minus_r0_mm'] < 0 for r in rows),
        'r2_better_in_every_ct_of_every_seed': all(
            x['r2_minus_r0_mm'] < 0 for row in rows for x in row['by_ct'].values()),
        'note': 'Repeated seeds share the same validation CTs: not independent anatomical replications.',
        'rows': rows}
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, default=Path('runs/local'))
    p.add_argument('--seeds', type=int, nargs='+', default=[17, 42, 97])
    p.add_argument('--out', type=Path, default=Path('runs/local/l3_seed_confirmation.json'))
    args = p.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        p.error('Duplicate seeds')
    report = compare(args.runs_root, args.seeds)
    print('seed    R0 step / MAE   R2 step / MAE    R2-R0 (mm)   improvement')
    for r in report['rows']:
        print(f"{r['seed']:>4}   {r['r0_step']:>4} / {r['r0_mae_mm']:.5f}  "
              f"{r['r2_step']:>4} / {r['r2_mae_mm']:.5f}  "
              f"{r['r2_minus_r0_mm']:+.5f}       {r['r2_improvement_pct']:+.2f}%")
    print(f"Mean relative improvement: {report['mean_r2_improvement_pct']:+.2f}% "
          f"(SD {report['sample_sd_r2_improvement_pct']})")
    print(f"R2 better in every CT and seed: {report['r2_better_in_every_ct_of_every_seed']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Report: {args.out}; TEST untouched')


if __name__ == '__main__':
    main()
