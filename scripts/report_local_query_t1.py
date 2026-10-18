"""Aggregate matched T1 Q0/Q1/Q2 best-validation reports, without test access."""
import argparse
import json
from pathlib import Path

ARMS = ('q0','q1','q2')
GAPS = ('1','2','4')


def summarize(root, seed):
    reports = {}
    for arm in ARMS:
        file = root / f't1_{arm}_s{seed}' / 'best_val.json'
        if not file.exists():
            raise FileNotFoundError(f'Complete {arm} training first: {file}')
        reports[arm] = json.loads(file.read_text())
        if reports[arm]['config']['variant'] != arm:
            raise ValueError(f'Unexpected variant in {file}')
        if reports[arm]['split'] != 'val':
            raise ValueError('Refusing a non-validation report')
    common = ('data','l1_hash','queries','val_clips_per_sweep','overlap_mm','training_gaps')
    for field in common:
        if len({json.dumps(r['config'][field], sort_keys=True) for r in reports.values()}) != 1:
            raise ValueError(f'T1 arms have mismatched {field}')
    cases_by_arm = {arm:set(reports[arm]['gaps']['1']['by_ct']) for arm in ARMS}
    if len({tuple(sorted(cases)) for cases in cases_by_arm.values()}) != 1:
        raise ValueError('CT cases do not match across arms')
    ans = {'seed':seed, 'dataset': reports['q0']['config']['data'],
           'ct_ids':sorted(cases_by_arm['q0']), 'gaps':{}}
    for gap in GAPS:
        for arm in ARMS:
            if set(reports[arm]['gaps'][gap]['by_ct']) != cases_by_arm['q0']:
                raise ValueError('Case set differs between gap and arm')
        row = {}
        for arm in ARMS:
            data = reports[arm]['gaps'][gap]
            micro = data['micro']['query']
            row[arm] = {'selected_step':reports[arm]['checkpoint_step'],
                        'n_points':micro['fov_all']['n_points'],
                        'micro_fov_epe_px':micro['fov_all']['epe_px'],
                        'macro_ct_fov_epe_px':data['macro_ct_fov_epe_px']['query'],
                        'macro_ct_proxy_epe_px':sum(
                            data['by_ct'][ct]['query']['proxy_overlap']['epe_px']
                            for ct in ans['ct_ids'] if 'proxy_overlap' in data['by_ct'][ct]['query']
                        ) / max(1, sum('proxy_overlap' in data['by_ct'][ct]['query'] for ct in ans['ct_ids'])),
                        'by_ct_fov_epe_px':{
                            ct:data['by_ct'][ct]['query']['fov_all']['epe_px']
                            for ct in ans['ct_ids']}}
        row['trained_minus_random_macro_px'] = row['q2']['macro_ct_fov_epe_px'] - row['q1']['macro_ct_fov_epe_px']
        row['trained_minus_coordinate_macro_px'] = row['q2']['macro_ct_fov_epe_px'] - row['q0']['macro_ct_fov_epe_px']
        row['q2_better_each_ct_vs_q0_q1'] = all(
            row['q2']['by_ct_fov_epe_px'][ct] < min(row['q0']['by_ct_fov_epe_px'][ct],
                                                      row['q1']['by_ct_fov_epe_px'][ct])
            for ct in ans['ct_ids'])
        ans['gaps'][gap] = row
    return ans


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=Path('runs/local'))
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--out',required=True,type=Path)
    a = p.parse_args()
    result = summarize(a.runs_root,a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print('Gap       Q0 coord     Q1 random    Q2 trained    Q2-Q1       Q2-Q0')
    for gap,row in result['gaps'].items():
        print(f"{gap:>3}  " + '  '.join(f"{row[arm]['macro_ct_fov_epe_px']:9.3f}px" for arm in ARMS)
              + f"  {row['trained_minus_random_macro_px']:+9.3f}px"
              + f"  {row['trained_minus_coordinate_macro_px']:+9.3f}px")
    print(f'CTs={result["ct_ids"]}; VAL only; TEST untouched; report={a.out}')


if __name__ == '__main__':
    main()
