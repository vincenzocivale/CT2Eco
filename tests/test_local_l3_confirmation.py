"""Checks for reproducible multi-seed L3 confirmation reports."""
import json

import pytest

from scripts.report_local_l3_seeds import best_record, compare


def _record(arm, step, score, full, base=.40, pixels=123):
    return {'variant': arm, 'step': step,
            'refined_dz_mae_mm': score, 'base_dz_mae_mm': base,
            'full_refined_dz_mae_mm': full, 'full_base_dz_mae_mm': .50,
            'primary_pixels': pixels, 'full_valid_pixels': pixels + 10,
            'by_ct': {'ct1': {'pixels': pixels, 'refined_dz_mae_mm': score}}}


def _write(root, arm, seed, records):
    folder = root / f'l3_{arm}_s{seed}'
    folder.mkdir(parents=True)
    (folder / 'best.pt').write_bytes(b'checkpoint-present')
    (folder / 'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return folder


def test_selects_best_and_pairs_validation_across_seeds(tmp_path):
    for seed in (17, 42):
        _write(tmp_path, 'r0', seed, [_record('r0', 100, .37, .47),
                                      _record('r0', 200, .35, .45)])
        _write(tmp_path, 'r2', seed, [_record('r2', 100, .34, .44),
                                      _record('r2', 200, .36, .46)])
    report = compare(tmp_path, [17, 42])
    assert report['r2_better_seeds'] == 2
    assert report['r2_better_in_every_ct_of_every_seed']
    assert report['mean_r2_improvement_pct'] == pytest.approx(100 * (.35-.34)/.35)
    assert report['rows'][0]['r0_step'] == 200
    assert report['rows'][0]['r2_step'] == 100


def test_refuses_nonpaired_ct_or_mask(tmp_path):
    _write(tmp_path, 'r0', 17, [_record('r0', 150, .35, .45)])
    _write(tmp_path, 'r2', 17, [_record('r2', 150, .34, .44, pixels=100)])
    with pytest.raises(ValueError, match='Unpaired'):
        compare(tmp_path, [17])


def test_refuses_missing_selected_checkpoint(tmp_path):
    _write(tmp_path, 'r0', 17, [_record('r0', 150, .35, .45)])
    folder = _write(tmp_path, 'r2', 17, [_record('r2', 150, .34, .44)])
    (folder / 'best.pt').unlink()
    with pytest.raises(FileNotFoundError, match='checkpoint'):
        compare(tmp_path, [17])
