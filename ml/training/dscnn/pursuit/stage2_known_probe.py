"""Evaluate fixed delivered readouts on the same known domestic source probes.

This is a training-cohort diagnostic and never an independent target test.
Existing native40 feature arrays are read only; no waveforms/features are made.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from .paths import ROOT, REPORT
from .data import audio_path, CACHE
from .augmentation import DEST as AUG_CACHE
from .stage2_paired_integer import BUNDLES, OUT as PAIRED_OUT
from .cpu_integer import open_interpreter
from .final_public_benchmark import cuts_for
from . import delivered_integer as di

PLAN = REPORT / 'stage2_known_domestic_probe_plan.json'
OUT = REPORT / 'stage2_known_domestic_probe_v1'
PROBES = ROOT / 'runs/pursuit_20261114_domestic_seen_aug_presence/known_positive_probes.csv'


def register():
    assert not PLAN.exists() and not OUT.exists()
    assert not PAIRED_OUT.exists(), 'Register this diagnostic before delivered choices exist'
    rows = read_csv(PROBES)
    assert len(rows) == 172 and len({r['original_sample_id'] for r in rows}) == 43
    write_json(PLAN, dict(created_at_unix=time.time(), names=list(BUNDLES.values()),
        probes_sha256=sha(PROBES), code_sha256=sha(Path(__file__)),
        unit='Same 43 training-cohort target recordings and the pre-existing quiet/room_snr20/room_snr10 variants. Not independent devices or activation events.',
        no_model_or_readout_changes_from_results=True, collection_allowed=False,
        feature_arrays_written=False, independent_target_recordings=0, actual_product_targets_verified=False))
    print('FIXED_KNOWN_DOMESTIC_DIAGNOSTIC_REGISTERED', flush=True)


def evaluate():
    spec = json.loads(PLAN.read_text())
    assert sha(PROBES) == spec['probes_sha256'] and sha(Path(__file__)) == spec['code_sha256']
    assert json.loads((PAIRED_OUT / 'status.json').read_text())['complete']
    assert not OUT.exists(), 'Preserve completed or partial diagnostic'
    rows = read_csv(PROBES)
    identities, arrays = {}, {}
    for name in spec['names']:
        p = REPORT / 'candidates' / name
        c = json.loads((p / 'candidate.json').read_text())
        assert all(sha(p / k) == v for k, v in c['files_sha256'].items())
        identities[name] = dict(model_sha256=sha(p / 'model_int8.tflite'),
            readout_sha256=sha(p / 'delivered_output_contract.json'), gate_sha256=sha(p / 'cpu_kernel_gate.json'))
    for row in rows:
        assert row['role'] == 'train' and row['label'] != 'other'
        raw = sha(audio_path(row)); kind = row['probe_condition']
        directory = CACHE / raw if kind == 'original' else AUG_CACHE / raw / kind
        m = json.loads((directory / 'metadata.json').read_text())
        assert m['source_sha256'] == raw and sha(directory / 'features.npy') == m['feature_sha256']
        arrays[row['sample_id']] = dict(path=str(directory / 'features.npy'), sha256=m['feature_sha256'])
    OUT.mkdir()
    write_json(OUT / 'freeze.json', dict(plan_sha256=sha(PLAN), candidates=identities,
        probes_sha256=sha(PROBES), read_only_feature_identities=arrays, readouts_frozen_before_probes=True,
        independent_target_recordings=0, actual_product_targets_verified=False))
    tf, _ = runtime('cpu')
    assert not tf.config.list_physical_devices('GPU')
    models = {}
    for name in spec['names']:
        p = REPORT / 'candidates' / name
        cfg = json.loads((p / 'delivered_output_contract.json').read_text())
        gate = json.loads((p / 'cpu_kernel_gate.json').read_text())
        assert gate['source_model_sha256'] == identities[name]['model_sha256']
        models[name] = (open_interpreter(tf, p / 'model_int8.tflite', bool(gate['passed']), 2), cfg)
    values = {name: {} for name in models}
    for i, row in enumerate(rows, 1):
        info = arrays[row['sample_id']]
        assert sha(Path(info['path'])) == info['sha256']
        x = np.load(info['path'], mmap_mode='r', allow_pickle=False)
        assert x.shape[1:] == (98, 40, 1) and x.dtype == np.float32
        for name, (interpreter, _) in models.items():
            values[name][row['sample_id']] = di.predict_q(interpreter, x)
        if i % 43 == 0:
            print('FIXED_DOMESTIC_PROBES', i, len(rows), flush=True)
    metrics = {}
    for name, (_, cfg) in models.items():
        metrics[name] = di.measure(rows, values[name], cfg['output_kind'], cfg['output_zero_point'], cfg['policy'], cuts_for(cfg))
        np.savez_compressed(OUT / (name + '_q.npz'), **values[name])
        p = REPORT / 'candidates' / name
        assert sha(p / 'model_int8.tflite') == identities[name]['model_sha256']
        assert sha(p / 'delivered_output_contract.json') == identities[name]['readout_sha256']
    references = {}
    for name in ['domestic_known_v1', 'domestic_known_aug_v2']:
        p = REPORT / 'candidates' / name
        analysis = json.loads((p / 'analysis.json').read_text())
        references[name] = dict(existing_analysis_sha256=sha(p / 'analysis.json'),
            result=analysis.get('selection_after_calibration', analysis), newly_evaluated=False)
    write_json(OUT / 'results.json', dict(complete=True, models=metrics, previous_frozen_references=references,
        original_target_recordings=43, transformed_probes=172, probes_from_training_cohort=True,
        independent_target_recordings=0, feature_arrays_written=False, thresholds_or_weights_refitted=False,
        actual_activation_event_or_monthly_rate_verified=False, actual_product_targets_verified=False))
    print('FIXED_KNOWN_DOMESTIC_DIAGNOSTIC_COMPLETE',
        {name: value['conditions']['original'] for name, value in metrics.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'evaluate'])
    args = parser.parse_args()
    register() if args.mode == 'register' else evaluate()
