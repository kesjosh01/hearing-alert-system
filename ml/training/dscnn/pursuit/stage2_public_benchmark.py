"""One new public cohort, locked before any waveform reaches a model.

No fitting loader accepts these inputs. Source recordings, labels, exact integer
models, readouts and CPU backends are frozen; results cannot drive retuning.
"""
import argparse
from collections import Counter
import hashlib
import json
import re
import shutil
import subprocess
import time
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from dscnn_baseline.frontend import windows
from .paths import ROOT, DATA, REPORT
from .data import audio_path
from .native import NativeFrontend, DIRECTORY
from .yamnet_frontend import transform
from .cpu_integer import open_interpreter
from . import delivered_integer as di
from .stage2_paired_integer import BUNDLES, OUT as PAIRED_OUT
from .final_public_benchmark import NAMES as OLD_NAMES, OUT as OLD_OUT, checked_freeze as old_checked_freeze
from .final_public_benchmark import summary, cuts_for, bout_metrics

OUT = REPORT / 'stage2_public_benchmark_v1'
PLAN = REPORT / 'stage2_public_benchmark_plan.json'
COHORT = DATA / 'round06_local_regions'
NAMES = list(OLD_NAMES) + list(BUNDLES.values())
CODE = ['pursuit/stage2_public_benchmark.py', 'pursuit/final_public_benchmark.py',
        'pursuit/delivered_integer.py', 'pursuit/cpu_integer.py', 'pursuit/native.py', 'pursuit/yamnet_frontend.py']


def identity_sets(rows):
    return dict(recordings={(r['source_dataset'], r['source_recording_id']) for r in rows},
        raw={r['source_sha256'] for r in rows}, pcm={r['canonical_pcm_sha256'] for r in rows},
        groups={r['split_group_id'] for r in rows},
        videos={r['source_video_id'] for r in rows if r.get('source_video_id')},
        known_creators={r['creator_group_id'] for r in rows if r.get('creator_group_id')})


def register():
    assert not PLAN.exists() and not OUT.exists()
    write_json(PLAN, dict(created_at_unix=time.time(), names=NAMES,
        cohort_freeze_sha256=sha(COHORT / 'freeze.json'), manifest_sha256=sha(COHORT / 'fresh_test.csv'),
        code_identity={p: sha(ROOT / p) for p in CODE},
        model_rule='Previous four locked bundles and four newly trained source models, each with its selection-chosen and calibration-fitted integer readout. The new global selection winner must be an exact readout alias of one paired bundle.',
        no_test_based_model_or_threshold_tuning=True, collection_allowed=False,
        test_unit='Whole-recording at-least-one causal positive window, source-mean also reported; not actual activation-event recall.',
        other_rule='Only label=other enters the false-alert denominator. Published overseas fire/siren/doorbell positives never enter it. Freeze target-like metadata checks before any model output.',
        foundation_pretraining_exposure_certified_absent=False, physical_device_independence_certified=False,
        actual_device_days=0, actual_product_targets_verified=False))
    print('NEW_PUBLIC_TEST_COMPARISON_RULES_REGISTERED', len(NAMES), flush=True)


def freeze():
    assert not OUT.exists(), 'Preserve a partial or completed frozen test'
    spec = json.loads(PLAN.read_text())
    assert all(sha(ROOT / p) == v for p, v in spec['code_identity'].items())
    assert sha(COHORT / 'freeze.json') == spec['cohort_freeze_sha256']
    cf = json.loads((COHORT / 'freeze.json').read_text())
    assert all(sha(COHORT / k) == v for k, v in cf['files'].items())
    assert sha(COHORT / 'fresh_test.csv') == spec['manifest_sha256']
    old_checked_freeze()
    paired = json.loads((PAIRED_OUT / 'status.json').read_text())
    assert paired['complete'] and not paired['fresh_test_used']
    rows = read_csv(COHORT / 'fresh_test.csv')
    assert len(rows) == 559 and dict(Counter(r['label'] for r in rows)) == dict(fire_alarm=41, doorbell=6, other=512)
    assert all(r['role'] == 'fresh_test' and not r.get('training_region_kind') for r in rows)
    ids = {r['sample_id'] for r in rows}
    assert len(ids) == len(rows)
    for field in ('raw', 'pcm', 'recordings'):
        assert len(identity_sets(rows)[field]) == len(rows), field
    previous_and_dev = [r for r in read_csv(COHORT / 'inventory.csv') if r['sample_id'] not in ids]
    a, b = identity_sets(rows), identity_sets(previous_and_dev)
    overlaps = {key: len(value & b[key]) for key, value in a.items()}
    assert not any(overlaps.values()), overlaps
    metadata_hits = []
    for r in rows:
        if r['label'] != 'other':
            continue
        text = ' '.join(str(r.get(k, '')) for k in ['title', 'tags', 'labels', 'fine_labels'])
        if re.search(r'\b(door[\s_.-]*bell|fire[\s_.-]*alarm|smoke[\s_.-]*(alarm|detector)|siren|feueralarm)\b', text, re.I):
            metadata_hits.append(r['sample_id'])
    assert not metadata_hits, 'Resolve published target metadata before opening this test, never after model errors'
    candidates = {}
    for name in NAMES:
        p = REPORT / 'candidates' / name
        candidate = json.loads((p / 'candidate.json').read_text())
        assert not candidate['deployment_ready'] and not candidate['actual_product_targets_verified']
        assert all(sha(p / k) == v for k, v in candidate['files_sha256'].items())
        gate = OLD_OUT / (name + '_cpu_kernel_gate.json') if name in OLD_NAMES else p / 'cpu_kernel_gate.json'
        gate_data = json.loads(gate.read_text())
        assert gate_data['source_model_sha256'] == sha(p / 'model_int8.tflite')
        candidates[name] = dict(files={k: sha(p / k) for k in
            ['model_int8.tflite', 'model_metadata.json', 'delivered_output_contract.json', 'candidate.json']},
            cpu_gate=str(gate), cpu_gate_sha256=sha(gate), optimized_cpu=bool(gate_data['passed']))
    global_path = REPORT / 'candidates/native40_r06_v1'
    alias_path = REPORT / 'candidates' / paired['global_bundle_equivalent_to']
    assert sha(global_path / 'model_int8.tflite') == sha(alias_path / 'model_int8.tflite')
    global_cfg = json.loads((global_path / 'delivered_output_contract.json').read_text())
    alias_cfg = json.loads((alias_path / 'delivered_output_contract.json').read_text())
    assert global_cfg == alias_cfg
    OUT.mkdir()
    write_json(OUT / 'freeze.json', dict(created_at_unix=time.time(), plan_sha256=sha(PLAN),
        names=NAMES, candidates=candidates, manifest_sha256=sha(COHORT / 'fresh_test.csv'),
        dataset_freeze_sha256=sha(COHORT / 'freeze.json'), native_binary_sha256=sha(DIRECTORY / 'libpursuit_frontend.so'),
        code_identity=spec['code_identity'], previous_and_development_identity_overlaps=overlaps,
        other_target_like_metadata_hits=metadata_hits, target_recordings_excluded_from_other_denominator=47,
        class_counts=dict(Counter(r['label'] for r in rows)),
        global_winner_alias=paired['global_bundle_equivalent_to'],
        global_candidate_sha256=sha(global_path / 'candidate.json'),
        no_test_based_model_or_threshold_tuning=True, acoustic_bouts_are_actual_activations=False,
        no_independent_domestic_positive_recordings=True, foundation_pretraining_exposure_certified_absent=False,
        all_physical_device_identities_certified=False, actual_device_days=0, actual_product_targets_verified=False))
    print('NEW_PUBLIC_TEST_MODELS_AND_READOUTS_LOCKED', len(rows), overlaps, flush=True)


def checked_freeze():
    f = json.loads((OUT / 'freeze.json').read_text())
    assert f['names'] == NAMES and sha(PLAN) == f['plan_sha256']
    assert sha(COHORT / 'fresh_test.csv') == f['manifest_sha256']
    assert sha(COHORT / 'freeze.json') == f['dataset_freeze_sha256']
    assert all(sha(ROOT / p) == v for p, v in f['code_identity'].items())
    assert sha(DIRECTORY / 'libpursuit_frontend.so') == f['native_binary_sha256']
    for name, meta in f['candidates'].items():
        assert all(sha(REPORT / 'candidates' / name / k) == v for k, v in meta['files'].items())
        assert sha(ROOT / meta['cpu_gate']) == meta['cpu_gate_sha256']
    assert sha(REPORT / 'candidates/native40_r06_v1/candidate.json') == f['global_candidate_sha256']
    return f


def evaluate():
    f = checked_freeze()
    assert not (OUT / 'results.json').exists(), 'Never rerun a completed fixed test for tuning'
    rows = read_csv(COHORT / 'fresh_test.csv')
    tf, _ = runtime('cpu')
    assert not tf.config.list_physical_devices('GPU')
    models = {}
    for name, fixed in f['candidates'].items():
        p = REPORT / 'candidates' / name
        cfg = json.loads((p / 'delivered_output_contract.json').read_text())
        interpreter = open_interpreter(tf, p / 'model_int8.tflite', fixed['optimized_cpu'], 2)
        models[name] = (interpreter, cfg)
    for name in ('recordings', 'q_traces'):
        (OUT / name).mkdir(exist_ok=True)
    native, records = NativeFrontend(), []
    started = time.monotonic()
    for i, row in enumerate(rows, 1):
        assert row['role'] == 'fresh_test' and re.fullmatch(r'[A-Za-z0-9_-]+', row['sample_id'])
        record_path = OUT / 'recordings' / (row['sample_id'] + '.json')
        if record_path.exists():
            record = json.loads(record_path.read_text())
            assert record['complete'] and record['freeze_sha256'] == sha(OUT / 'freeze.json')
            assert sha(OUT / 'q_traces' / record['trace_filename']) == record['trace_sha256']
            records.append(record)
            continue
        assert shutil.disk_usage(ROOT).free > 8 * 2**30
        source = audio_path(row)
        assert sha(source) == row['source_sha256']
        pcm_bytes = subprocess.check_output(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source),
            '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-f', 's16le', '-'], stderr=subprocess.PIPE)
        assert hashlib.sha256(pcm_bytes).hexdigest() == row['canonical_pcm_sha256']
        pcm = np.frombuffer(pcm_bytes, dtype='<i2')
        pairs = list(windows(pcm)); starts = [s for s, _ in pairs]
        x40 = np.stack([native(np.ascontiguousarray(a)) for _, a in pairs])
        with tf.device('/CPU:0'):
            x64 = np.concatenate([transform(tf, np.stack([a[:15600].astype(np.float32) / 32768.
                for _, a in pairs[j:j+128]])).numpy() for j in range(0, len(pairs), 128)])
        assert x40.shape[1:] == (98, 40, 1) and x64.shape[1:] == (96, 64, 1)
        outputs, traces = {}, {}
        for name, (interpreter, cfg) in models.items():
            input_shape = interpreter.get_input_details()[0]['shape'].tolist()
            x = x64 if input_shape == [1, 96, 64, 1] else x40
            assert input_shape == [1, *x.shape[1:]]
            q = di.predict_q(interpreter, x); traces[name] = q
            scores = di.scores(q, cfg.get('output_kind', 'probability'), cfg['output_zero_point'], cfg['policy'])
            flags = (scores >= cuts_for(cfg)) & (scores > 0)
            output = dict(maximum_scores=scores.max(0).tolist(), positive_window_counts=flags.sum(0).tolist())
            if row['label'] != 'other' and row.get('event_intervals_json'):
                intervals = json.loads(row['event_intervals_json'])
                output['published_target_acoustic_bouts'] = bout_metrics(flags[:, di.LABELS.index(row['label'])], starts, intervals)
            outputs[name] = output
        trace = OUT / 'q_traces' / (row['sample_id'] + '.npz')
        np.savez_compressed(trace, **traces)
        record = dict(complete=True, freeze_sha256=sha(OUT / 'freeze.json'), row=row, models=outputs,
            trace_filename=trace.name, trace_sha256=sha(trace), feature_arrays_saved=False)
        write_json(record_path, record); records.append(record)
        if i % 25 == 0:
            write_json(OUT / 'progress.json', dict(records_complete=i, total=len(rows), elapsed_seconds=time.monotonic()-started))
            print('NEW_PUBLIC_TEST_RECORDINGS', i, len(rows), flush=True)
    checked_freeze()
    result = dict(complete=True, freeze_sha256=sha(OUT / 'freeze.json'), recordings=len(records),
        models={name: summary(records, name) for name in NAMES}, global_winner_alias=f['global_winner_alias'],
        no_test_based_model_or_threshold_tuning=True, feature_arrays_saved=False,
        actual_activation_event_recall_verified=False, actual_device_days=0,
        actual_product_targets_verified=False, elapsed_seconds=time.monotonic()-started)
    write_json(OUT / 'results.json', result)
    for name, metric in result['models'].items():
        print('NEW_PUBLIC_TEST_RESULT', name, {k:(v['detected_files'], v['files']) for k,v in metric['classes'].items()},
              'OTHER_FALSE_FILES', metric['other_false_alert_files'], '/', metric['other_files'], flush=True)
    print('NEW_FIXED_PUBLIC_TEST_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'freeze', 'evaluate'])
    args = parser.parse_args()
    {'register': register, 'freeze': freeze, 'evaluate': evaluate}[args.mode]()
