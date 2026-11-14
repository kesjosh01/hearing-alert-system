"""Freeze one selection-chosen int8 readout per trained source before new test use."""
import argparse
import json
import shutil
import time
import traceback
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from .paths import ROOT, DATA, REPORT
from .stage2_stream import FeatureStore
from .stage2_native_export import PLAN as EXPORT_PLAN, OUT as EXPORT_OUT, NAMES, selection_key, predict
from .cpu_integer import open_interpreter
from . import delivered_integer as di

PLAN = REPORT / 'stage2_paired_integer_plan.json'
OUT = REPORT / 'stage2_paired_integer_v1'
BUNDLES = dict(zip(NAMES, ['native40_r06_supervised_v1', 'native40_r06_distilled_v1',
                          'native40_r06_supervised_bnrefit_v1', 'native40_r06_distilled_bnrefit_v1']))


def register():
    assert not PLAN.exists() and not OUT.exists()
    assert not (EXPORT_OUT / 'status.json').exists(), 'Register before observing the int8 comparison'
    assert not (REPORT / 'stage2_public_benchmark_v1').exists()
    write_json(PLAN, dict(created_at_unix=time.time(), export_plan_sha256=sha(EXPORT_PLAN),
        source_bundles=BUNDLES, rule='For each of the four pre-registered sources, independently select its best export/readout using the exact original selection key. Freeze all four choices before reading any calibration outputs here.',
        calibration='For each fixed choice, only raise cuts using the same held-out calibration rows; never compare or re-select on calibration results.',
        comparison='Report supervised versus KD before and after the identical train-only BN moment recipe. New public test remains unopened until all model/readout bytes are frozen.',
        original_global_export_rule_unchanged=True, no_additional_model_fit=True,
        collection_allowed=False, feature_arrays_saved=False, new_fresh_used=False,
        actual_product_targets_verified=False,
        code_identity={str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'pursuit/stage2_paired_integer.py',
            ROOT / 'pursuit/stage2_native_export.py', ROOT / 'pursuit/stage2_stream.py',
            ROOT / 'pursuit/cpu_integer.py', ROOT / 'pursuit/delivered_integer.py']}))
    print('PAIRED_ACTUAL_INT8_COMPARISON_REGISTERED', flush=True)


def evaluate():
    spec = json.loads(PLAN.read_text())
    assert all(sha(ROOT / k) == v for k, v in spec['code_identity'].items())
    assert sha(EXPORT_PLAN) == spec['export_plan_sha256']
    assert json.loads((REPORT / 'collection_stopped_by_user.json').read_text())['collection_disabled_by_user']
    original = json.loads((EXPORT_OUT / 'status.json').read_text())
    assert original['complete'] and not original['fresh_test_used']
    export_spec = json.loads(EXPORT_PLAN.read_text())
    directory = DATA / export_spec['dataset']
    assert sha(directory / 'freeze.json') == export_spec['dataset_freeze_sha256']
    frozen = json.loads((directory / 'freeze.json').read_text())
    assert all(sha(directory / k) == v for k, v in frozen['files'].items())
    rows = read_csv(directory / 'development.csv')
    selection = [r for r in rows if r['role'] == 'selection']
    calibration = [r for r in rows if r['role'] == 'calibration']
    choices = json.loads((EXPORT_OUT / 'chosen_before_calibration.json').read_text())['comparison']
    chosen = []
    for name, bundle in spec['source_bundles'].items():
        candidates = [v for v in choices if v['name'] == name]
        assert len(candidates) == len(export_spec['variants']) * len(export_spec['policies'])
        best = max(candidates, key=lambda v: selection_key(v['metric']))
        target = EXPORT_OUT / name / (best['kind'] + '_t' + str(int(best['temperature'])))
        meta = json.loads((target / 'model_metadata.json').read_text())
        assert sha(target / 'model_int8.tflite') == meta['model_sha256']
        analysis = json.loads((target / 'selection_analysis.json').read_text())
        exact = next(v for v in analysis['results'] if v['policy'] == best['policy'])
        assert exact['metric'] == best['metric']
        chosen.append(dict(source=name, bundle=bundle, kind=best['kind'], temperature=best['temperature'],
            policy=best['policy'], directory=str(target), model_sha256=meta['model_sha256'],
            source_model_sha256=sha(ROOT / 'runs' / name / 'classifier.keras'),
            selection_cuts=exact['cuts'], selection=exact['metric']))
    OUT.mkdir(exist_ok=False)
    write_json(OUT / 'choices_before_calibration.json', dict(choices=chosen,
        plan_sha256=sha(PLAN), calibration_or_fresh_used_to_choose=False))
    started = time.monotonic()
    state = dict(complete=False, exit_code=None, fresh_test_used=False, actual_product_targets_verified=False)
    write_json(OUT / 'status.json', state)
    try:
        tf, _ = runtime('cpu')
        assert not tf.config.list_physical_devices('GPU')
        store = FeatureStore(rows, tf, kind='portable40', budget_bytes=1536 * 2**20)
        results = []
        for choice in chosen:
            target = ROOT / choice['directory']
            meta = json.loads((target / 'model_metadata.json').read_text())
            gate = json.loads((target / 'cpu_kernel_gate.json').read_text())
            assert gate['source_model_sha256'] == choice['model_sha256']
            interpreter = open_interpreter(tf, target / 'model_int8.tflite', bool(gate['passed']), 2)
            with np.load(target / 'selection_outputs_q.npz', allow_pickle=False) as data:
                sel = {k: data[k] for k in data.files}
            cuts = np.asarray(choice['selection_cuts'], np.int16)
            assert di.measure(selection, sel, choice['kind'], meta['output']['zero_point'], choice['policy'], cuts) == choice['selection']
            cal = predict(interpreter, calibration, store)
            final_cuts = di.fit_cuts(calibration, cal, choice['kind'], meta['output']['zero_point'], choice['policy'], cuts)
            assert np.all(final_cuts >= cuts)
            result = dict(**choice, cuts_integer_units=final_cuts.tolist(),
                selection_after_calibration=di.measure(selection, sel, choice['kind'], meta['output']['zero_point'], choice['policy'], final_cuts),
                calibration_after_fit=di.measure(calibration, cal, choice['kind'], meta['output']['zero_point'], choice['policy'], final_cuts),
                calibration_is_fitting_not_independent_test=True, actual_product_targets_verified=False)
            np.savez_compressed(OUT / (choice['bundle'] + '_selected_q.npz'), **dict(sel, **cal))
            bundle = REPORT / 'candidates' / choice['bundle']
            bundle.mkdir(exist_ok=False)
            for filename in ['model_int8.tflite', 'model_int8.h', 'model_metadata.json', 'cpu_kernel_gate.json']:
                shutil.copyfile(target / filename, bundle / filename)
            write_json(bundle / 'delivered_output_contract.json', dict(class_order=di.LABELS,
                output_kind=choice['kind'], output_scale=meta['output']['scale'], output_zero_point=meta['output']['zero_point'],
                policy=choice['policy'], cuts_integer_units=final_cuts.tolist(), input_shape=[1, 98, 40, 1],
                frontend=export_spec['frontend'], target_argmax_required=True, threshold_comparison='score>=cut and score>0',
                tie_break='First maximum', existing_portable_frontend_compatible=True,
                score_semantics='Argmax-gated unsigned probability' if choice['kind'] == 'probability' else 'Argmax-gated positive logit margin to strongest other class',
                notification_controller_on_actual_hardware_validated=False, board_validated=False))
            write_json(bundle / 'analysis.json', result)
            (bundle / 'README.md').write_text('Trained research comparison model, selected on development data and calibrated before the new fixed public test. '
                'See delivered_output_contract.json for exact integer readout. Microphone, board and field reliability remain unverified.\n')
            write_json(bundle / 'candidate.json', dict(kind='PAIRED_NATIVE40_R06_RESEARCH_CANDIDATE',
                source_model=choice['source'], source_model_sha256=choice['source_model_sha256'],
                actual_device_days=0, deployment_ready=False, actual_product_targets_verified=False,
                files_sha256={p.name: sha(p) for p in bundle.iterdir() if p.is_file()}))
            assert sha(ROOT / 'runs' / choice['source'] / 'classifier.keras') == choice['source_model_sha256']
            results.append(result)
            print('PAIRED_INT8_CALIBRATED', choice['bundle'],
                result['selection_after_calibration']['fire_recall'], result['selection_after_calibration']['doorbell_recall'], flush=True)
        winner = next(v for v in results if v['source'] == original['source_model'])
        assert winner['model_sha256'] == original['selected_tflite_sha256']
        assert winner['kind'] == original['output_kind'] and winner['policy'] == original['policy']
        assert winner['cuts_integer_units'] == original['cuts_integer_units']
        state.update(complete=True, exit_code=0, results=results, plan_sha256=sha(PLAN),
            global_bundle_equivalent_to=winner['bundle'], global_bundle='native40_r06_v1',
            feature_arrays_saved=False, collection_allowed=False, elapsed_seconds=time.monotonic() - started)
        write_json(OUT / 'status.json', state)
        print('PAIRED_NATIVE40_INT8_READOUTS_FROZEN', len(results), flush=True)
    except Exception as error:
        state.update(exit_code=1, first_error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        write_json(OUT / 'status.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['register', 'evaluate'])
    args = parser.parse_args()
    register() if args.mode == 'register' else evaluate()
