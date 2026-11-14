"""Final read-only provenance, leakage, exposure and preservation audit."""
import ast
import importlib.util
import json
import subprocess
from pathlib import Path
import numpy as np
from .protocol import ROOT,REPO,DATA,REPORT
from .dataset import sha,read_csv,write_json,validate


def read(path):return json.loads(Path(path).read_text())


def main():
    out=REPORT/'final_audit.json'
    if out.exists():raise FileExistsError('Preserve completed audit; inspect any failure instead of replacing it')
    checks={}
    def check(name,value,detail=None):
        checks[name]=dict(passed=bool(value),detail=detail)
        if not value:raise AssertionError(name+': '+str(detail))
    try:
        freeze=read(DATA/'source_freeze.json');development=read_csv(DATA/'development.csv');added=read_csv(DATA/'new_training.csv');fresh=read_csv(DATA/'fresh_test.csv')
        check('frozen_source_manifests_unchanged',all(sha(DATA/k)==v for k,v in freeze['files'].items()))
        validate(development+added+fresh);check('no_cross_role_source_recording_hash_or_known_creator_family',True)
        check('all_accepted_public_source_bytes_unchanged',all(sha(ROOT/r['relative_path'])==r['source_sha256'] for r in added+fresh),len(added+fresh))
        code=read(REPORT/'training_code_freeze.json')['code_identity'];check('entire_pretraining_code_freeze_unchanged',all(sha(ROOT/k)==v for k,v in code.items()),len(code))
        for p in (ROOT/'comparison').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
        check('comparison_modules_parse',True)
        ledger=read(REPORT/'ledger.json');byname={e['name']:e for e in ledger};run_details=[]
        for entry in ledger:
            run=Path(entry['run']);status=read(run/'status.json');meta=read(run/'run.json');rr=read_csv(run/'used_train_selection.csv');feature=read(run/'feature_provenance.json')
            assert status['complete'] and status['exit_code']==0
            assert status['actual_optimizer_steps']==status['epochs_completed']*meta['options']['steps_per_epoch']
            assert sha(run/'classifier.keras')==status['model_sha256']==entry['model_sha256']
            assert meta['options']==entry['options'] and meta['spec']['seed']==entry['seed']
            assert not meta['test_used'] and not meta['calibration_used']
            assert set(r['role'] for r in rr)=={'train','selection'}
            assert {r['sample_id'] for r in rr}=={r['sample_id'] for r in feature}
            assert all(r['role'] in ('train','selection') for r in feature)
            for key in ('train','model','metrics','features','sampling','protocol'):
                path='comparison/'+key+'.py';assert meta['code_identity'][path]==code[path]
                assert sha(run/'source_snapshot'/path)==code[path]
            expected=development+(added if meta['options']['added_training_data'] else [])
            assert {r['sample_id'] for r in rr}=={r['sample_id'] for r in expected if r['role'] in ('train','selection')}
            ids={r['sample_id'] for r in rr if r['role']=='train'}
            teacher_provenance=run/'teacher_provenance.json'
            if meta['options']['kd_weight']>0:
                receipt=read(teacher_provenance);teacher=Path(receipt['run'])
                assert receipt['model_sha256']==sha(teacher/'classifier.keras')
                tr=read_csv(teacher/'used_train_selection.csv');assert {r['sample_id'] for r in tr if r['role']=='train'}==ids
                assert receipt['logits_sha256']==sha(run/'training_teacher_logits.npz')
                with np.load(run/'training_teacher_logits.npz',allow_pickle=False) as z:assert set(z.files)==ids
            else:assert not teacher_provenance.exists()
            exposure=read(run/'selected_checkpoint_exposure_audit.json')
            assert exposure['deterministic_sampling_replay_matches_logged_full_run_exactly'] and exposure['selected_epoch']==status['selected_epoch']
            parent=byname.get(entry['parent']);changes=[]
            if parent:
                changes=[k for k in meta['options'] if meta['options'][k]!=parent['options'][k]]
                if entry['factor']=='architecture_paired_seed':assert changes in ([],['architecture'])
                else:assert changes==[entry['factor']]
            run_details.append(dict(name=entry['name'],optimizer_steps=status['actual_optimizer_steps'],selected_epoch=status['selected_epoch'],
                                    changed_options=changes,negative_exposure_at_saved_checkpoint=exposure['selected_checkpoint']['other'],real_training=True))
        check('all_student_training_and_single_setting_comparisons_verified',True,run_details)
        check('all_requested_model_stages_represented',set(range(1,8)).issubset({e['stage'] for e in ledger}))
        architecture=read(REPORT/'architecture_replicates.json');paired={}
        for name in ('temporal24','freq24'):
            paired[name]=[read(Path(p)/'run.json') for p in architecture['results'][name]['runs']]
            assert [x['spec']['seed'] for x in paired[name]]==[42,43,44]
        for a,b in zip(paired['temporal24'],paired['freq24']):
            assert {k:v for k,v in a['options'].items() if k!='architecture'}=={k:v for k,v in b['options'].items() if k!='architecture'}
        check('paired_three_seed_architecture_same_settings',True)
        candidate=read(REPORT/'final_candidate.json');run=Path(candidate['champion']['run']);saved=read(run/'selected_checkpoint_exposure_audit.json')
        check('all_eligible_negative_files_reached_saved_final_checkpoint',saved['selected_checkpoint']['other']['sampled']==saved['selected_checkpoint']['other']['total'])
        cal=read(REPORT/'calibration'/run.name/'results.json');baseline_cal=read(REPORT/'calibration/dscnn_comparison_20261114_s1_baseline/results.json')
        for cr in (cal,baseline_cal):
            assert cr['model_settings_frozen_before_calibration'] and not cr['calibration_used_to_rank_models'] and not cr['fresh_test_used']
            assert cr['decision_calibration_role']=='calibration' and cr['quantization_representative_role']=='train'
            assert cr['after_float_calibration']['guards_passed'] and cr['after_int8_calibration']['guards_passed']
        check('actual_float_int8_models_separately_calibrated_on_held_out_sources',True)
        selection=read(REPORT/'temporal_replay/selection.json');event=read(REPORT/'temporal_replay/frozen_policy.json')
        check('all_temporal_factors_compared_on_selection_only',selection['selection_only'] and not selection['calibration_or_test_used_to_select'] and
              [x['factor'] for x in selection['ledger']]==['baseline','vote_rule','release_ratio','cooldown_s'])
        registered=read(REPORT/'evaluation/preregistered_models.json');evaluation=read(REPORT/'evaluation/status.json')
        check('final_policy_and_model_hashes_frozen_before_test',registered['thresholds_and_models_fixed_before_test'] and
              registered['event_policy']==event and registered['models'][-1]['float_sha256']==sha(run/'classifier.keras')==candidate['model_sha256'] and
              registered['models'][-1]['int8_sha256']==sha(run/'export/dscnn_int8.tflite'))
        check('evaluation_completed_without_test_driven_reselection',evaluation['complete'] and evaluation['no_test_driven_reselection'])
        results=read(REPORT/'evaluation/results.json')
        check('all_five_models_both_backends_both_cohorts_scored',len(results)==20)
        check('sirens_excluded_before_foreign_fpr_scoring',all(x['metrics']['foreign_negative_clips']==(394 if x['scope']=='historical' else 67) for x in results) and len(evaluation['excluded_historical_sirens'])==8)
        bundle=Path(read(REPORT/'packaging.json')['bundle']);manifest=read(bundle/'release_manifest.json')
        check('trained_bundle_file_hashes_match',all(sha(bundle/k)==v for k,v in manifest['files'].items()))
        check('packaged_checkpoint_matches_frozen_model',sha(bundle/'classifier.keras')==candidate['model_sha256'])
        metadata=read(bundle/'model_metadata.json');check('no_synthetic_or_board_claims',not metadata['pipeline_only'] and metadata['trained'] and not metadata['deployment_ready'] and not evaluation['field_deployment_ready'])
        cli=read(REPORT/'cli_verification.json');check('offline_cli_matches_frozen_predictions_and_replay',cli['passed'])
        # Re-hash originals, not new experiment files. Package metadata inventory
        # equality also detects installations/removals in the existing venv.
        before=read(REPORT/'protected_dscnn_before.json')
        current=dict(files={k:sha(k) for k in before['files']},environment_metadata={str(p):sha(p) for p in (ROOT/'.venv').glob('lib/python*/site-packages/*.dist-info/METADATA')})
        write_json(REPORT/'protected_dscnn_after.json',current)
        check('previous_dscnn_code_data_reports_weights_and_environment_preserved',current==before,dict(protected_files=len(before['files']),packages=len(current['environment_metadata'])))
        spec=importlib.util.spec_from_file_location('comparison_protected_snapshot',ROOT/'tools/protect_baseline.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        current_yamnet=module.snapshot();write_json(REPORT/'protected_yamnet_after.json',current_yamnet)
        check('yamnet_sources_data_weights_reports_and_all_three_environments_preserved',current_yamnet==read(REPORT/'protected_yamnet_before.json'),dict(protected_files=len(current_yamnet['files']),environments=len(current_yamnet['packages'])))
        receipts=[read(p) for p in (REPORT/'processes').glob('*.json')]
        check('all_tracked_training_calibration_and_evaluation_processes_exited_successfully',all(x['exit_code']==0 for x in receipts),len(receipts))
        check('every_tracked_stage_used_existing_packages_without_environment_mutation',all(not x['package_changes'] and x['process_environment_only'] for x in receipts))
        write_json(out,dict(passed=True,checks=checks,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  actual_training_and_host_audio_evaluation=True,gpu_or_model_execution_started_by_this_audit=False,
                  hardware_microphone_or_board_validated=False,field_deployment_ready=False))
        print('COMPARISON_FINAL_AUDIT_PASS',len(checks),flush=True)
    except Exception as e:
        write_json(out,dict(passed=False,checks=checks,first_fault=f'{type(e).__name__}: {e}'))
        raise


if __name__=='__main__':main()
