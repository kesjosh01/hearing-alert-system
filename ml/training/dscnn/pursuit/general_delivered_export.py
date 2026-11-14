"""Select an existing general checkpoint and its actual int8 readout, before test."""
import itertools
import json
import shutil
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from .paths import ROOT, DATA, REPORT
from .yamnet_frontend import feature, CACHE, CONTRACT
from .cpu_integer import validated_interpreter
from . import delivered_integer as di

OUT = REPORT/'general_delivered_v1'
NAMES = ['pursuit_20261114_r04_prefix64_pretrained', 'pursuit_20261114_r05_prefix64_targets']


def inputs(row):
    m = feature(row)
    return np.load(CACHE/m['source_sha256']/'features.npy', mmap_mode='r', allow_pickle=False), m


def predict(interpreter, rows):
    result = {}
    for i, row in enumerate(rows, 1):
        result[row['sample_id']] = di.predict_q(interpreter, inputs(row)[0])
        if i % 100 == 0:
            print('GENERAL_DELIVERED_CPU_RECORDINGS', i, len(rows), flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=False)
    comparisons = [json.loads((REPORT/'operating/round04_enriched'/(n+'.json')).read_text()) for n in NAMES]
    assert all(j['complete'] and j['dataset']=='round04_enriched' for j in comparisons)
    # Both checkpoints were evaluated on exactly the same development cohort.
    chosen = max(comparisons, key=lambda j: j['best']['macro_recall'])
    name = chosen['model']; model_path = ROOT/'runs'/name/'classifier.keras'
    assert sha(model_path) == chosen['model_sha256']
    write_json(OUT/'checkpoint_choice_before_quantization.json', dict(
        candidates=[dict(name=j['model'], model_sha256=j['model_sha256'], selection=j['best']) for j in comparisons],
        selected=name, criterion='Identical selection cohort source-mean macro file recall',
        calibration_or_fresh_used=False, actual_product_targets_verified=False))
    d = DATA/'round04_enriched'; freeze = json.loads((d/'freeze.json').read_text())
    assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=read_csv(d/'development.csv'); train=[r for r in rows if r['role']=='train']
    selection=[r for r in rows if r['role']=='selection']; calibration=[r for r in rows if r['role']=='calibration']
    assert len(selection)==978 and len(calibration)==702
    rng=np.random.default_rng(42); negatives=[r for r in train if r['label']=='other']
    selected_negatives=[negatives[int(i)] for i in rng.choice(len(negatives),512,replace=False)]
    representative=[]; provenance=[]
    for row in [r for r in train if r['label']!='other']+selected_negatives:
        x,m=inputs(row)
        for i in np.unique([len(x)//2, int(np.argmax(np.mean(x,axis=(1,2,3))))]):
            representative.append(x[i]); provenance.append(dict(split='train', sample_id=row['sample_id'],
                source_sha256=m['source_sha256'], feature_sha256=m['feature_sha256'], window_index=int(i)))
    x=np.stack(representative); np.save(OUT/'representative_training.npy',x,allow_pickle=False)
    write_json(OUT/'representative_provenance.json',provenance)
    recipe=[('probability',4.),('probability',8.),('logits',1.)]
    policies=list(itertools.product([(1,1),(2,3)], repeat=2))
    write_json(OUT/'export_plan.json',dict(source_model=name, source_model_sha256=sha(model_path),
        dataset_freeze_sha256=sha(d/'freeze.json'), variants=recipe, policies=policies,
        representative_windows=len(x), representative_role='train',
        selection_criterion='Balanced target source-mean file recall in actual integer output units',
        calibration='Only raises selected cuts; never ranks variants or policies',
        no_fresh_features_or_predictions_used=True, frontend=CONTRACT,
        code_sha256={p:sha(ROOT/p) for p in ['pursuit/general_delivered_export.py','pursuit/delivered_integer.py','pursuit/cpu_integer.py']}))
    tf,keras=runtime('cpu'); assert not tf.config.list_physical_devices('GPU')
    model=keras.models.load_model(str(model_path),compile=False); variants=[]
    for kind,temp in recipe:
        directory=OUT/(kind+'_t'+str(int(temp)))
        _,meta=di.export(tf,keras,model,x,provenance,directory,CONTRACT,kind,temp)
        interpreter=validated_interpreter(tf,directory/'model_int8.tflite',x,directory/'cpu_kernel_gate.json',64)
        values=predict(interpreter,selection); np.savez(directory/'selection_outputs_q.npz',**values)
        analyses=[]
        for policy in policies:
            cuts=di.fit_cuts(selection,values,kind,meta['output']['zero_point'],policy)
            metric=di.measure(selection,values,kind,meta['output']['zero_point'],policy,cuts)
            analyses.append(dict(policy=policy,cuts=cuts,metric=metric))
            variants.append(dict(directory=directory,kind=kind,temp=temp,meta=meta,interpreter=interpreter,
                values=values,policy=policy,cuts=cuts,metric=metric))
        write_json(directory/'selection_analysis.json',dict(results=[dict(policy=a['policy'],cuts=a['cuts'].tolist(),metric=a['metric']) for a in analyses],
            selection_zero_errors_by_construction=True, fresh_test_used=False))
        print('GENERAL_DELIVERED_VARIANT_COMPLETE',kind,temp,flush=True)
    best=max(variants,key=lambda v:di.key(v['metric'])); meta=best['meta']; zero=meta['output']['zero_point']
    write_json(OUT/'chosen_before_calibration.json',dict(source_model=name,directory=str(best['directory']),
        kind=best['kind'],temperature=best['temp'],policy=best['policy'],selection=best['metric'],
        model_sha256=meta['model_sha256'],calibration_or_fresh_used_to_rank=False))
    cal=predict(best['interpreter'],calibration)
    cuts=di.fit_cuts(calibration,cal,best['kind'],zero,best['policy'],best['cuts'])
    metric=di.measure(selection,best['values'],best['kind'],zero,best['policy'],cuts)
    calibration_metric=di.measure(calibration,cal,best['kind'],zero,best['policy'],cuts)
    np.savez(OUT/'chosen_selection_and_calibration_outputs_q.npz',**dict(best['values'],**cal))
    assert sha(model_path)==chosen['model_sha256']
    result=dict(complete=True,source_model=name,source_model_sha256=chosen['model_sha256'],
        selected_tflite_sha256=meta['model_sha256'],output_kind=best['kind'],temperature=best['temp'],
        policy=best['policy'],cuts_integer_units=cuts.tolist(),selection_after_calibration=metric,
        calibration_after_fit=calibration_metric,calibration_is_threshold_fitting_not_independent_test=True,
        fresh_test_used=False,board_validated=False,actual_product_targets_verified=False)
    write_json(OUT/'status.json',result)
    bundle=REPORT/'candidates/general_prefix64_v1';bundle.mkdir(exist_ok=False)
    for f in ['model_int8.tflite','model_int8.h','model_metadata.json']:shutil.copyfile(best['directory']/f,bundle/f)
    contract=dict(class_order=di.LABELS,output_kind=best['kind'],output_scale=meta['output']['scale'],
        output_zero_point=zero,policy=best['policy'],cuts_integer_units=cuts.tolist(),
        output_semantics='Quantized softmax probabilities' if best['kind']=='probability' else 'Quantized Dense logits',
        score_semantics='Argmax gated unsigned probability' if best['kind']=='probability' else 'Argmax gated positive logit margin to strongest other class',
        target_argmax_required=True,threshold_comparison='score>=cut and score>0',
        tie_break='First maximum, matching numpy.argmax',input_shape=meta['input']['shape'],
        frontend=CONTRACT,existing_98x40_firmware_compatible=False,
        actual_notification_controller_tested=False,board_validated=False)
    write_json(bundle/'delivered_output_contract.json',contract);write_json(bundle/'analysis.json',result)
    (bundle/'README.md').write_text('# General 64-bin research candidate\n\n'
        'Frozen on development data before the public fresh-recording benchmark. '
        'Uses the separately verified official YAMNet 96x64 frontend. '
        'The existing 98x40 firmware cannot consume this model directly. '
        'No board execution, notification controller or real monthly reliability was tested. '
        'Dataset source and noncommercial restrictions remain; no commercial clearance is claimed.\n')
    write_json(bundle/'candidate.json',dict(kind='GENERAL_64BIN_RESEARCH_CANDIDATE',source_model=name,
        source_model_sha256=chosen['model_sha256'],actual_device_days=0,deployment_ready=False,
        actual_product_targets_verified=False,files_sha256={p.name:sha(p) for p in bundle.iterdir() if p.is_file()}))
    print('GENERAL_DELIVERED_BUNDLE_COMPLETE',result,flush=True)


if __name__=='__main__':main()
