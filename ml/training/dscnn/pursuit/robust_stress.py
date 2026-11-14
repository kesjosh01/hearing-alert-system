"""New acoustic combinations of seen components, not an independent device test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.frontend import windows
from .paths import DATA,ROOT,REPORT
from .augmentation import pcm
from .native import NativeFrontend

PLAN=REPORT/'domestic_holdout_acoustic_plan.json'
DEST=ROOT/'.cache/pursuit_domestic_acoustic_holdout_v1'
KINDS=['quiet18','room_snr15_alt','room_snr5_alt']
CANDIDATES=['domestic_known_v1','domestic_known_aug_v2']


def plan():
    if PLAN.exists():raise FileExistsError('Preserve the acoustic stress design')
    augmentation=json.loads((REPORT/'domestic_seen_v1_waveform_augmentation.json').read_text());assert augmentation['complete']
    used={r['noise']['sample_id'] for r in augmentation['receipts'] if r['noise']}
    ambient={'Speech','Male_speech_and_man_speaking','Female_speech_and_woman_speaking','Rain','Water','Wind',
        'Water_tap_and_faucet','Vacuum_cleaner','Washing_machine','Traffic_noise_and_roadway_noise'}
    rows=read_csv(DATA/'domestic_seen_v1/development.csv')
    noise=sorted(r['sample_id'] for r in rows if r['role']=='train' and r['label']=='other' and r['sample_id'] not in used and set(r.get('source_category','').split(','))&ambient)
    assert len(noise)>=6
    write_json(PLAN,dict(candidates=CANDIDATES,conditions=KINDS,noise_source_ids=noise,
        source_freeze_sha256=sha(DATA/'domestic_seen_v1/freeze.json'),
        prepare_code_sha256=sha(ROOT/'pursuit/robust_stress.py'),
        reflection_delays_seconds=[.037,.089,.173,.281],reflection_gains=[.28,.17,.09,.05],
        attenuation_db=dict(quiet18=18,room_snr15_alt=6,room_snr5_alt=12),snr_db=dict(room_snr15_alt=15,room_snr5_alt=5),
        snr_definition='Whole recording RMS after reflections; event-local SNR is unknown',
        noise_components_seen_as_training_other=True,target_signal_recordings_training_seen=True,
        exact_transformation_combinations_not_in_training=True,independent_new_recordings=0,
        candidate_readouts_must_both_be_frozen_before_evaluation=True,models_or_scores_used_to_select_noise=False,
        actual_product_targets_verified=False))
    print('NEW_ACOUSTIC_COMBINATION_STRESS_PLAN_FROZEN',len(noise),flush=True)


def prepare():
    j=json.loads(PLAN.read_text());assert sha(ROOT/'pursuit/robust_stress.py')==j['prepare_code_sha256']
    assert sha(DATA/'domestic_seen_v1/freeze.json')==j['source_freeze_sha256']
    output=REPORT/'domestic_holdout_acoustic_features.json'
    if output.exists():raise FileExistsError('Preserve completed stress features')
    lookup={r['sample_id']:r for r in read_csv(DATA/'domestic_seen_v1/development.csv')};noise_rows=[lookup[k] for k in j['noise_source_ids']]
    rows=read_csv(DATA/'domestic_seen_v1/known_positive_probes.csv');native=NativeFrontend();receipts=[]
    for row in rows:
        a,m=pcm(row)
        for kind in KINDS:
            out=DEST/m['source_sha256']/kind;meta=out/'metadata.json'
            if meta.exists():
                r=json.loads(meta.read_text());assert r['plan_sha256']==sha(PLAN) and sha(out/'features.npy')==r['feature_sha256'];receipts.append(r);continue
            if out.exists():raise FileExistsError('Preserve partial transformation output')
            seed=int(hashlib.sha256(('heldout-combinations-v1:'+m['source_sha256']+kind).encode()).hexdigest()[:16],16);rng=np.random.default_rng(seed)
            b=a.copy();noise_record=None
            if kind!='quiet18':
                for delay,gain in zip(j['reflection_delays_seconds'],j['reflection_gains']):
                    lag=round(delay*16000)
                    if lag<len(a):b[lag:]+=gain*a[:-lag]
                selected=None
                for index in rng.permutation(len(noise_rows)):
                    nrow=noise_rows[int(index)];noise,nm=pcm(nrow);nrms=float(np.sqrt(np.mean(noise.astype(np.float64)**2)))
                    if nrms>1e-6:selected=(nrow,noise,nm);break
                assert selected is not None,'No nonzero registered ambient noise'
                nrow,noise,nm=selected;offset=int(rng.integers(len(noise)));noise=np.roll(noise,-offset)
                noise=np.tile(noise,int(np.ceil(len(a)/len(noise))))[:len(a)]
                nrms=float(np.sqrt(np.mean(noise.astype(np.float64)**2)));assert nrms>1e-8
                srms=max(float(np.sqrt(np.mean(b.astype(np.float64)**2))),1e-8)
                b+=noise*(srms/nrms)*10**(-j['snr_db'][kind]/20)
                noise_record=dict(sample_id=nrow['sample_id'],source_sha256=nm['source_sha256'],offset_samples=offset,snr_db=j['snr_db'][kind],role='train')
            b*=10**(-j['attenuation_db'][kind]/20);peak=float(np.abs(b).max())
            if peak>.98:b*=.98/peak
            q=np.clip(np.rint(b*32768),-32768,32767).astype(np.int16);out.mkdir(parents=True)
            pairs=list(windows(q));x=np.stack([native(np.ascontiguousarray(chunk)) for _,chunk in pairs]);assert x.shape[1:]==(98,40,1)
            with (out/'features.npy').open('xb') as f:np.save(f,x,allow_pickle=False)
            r=dict(sample_id=row['sample_id'],label=row['label'],kind=kind,source_sha256=m['source_sha256'],noise=noise_record,
                plan_sha256=sha(PLAN),feature_sha256=sha(out/'features.npy'),pcm_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                seed=seed,window_count=len(x),source_signals_and_noise_components_training_seen=True,independent_new_recordings=0)
            write_json(meta,r);receipts.append(r)
        print('NEW_COMBINATION_STRESS_FEATURES_PREPARED',row['sample_id'],flush=True)
    write_json(output,dict(complete=True,recordings=43,transformed_combinations=len(receipts),receipts=receipts,
        no_model_inference_yet=True,independent_new_recordings=0,actual_product_targets_verified=False))


def evaluate():
    from dscnn_baseline.common import runtime
    from dscnn_baseline.export import interpreter_for,predict_int8
    j=json.loads(PLAN.read_text());data=json.loads((REPORT/'domestic_holdout_acoustic_features.json').read_text());assert data['complete']
    paths={name:REPORT/'candidates'/name for name in CANDIDATES};identities={}
    for name,p in paths.items():
        assert (p/'model_int8.tflite').is_file() and (p/'delivered_output_contract.json').is_file(),'Both candidate readouts must exist before stress scoring'
        identities[name]=dict(model_sha256=sha(p/'model_int8.tflite'),readout_sha256=sha(p/'delivered_output_contract.json'))
    frozen=REPORT/'domestic_holdout_acoustic_evaluation_freeze.json'
    if frozen.exists():assert json.loads(frozen.read_text())['candidates']==identities
    else:write_json(frozen,dict(candidates=identities,plan_sha256=sha(PLAN),no_scores_used_to_freeze_readouts=True))
    tf,_=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    for name,p in paths.items():
        target=REPORT/'domestic_holdout_acoustic_results'/(name+'.json')
        if target.exists():
            assert json.loads(target.read_text())['identity']==identities[name];continue
        cfg=json.loads((p/'delivered_output_contract.json').read_text());assert cfg['output_semantics']=='Quantized softmax probabilities'
        interpreter=interpreter_for(tf,p/'model_int8.tflite');details=[]
        for r in data['receipts']:
            d=DEST/r['source_sha256']/r['kind'];assert sha(d/'features.npy')==r['feature_sha256']
            x=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False);prob=predict_int8(interpreter,x)
            q=np.rint(prob/cfg['output_scale']+cfg['output_zero_point']).astype(np.int16);arg=q.argmax(1)
            u=q.astype(np.int32)-cfg['output_zero_point'];hits=[]
            for k in (0,1):
                need,width=cfg['policy'][k];history=[];hit=False;cut=cfg['per_class_int8_minimum_q'][k]-cfg['output_zero_point']
                for n in range(len(q)):
                    history.append(int(u[n,k]) if arg[n]==k else 0);history=history[-width:]
                    if len(history)>=need and sorted(history,reverse=True)[need-1]>=cut:hit=True
                hits.append(hit)
            y=0 if r['label']=='fire_alarm' else 1
            details.append(dict(sample_id=r['sample_id'],condition=r['kind'],label=r['label'],detected=hits[y],wrong_target_alert=hits[1-y]))
        counts={}
        for condition in KINDS:
            counts[condition]={label:dict(files=sum(r['condition']==condition and r['label']==label for r in details),
                detected_files=sum(r['detected'] for r in details if r['condition']==condition and r['label']==label),
                wrong_target_alert_files=sum(r['wrong_target_alert'] for r in details if r['condition']==condition and r['label']==label)) for label in ['fire_alarm','doorbell']}
        write_json(target,dict(complete=True,identity=identities[name],conditions=counts,details=details,plan_sha256=sha(PLAN),
            signal_recordings_training_seen=True,noise_components_training_seen=True,transformation_combinations_not_used_in_training=True,
            thresholds_or_weights_refitted=False,independent_new_recordings=0,actual_device_or_event_reliability_proven=False))
        print('NEW_COMBINATION_FROZEN_CANDIDATE_STRESS_COMPLETE',name,counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['plan','prepare','evaluate']);a=p.parse_args();globals()[a.mode]()
