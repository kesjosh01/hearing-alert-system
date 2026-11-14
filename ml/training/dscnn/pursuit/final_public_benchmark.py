"""One locked public recording benchmark, separate from all optimization loaders.

The original fresh-test feature gates stay closed. This module accepts only the
exact frozen final evaluation manifest and candidate identities. Features are
streamed in memory and cannot be reused by a fitting routine.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import subprocess
import time
import numpy as np
from comparison.dataset import read_csv, write_json, sha
from dscnn_baseline.common import runtime
from dscnn_baseline.frontend import windows
from .paths import ROOT, DATA, REPORT
from .data import audio_path, CACHE as NATIVE_CACHE
from .native import NativeFrontend, DIRECTORY
from .waveform_identity import DEST as IDENTITIES
from .yamnet_frontend import CONTRACT, transform
from .cpu_integer import validated_interpreter
from . import delivered_integer as di

NAMES=['domestic_known_v1','domestic_known_aug_v2','domestic_known_enriched_v3','general_prefix64_v1']
OUT=REPORT/'final_public_benchmark_v1'
MANIFEST=DATA/'round04_enriched_identity_audit/eligible_fresh_recordings.csv'
FREEZE=OUT/'freeze.json'


def identity_set(rows):
    result=dict(recordings=set(),groups=set(),creators=set(),pcm=set())
    for row in rows:
        result['recordings'].add((row['source_dataset'],row['source_recording_id']))
        result['groups'].add(row['split_group_id'])
        if row.get('creator_group_id'):result['creators'].add(row['creator_group_id'])
        raw=sha(audio_path(row));receipt=IDENTITIES/(raw+'.json')
        if not receipt.exists():
            assert row['role']!='fresh_test', 'Fresh identity must already have been audited'
            receipt=NATIVE_CACHE/raw/'metadata.json'
        meta=json.loads(receipt.read_text())
        assert raw==meta['source_sha256'];result['pcm'].add(meta['decoded_pcm_sha256'])
    return result


def freeze():
    OUT.mkdir(exist_ok=False)
    audit=REPORT/'round04_enriched_waveform_identity_audit.json';j=json.loads(audit.read_text())
    assert j['complete'] and j['eligible_manifest_sha256']==sha(MANIFEST)
    fresh=read_csv(MANIFEST);assert len(fresh)==1959 and all(r['role']=='fresh_test' for r in fresh)
    fids=identity_set(fresh)
    cohorts={'domestic_seen_v1','domestic_seen_enriched_v1','round04_enriched'}
    overlaps={}
    for name in sorted(cohorts):
        dev=read_csv(DATA/name/'development.csv');assert all(r['role']!='fresh_test' for r in dev)
        ids=identity_set(dev);overlaps[name]={k:len(v&fids[k]) for k,v in ids.items()}
        assert not any(overlaps[name].values()),(name,overlaps[name])
    candidates={}
    for name in NAMES:
        p=REPORT/'candidates'/name;c=json.loads((p/'candidate.json').read_text())
        assert not c['deployment_ready'] and not c['actual_product_targets_verified']
        assert all(sha(p/k)==v for k,v in c['files_sha256'].items())
        candidates[name]={k:sha(p/k) for k in ['model_int8.tflite','model_metadata.json','delivered_output_contract.json','candidate.json']}
    write_json(FREEZE,dict(created_at_unix=time.time(),names=NAMES,candidates=candidates,
        manifest_sha256=sha(MANIFEST),identity_audit_sha256=sha(audit),cohort_identity_overlaps=overlaps,
        code_sha256={p:sha(ROOT/p) for p in ['pursuit/final_public_benchmark.py','pursuit/delivered_integer.py',
            'pursuit/cpu_integer.py','pursuit/yamnet_frontend.py','pursuit/native.py']},
        native_binary_sha256=sha(DIRECTORY/'libpursuit_frontend.so'),frontend64=CONTRACT,
        class_counts=dict(Counter(r['label'] for r in fresh)),
        evaluation='One final fixed public-recording comparison; no changes from its outcomes',
        file_detection='At least one positive causal model window anywhere in recording; not activation-event recall',
        other_error='Only label=other recordings in denominator; known overseas target labels remain positive',
        wrong_target_confusions='Reported separately from eligible-other errors',
        bout_coverage='RealDESED published bell acoustic intervals only; any positive endpoint between onset and offset+1s',
        rising_edges='One-to-one greedy match of positive rising edges to published acoustic bouts with 1s offset tolerance; diagnostic only',
        acoustic_bouts_are_button_activations=False,independent_domestic_positive_recordings=0,
        all_physical_device_identity_or_foundation_pretraining_exposure_certified=False,
        actual_device_days=0,actual_product_targets_verified=False,no_test_based_model_or_threshold_tuning=True))
    print('FINAL_PUBLIC_RECORDING_BENCHMARK_LOCKED',len(fresh),overlaps,flush=True)


def checked_freeze():
    j=json.loads(FREEZE.read_text());assert j['names']==NAMES and sha(MANIFEST)==j['manifest_sha256']
    assert all(sha(ROOT/p)==v for p,v in j['code_sha256'].items())
    assert sha(DIRECTORY/'libpursuit_frontend.so')==j['native_binary_sha256']
    for name,files in j['candidates'].items():
        assert all(sha(REPORT/'candidates'/name/k)==v for k,v in files.items())
    return j


def bout_metrics(flags,starts,intervals):
    times=(np.asarray(starts,np.float64)+16000)/16000.
    active=np.asarray(flags,bool);edges=times[active & ~np.r_[False,active[:-1]]]
    coverage=sum(bool(np.any(active & (times>=a) & (times<=b+1.))) for a,b in intervals)
    unused=set(range(len(intervals)));matched=[];unmatched=[]
    for edge in edges:
        choices=[i for i in unused if intervals[i][0]<=edge<=intervals[i][1]+1.]
        if choices:
            i=min(choices,key=lambda k:intervals[k][1]);unused.remove(i)
            matched.append(dict(bout_index=i,edge_s=float(edge),window_endpoint_delay_from_onset_s=float(edge-intervals[i][0])))
        else:unmatched.append(float(edge))
    return dict(published_acoustic_bouts=len(intervals),bouts_with_any_positive_endpoint=int(coverage),
        positive_rising_edges=len(edges),matched_rising_edges=len(matched),unmatched_rising_edges=len(unmatched),
        matches=matched,unmatched_edge_times_s=unmatched,actual_button_activation_or_notification_metrics=False)


def summary(records,name):
    rows=[];values={}
    for r in records:
        row=dict(r['row'],role='fresh_test');rows.append(row);values[row['sample_id']]=np.array(r['models'][name]['maximum_scores'],np.int16)
    cfg=json.loads((REPORT/'candidates'/name/'delivered_output_contract.json').read_text());cuts=cuts_for(cfg)
    y=np.array([di.LABELS.index(r['label']) for r in rows]);maxima=np.stack([values[r['sample_id']] for r in rows])
    flags=(maxima>=cuts)&(maxima>0);classes={}
    for k,label in enumerate(di.LABELS[:2]):
        group=defaultdict(list)
        for i,r in enumerate(rows):
            if y[i]==k:group[r['split_group_id']].append(bool(flags[i,k]))
        classes[label]=dict(files=int((y==k).sum()),detected_files=int(flags[y==k,k].sum()),
            file_recall=float(flags[y==k,k].mean()),known_source_groups=len(group),
            source_mean_file_recall=float(np.mean([np.mean(v) for v in group.values()])),
            wrong_other_target_alert_files=int(flags[y==k,1-k].sum()))
    by_source={}
    for source in sorted({r['source_dataset'] for r in rows}):
        ids=np.array([r['source_dataset']==source for r in rows]);n=ids&(y==2)
        by_source[source]=dict(files=int(ids.sum()),other_files=int(n.sum()),
            other_false_alert_files=int(flags[n].any(1).sum()),
            targets={label:dict(files=int((ids&(y==k)).sum()),detected_files=int(flags[ids&(y==k),k].sum())) for k,label in enumerate(di.LABELS[:2])})
    bouts=[r['models'][name]['published_doorbell_bouts'] for r in records if 'published_doorbell_bouts' in r['models'][name]]
    return dict(classes=classes,other_files=int((y==2).sum()),other_false_alert_files=int(flags[y==2].any(1).sum()),
        other_false_alert_recording_fraction=float(flags[y==2].any(1).mean()),by_source=by_source,
        published_realdesed_doorbell_acoustic_bouts=dict(recordings=len(bouts),
            **{key:sum(b[key] for b in bouts) for key in ['published_acoustic_bouts','bouts_with_any_positive_endpoint',
                'positive_rising_edges','matched_rising_edges','unmatched_rising_edges']}),
        actual_activation_event_recall_verified=False,actual_notifications_or_monthly_false_rate_verified=False,
        actual_device_days=0,actual_product_targets_verified=False)


def cuts_for(cfg):
    if 'cuts_integer_units' in cfg:return np.array(cfg['cuts_integer_units'],np.int16)
    return np.array(cfg['per_class_int8_minimum_q'],np.int16)-cfg['output_zero_point']


def evaluate():
    frozen=checked_freeze();target=OUT/'results.json';assert not target.exists(),'Preserve the completed one-time benchmark'
    rows=read_csv(MANIFEST);tf,_=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    models={}
    for name in NAMES:
        p=REPORT/'candidates'/name;cfg=json.loads((p/'delivered_output_contract.json').read_text())
        rep_path=(REPORT/'general_delivered_v1/representative_training.npy' if name=='general_prefix64_v1' else
            REPORT/'domestic_enriched_delivered_v1/representative_training.npy' if name=='domestic_known_enriched_v3' else
            REPORT/'domestic_aug_delivered_v1/representative_training.npy')
        rep=np.load(rep_path,mmap_mode='r',allow_pickle=False)
        interpreter=validated_interpreter(tf,p/'model_int8.tflite',rep,OUT/(name+'_cpu_kernel_gate.json'),32)
        models[name]=(interpreter,cfg)
    native=NativeFrontend();records=[];(OUT/'recordings').mkdir(exist_ok=True);(OUT/'q_traces').mkdir(exist_ok=True)
    for i,row in enumerate(rows,1):
        assert row['role']=='fresh_test';record=OUT/'recordings'/(row['sample_id']+'.json')
        if record.exists():
            r=json.loads(record.read_text());assert r['complete'] and r['freeze_sha256']==sha(FREEZE)
            assert sha(OUT/'q_traces'/r['trace_filename'])==r['trace_sha256'];records.append(r);continue
        raw=audio_path(row);source=sha(raw);assert source==row['source_sha256']
        identity=json.loads((IDENTITIES/(source+'.json')).read_text())
        pcm_bytes=subprocess.check_output(['ffmpeg','-v','error','-nostdin','-i',str(raw),'-ac','1','-ar','16000',
            '-c:a','pcm_s16le','-f','s16le','-'],stderr=subprocess.PIPE)
        assert hashlib.sha256(pcm_bytes).hexdigest()==identity['decoded_pcm_sha256']
        pcm=np.frombuffer(pcm_bytes,dtype='<i2');assert len(pcm)==identity['sample_count']
        pairs=list(windows(pcm));starts=[s for s,_ in pairs]
        x40=np.stack([native(np.ascontiguousarray(chunk)) for _,chunk in pairs])
        wave64=np.stack([chunk[:15600].astype(np.float32)/32768. for _,chunk in pairs])
        with tf.device('/CPU:0'):x64=transform(tf,wave64).numpy()
        assert x40.shape[1:]==(98,40,1) and x64.shape[1:]==(96,64,1)
        outputs={};traces={}
        for name,(interpreter,cfg) in models.items():
            q=di.predict_q(interpreter,x64 if name=='general_prefix64_v1' else x40);traces[name]=q
            kind=cfg.get('output_kind','probability');scores=di.scores(q,kind,cfg['output_zero_point'],cfg['policy'])
            flags=(scores>=cuts_for(cfg))&(scores>0)
            outputs[name]=dict(maximum_scores=scores.max(0).tolist(),positive_window_counts=flags.sum(0).tolist())
            if row['source_dataset']=='RealDESED-20056072' and row['label']=='doorbell':
                intervals=json.loads(row['event_intervals_json']);assert intervals
                outputs[name]['published_doorbell_bouts']=bout_metrics(flags[:,1],starts,intervals)
        trace=OUT/'q_traces'/(row['sample_id']+'.npz');assert not trace.exists(),'Preserve partial trace and inspect interruption'
        np.savez_compressed(trace,starts=np.array(starts,np.int64),**traces)
        r=dict(complete=True,freeze_sha256=sha(FREEZE),source_sha256=source,decoded_pcm_sha256=identity['decoded_pcm_sha256'],
            sample_count=len(pcm),window_count=len(starts),row={k:row[k] for k in ['sample_id','label','source_dataset','split_group_id']},
            trace_filename=trace.name,trace_sha256=sha(trace),models=outputs)
        write_json(record,r);records.append(r)
        if i%50==0:print('FINAL_PUBLIC_BENCHMARK_RECORDINGS',i,len(rows),flush=True)
    checked_freeze()
    results={name:summary(records,name) for name in NAMES}
    write_json(target,dict(complete=True,freeze_sha256=sha(FREEZE),recordings=len(records),results=results,
        raw_audio_or_existing_model_modified=False,features_retained_for_training=False,
        test_outcomes_used_to_optimize=False,actual_product_targets_verified=False,actual_device_days=0))
    print('FINAL_FIXED_PUBLIC_RECORDING_BENCHMARK_COMPLETE',results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','evaluate']);a=p.parse_args()
    freeze() if a.mode=='freeze' else evaluate()
