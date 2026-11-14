"""Training-only int8 calibration, comparing softmax and raw-logit output.

These are local research artifacts. Raw-logit output requires an explicitly
different readout contract and must not be substituted into probability-based
firmware. No fresh test or actual board execution occurs here.
"""
import argparse
import json
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.export import export_int8,interpreter_for,predict_int8
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT
from .data import feature,CACHE
from .operating import measure,key


def main(name):
    run=ROOT/'runs'/name;status=json.loads((run/'status.json').read_text());assert status['complete'] and not status.get('target_results_training_seen',False)
    j=json.loads((run/'run.json').read_text());dataset=j['spec']['dataset'];d=DATA/dataset;freeze=json.loads((d/'freeze.json').read_text());assert all(sha(d/k)==v for k,v in freeze['files'].items())
    rows=read_csv(d/'development.csv');training=[r for r in rows if r['role']=='train'];selection=[r for r in rows if r['role']=='selection']
    out=REPORT/'int8_readout_trials'/name;out.mkdir(parents=True,exist_ok=False);representative=[];provenance=[];rng=np.random.default_rng(42)
    for label in ['fire_alarm','doorbell','other']:
        rr=[r for r in training if r['label']==label];rr=[rr[i] for i in rng.permutation(len(rr))[:64]]
        for row in rr:
            m=feature(row);a=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
            for i in np.unique(np.linspace(0,len(a)-1,min(len(a),2)).astype(int)):
                representative.append(a[i]);provenance.append(dict(split='train',sample_id=row['sample_id'],source_sha256=m['source_sha256'],window_index=int(i),start_sample=m['starts'][i]))
    x=np.stack(representative);assert x.shape[1:]==(98,40,1)
    tf,keras=runtime('cpu');assert not tf.config.list_physical_devices('GPU')
    model_path=run/'classifier.keras';mh=sha(model_path);assert mh==status['model_sha256']
    with tf.device('/CPU:0'):model=keras.models.load_model(str(model_path),compile=False);logits=keras.Model(model.input,model.get_layer('class_logits').output)
    ledger=[]
    for kind,net in [('probability',model),('logits',logits)]:
        directory=out/kind;path,meta=export_int8(tf,net,x,directory,provenance,pipeline_only=False)
        if kind=='logits':
            newpath=path.with_name('dscnn_logits_int8.tflite');path.rename(newpath);path=newpath
            old_header=directory/'dscnn_int8.h';new_header=directory/'dscnn_logits_int8.h';old_header.rename(new_header)
            new_header.write_text('// OUTPUTS ARE RAW LOGITS: requires argmax/logit-margin readout, NOT probability thresholds.\n'+new_header.read_text())
        meta.update(output_semantics=kind,artifact_filename=path.name,source_trained_model_sha256=mh,training_data_freeze_sha256=sha(d/'freeze.json'),
            deployment_ready=False,actual_board_or_microphone_tested=False,not_for_probability_firmware_without_output_contract_change=kind=='logits')
        write_json(directory/'model_metadata.json',meta)
        interpreter=interpreter_for(tf,path);integer={};float_logits={};saturated=0;windows=0
        for row in selection:
            m=feature(row);a=np.load(CACHE/m['source_sha256']/'features.npy',mmap_mode='r',allow_pickle=False)
            integer[row['sample_id']]=predict_int8(interpreter,a)
            if kind=='logits':
                with tf.device('/CPU:0'):float_logits[row['sample_id']]=np.concatenate([logits(a[i:i+128],training=False).numpy() for i in range(0,len(a),128)])
            values=integer[row['sample_id']];saturated+=int((values.max(1)>=meta['output']['scale']*(127-meta['output']['zero_point'])).sum());windows+=len(values)
        archive=directory/'selection_outputs.npz';np.savez(archive,**integer);results=[]
        if kind=='logits':
            for fire in [(1,1),(2,3)]:
                for bell in [(1,1),(2,3)]:results.append(measure(selection,integer,mode='margin',policy=(fire,bell)))
            max_error=max(float(np.max(np.abs(integer[k]-float_logits[k]))) for k in integer)
            argmax_disagreements=sum(int((integer[k].argmax(1)!=float_logits[k].argmax(1)).sum()) for k in integer)
            selected=max(results,key=key)
        else:
            # The dequantized probability tensor is the delivered output;
            # convert to log probabilities solely to reuse the metric scorer.
            z={k:np.log(np.maximum(v,1e-12)) for k,v in integer.items()}
            for fire in [(1,1),(2,3)]:
                for bell in [(1,1),(2,3)]:results.append(measure(selection,z,mode='probability',policy=(fire,bell)))
            selected=max(results,key=key);max_error=None;argmax_disagreements=None
        record=dict(kind=kind,model_sha256=sha(path),metadata=meta,selection_best=selected,selection_results=results,
             output_saturation_windows=saturated,selection_windows=windows,max_float_logit_absolute_error=max_error,argmax_disagreements=argmax_disagreements,
             outputs_sha256=sha(archive),selection_only=True,calibration_holdout_or_fresh_test_used=False,product_targets_verified=False)
        write_json(directory/'selection_analysis.json',record);ledger.append(record);print('INT8_READOUT_SELECTION',kind,selected,flush=True)
    assert sha(model_path)==mh
    write_json(out/'status.json',dict(complete=True,source_model=name,source_model_sha256=mh,representative_windows=len(x),representative_role='train',
        output_modes=[r['kind'] for r in ledger],board_or_product_validation=False,results=ledger))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();main(a.name)
