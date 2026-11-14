"""Fixed-candidate stress test on transformed TRAINING-SEEN KR recordings.

These transformations are not independent devices, households or activation
events. The candidate is frozen before scoring; no threshold is refitted.
"""
import json
import numpy as np
from comparison.dataset import read_csv,write_json,sha
from dscnn_baseline.common import runtime
from dscnn_baseline.export import interpreter_for,predict_int8
from .paths import DATA,REPORT
from .augmentation import DEST,KINDS
from .data import feature


def main():
    target=REPORT/'domestic_known_v1_stress.json'
    if target.exists():raise FileExistsError('Preserve completed candidate stress test')
    source=REPORT/'candidates/domestic_known_v1';cfg=json.loads((source/'delivered_output_contract.json').read_text())
    assert cfg['policy']==[[1,1],[1,1]] and cfg['output_semantics']=='Quantized softmax probabilities'
    model=source/'model_int8.tflite';before=sha(model)
    rows=read_csv(DATA/'domestic_seen_v1/known_positive_probes.csv');assert len(rows)==43
    tf,_=runtime('cpu');assert not tf.config.list_physical_devices('GPU');i=interpreter_for(tf,model)
    meta=json.loads((source/'model_metadata.json').read_text());scale=meta['output']['scale'];zero=meta['output']['zero_point']
    counts={};details=[]
    for kind in KINDS:
        flags={};wrong_flags={}
        for row in rows:
            original=feature(row);d=DEST/original['source_sha256']/kind;m=json.loads((d/'metadata.json').read_text())
            assert m['source_sha256']==original['source_sha256'] and sha(d/'features.npy')==m['feature_sha256']
            a=np.load(d/'features.npy',mmap_mode='r',allow_pickle=False);p=predict_int8(i,a);q=np.rint(p/scale+zero).astype(np.int16)
            y=int(row['label_id']);win=(q.argmax(1)==y)&(q[:,y]>=cfg['per_class_int8_minimum_q'][y]);hit=bool(win.any());flags[row['sample_id']]=hit
            wrong=bool(((q.argmax(1)==1-y)&(q[:,1-y]>=cfg['per_class_int8_minimum_q'][1-y])).any());wrong_flags[row['sample_id']]=wrong
            details.append(dict(sample_id=row['sample_id'],label=row['label'],condition=kind,detected=hit,
                transformed_feature_sha256=m['feature_sha256'],noise=m['noise'],wrong_target_alert=wrong,original_signal_training_seen=True))
        counts[kind]={label:dict(files=sum(r['label']==label for r in rows),detected_files=sum(flags[r['sample_id']] for r in rows if r['label']==label),
            wrong_target_alert_files=sum(wrong_flags[r['sample_id']] for r in rows if r['label']==label)) for label in ['fire_alarm','doorbell']}
        print('DOMESTIC_FROZEN_INT8_STRESS',kind,counts[kind],flush=True)
    assert sha(model)==before
    write_json(target,dict(complete=True,frozen_candidate_sha256=before,frozen_readout_contract_sha256=sha(source/'delivered_output_contract.json'),
        thresholds_or_weights_changed=False,source_files=43,independent_new_recordings=0,
        source_signals_training_seen=True,transforms_used_to_train_this_checkpoint=False,
        conditions=counts,details=details,actual_microphone_or_room_measurement=False,
        actual_event_or_monthly_false_rate_verified=False,fresh_public_test_recordings_used=False))


if __name__=='__main__':main()
