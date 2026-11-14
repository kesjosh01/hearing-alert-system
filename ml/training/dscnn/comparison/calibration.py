"""Post-fit held-out calibration, separately for actual float and int8 outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.export import export_int8,interpreter_for,predict_int8
from .dataset import write_json,write_csv,sha
from .protocol import REPORT
from .features import rows,load
from .train import predict
from .metrics import LABELS,calibrate,assess,apply,pool


def calibrate_run(run):
    run=Path(run).resolve();out=REPORT/'calibration'/run.name
    if out.exists():raise FileExistsError('Do not repeatedly tune on calibration data')
    out.mkdir(parents=True);spec=json.loads((run/'run.json').read_text());o=spec['options'];status=json.loads((run/'status.json').read_text())
    assert status['complete'] and status['exit_code']==0
    rr=rows(o['added_training_data'],roles=('train','calibration'));train=[r for r in rr if r['role']=='train'];cal=[r for r in rr if r['role']=='calibration']
    assert set(r['split_group_id'] for r in train).isdisjoint(r['split_group_id'] for r in cal)
    features,receipt=load(rr);tf,keras=runtime('gpu')
    with tf.device('/GPU:0'):model=keras.models.load_model(str(run/'classifier.keras'),compile=False)
    initial=json.loads((run/'selection_policy.json').read_text())['thresholds'];old=tuple(initial[k] for k in LABELS[:2])
    with tf.device('/GPU:0'):floating=predict(model,cal,features,o['pooling_fraction'])
    ft,fm=calibrate(floating);before=assess(floating,old)
    rng=np.random.default_rng(20261114);representative=[];prov=[]
    for label in LABELS:
        members=[r for r in train if r['label']==label]
        for _ in range(100):
            r=members[int(rng.integers(len(members)))];a=features[r['sample_id']];i=int(rng.integers(len(a)))
            representative.append(a[i]);prov.append(dict(split='train',sample_id=r['sample_id'],split_group_id=r['split_group_id'],window_index=i))
    integer_path,export=export_int8(tf,model,np.asarray(representative),run/'export',prov)
    integer=interpreter_for(tf,integer_path);quantized=[]
    for r in cal:
        p=predict_int8(integer,features[r['sample_id']]);q=pool(p,o['pooling_fraction'])
        quantized.append(dict(r,**{'p_'+k:float(v) for k,v in zip(LABELS,q)},patch_count=len(p)))
    it,im=calibrate(quantized)
    write_csv(out/'float_raw.csv',floating);write_csv(out/'int8_raw.csv',quantized)
    write_csv(out/'float_clips.csv',apply(floating,ft));write_csv(out/'int8_clips.csv',apply(quantized,it))
    write_json(out/'results.json',dict(before_float_selection_threshold_transfer=before,after_float_calibration=fm,after_int8_calibration=im,
               float_thresholds=dict(zip(LABELS[:2],ft)),int8_thresholds=dict(zip(LABELS[:2],it)),model_sha256=sha(run/'classifier.keras'),int8_sha256=sha(integer_path),
               model_settings_frozen_before_calibration=True,calibration_used_to_rank_models=False,fresh_test_used=False,
               quantization_representative_role='train',decision_calibration_role='calibration',calibration_groups=len({r['split_group_id'] for r in cal}),export=export))
    write_json(run/'calibrated_policy.json',dict(float=dict(zip(LABELS[:2],ft)),int8=dict(zip(LABELS[:2],it)),pooling_fraction=o['pooling_fraction'],source=str(out/'results.json')))
    print('CALIBRATION_COMPLETE',run.name,'before',before['foreign_fpr'],'after_float',fm['foreign_fpr'],'int8',im['foreign_fpr'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();calibrate_run(a.run)
