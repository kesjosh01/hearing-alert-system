"""Choose causal policy on selection only; calibrate it on held-out sources."""
import json
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime
from dscnn_baseline.export import interpreter_for,predict_int8
from expansion.data import ExpandedFeatureStore
from .protocol import REPORT
from .dataset import write_json,sha
from .features import rows,load
from .train import predict
from .temporal import BASE_POLICY,calibrate,rank,added_delay


def main():
    out=REPORT/'temporal_replay'
    if out.exists():raise FileExistsError('Preserve replay selection and calibration')
    out.mkdir();frozen=json.loads((REPORT/'final_candidate.json').read_text());run=Path(frozen['champion']['run'])
    assert sha(run/'classifier.keras')==frozen['model_sha256']
    options=json.loads((run/'run.json').read_text())['options'];rr=rows(options['added_training_data'],roles=('selection','calibration'));features,_=load(rr)
    tf,keras=runtime('gpu')
    with tf.device('/GPU:0'):model=keras.models.load_model(str(run/'classifier.keras'),compile=False)
    _,window_float=predict(model,rr,features,return_windows=True)
    store=ExpandedFeatureStore();durations={r['sample_id']:store.load(r)[1]['samples']/16000 for r in rr}
    selection=[r for r in rr if r['role']=='selection'];cal=[r for r in rr if r['role']=='calibration'];policy=dict(BASE_POLICY)
    thresholds,m=calibrate(selection,window_float,durations,policy);ledger=[dict(factor='baseline',policy=dict(policy),thresholds=thresholds,selection=m,adopted=True)]
    for factor,value in [('vote_rule','two_of_three'),('release_ratio',.8),('cooldown_s',2.)]:
        candidate=dict(policy,**{factor:value});t,metric=calibrate(selection,window_float,durations,candidate);adopt=rank(metric)>rank(m)
        ledger.append(dict(factor=factor,policy=candidate,thresholds=t,selection=metric,adopted=adopt,
                           added_delay_at_same_thresholds=added_delay(selection,window_float,t,candidate,durations)))
        if adopt:policy,thresholds,m=candidate,t,metric
    write_json(out/'selection.json',dict(ledger=ledger,chosen_policy=policy,selection_only=True,calibration_or_test_used_to_select=False))
    ft,fm=calibrate(cal,window_float,durations,policy)
    integer=interpreter_for(tf,run/'export/dscnn_int8.tflite');window_int8={r['sample_id']:predict_int8(integer,features[r['sample_id']]) for r in cal}
    it,im=calibrate(cal,window_int8,durations,policy)
    write_json(out/'calibration.json',dict(float=fm,int8=im,float_thresholds=ft,int8_thresholds=it,policy=policy,
               float_added_delay=added_delay(cal,window_float,ft,policy,durations),int8_added_delay=added_delay(cal,window_int8,it,policy,durations),
               exact_event_onset_latency_not_measured=True,no_microphone_or_board_execution=True,
               zero_padding_and_recording_resets='Offline files follow existing one-second right-zero-pad frontend; each file resets policy state. Field alarm rate unverified.'))
    np.savez(out/'selection_float_windows.npz',**{r['sample_id']:window_float[r['sample_id']] for r in selection})
    np.savez(out/'calibration_float_windows.npz',**{r['sample_id']:window_float[r['sample_id']] for r in cal})
    np.savez(out/'calibration_int8_windows.npz',**window_int8)
    write_json(out/'frozen_policy.json',dict(policy=policy,float_thresholds=ft,int8_thresholds=it,model_sha256=sha(run/'classifier.keras'),int8_sha256=sha(run/'export/dscnn_int8.tflite'),
               source=str(out/'calibration.json'),fresh_test_used=False))
    print('CAUSAL_REPLAY_POLICY_FROZEN',policy,'calibration_int8_recall',im['target_macro_group_recall'],'fpr',im['foreign_fpr'],flush=True)


if __name__=='__main__':main()
