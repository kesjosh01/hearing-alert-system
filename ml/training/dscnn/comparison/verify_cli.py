"""Check the packaged user CLI against frozen evaluation outputs, on CPU."""
import json
import subprocess
from pathlib import Path
import numpy as np
from .protocol import ROOT,REPORT,DATA
from .dataset import read_csv,write_json,sha
from .metrics import LABELS
from .temporal import events,frame_times


def main():
    out=REPORT/'cli_verification.json'
    if out.exists():raise FileExistsError('Preserve CLI verification')
    bundle=Path(json.loads((REPORT/'packaging.json').read_text())['bundle'])
    rr=read_csv(DATA/'fresh_test.csv');pred=read_csv(REPORT/'evaluation/fresh_test/final/int8_clips.csv');ids={r['sample_id']:r for r in rr}
    # Smallest already-evaluated file per label keeps this interface check
    # bounded. This never selects a model or adjusts thresholds.
    samples=[min((r for r in pred if r['label']==label),key=lambda r:int(r['patch_count'])) for label in LABELS]
    windows=np.load(REPORT/'evaluation/fresh_final_int8_windows.npz',allow_pickle=False)
    policy=json.loads((bundle/'temporal_policy.json').read_text());records=[]
    for saved in samples:
        row=ids[saved['sample_id']];audio=ROOT/row['relative_path']
        for mode in ('clip','replay'):
            cmd=[str(ROOT/'.venv/bin/python'),'-B','-m','comparison.infer','--bundle',str(bundle),'--audio',str(audio),'--backend','int8','--mode',mode]
            result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
            if result.returncode:raise RuntimeError('CLI first failure: '+result.stderr[-2000:])
            actual=json.loads(result.stdout)
            diff=max(abs(actual['probabilities'][k]-float(saved['p_'+k])) for k in LABELS)
            assert actual['label']==saved['predicted_label'] and diff<1e-6
            assert actual['window_count']==int(saved['patch_count'])
            if mode=='replay':
                w=windows[row['sample_id']];expected=events(w,policy['int8_thresholds'],policy['policy'],frame_times(actual['duration_s'],len(w)))
                got=[[],[]]
                for alert in actual['alerts']:got[LABELS.index(alert['label'])].append(alert['window_end_s'])
                assert got==expected,'Causal CLI differs from frozen replay'
            records.append(dict(sample_id=row['sample_id'],backend='int8',mode=mode,max_probability_abs_difference=diff,decision_matches=True,windows=actual['window_count']))
    sample=samples[0];row=ids[sample['sample_id']]
    result=subprocess.run([str(ROOT/'.venv/bin/python'),'-B','-m','comparison.infer','--bundle',str(bundle),'--audio',str(ROOT/row['relative_path']),'--backend','float'],cwd=ROOT,text=True,capture_output=True)
    if result.returncode:raise RuntimeError('Float CLI first failure: '+result.stderr[-2000:])
    actual=json.loads(result.stdout);saved=next(r for r in read_csv(REPORT/'evaluation/fresh_test/final/float_clips.csv') if r['sample_id']==row['sample_id'])
    diff=max(abs(actual['probabilities'][k]-float(saved['p_'+k])) for k in LABELS)
    assert diff<2e-4 and actual['label']==saved['predicted_label']
    records.append(dict(sample_id=row['sample_id'],backend='float',mode='clip',max_probability_abs_difference=diff,decision_matches=True))
    windows.close()
    write_json(out,dict(passed=True,cases=records,backend='CPU only; existing venv; no installs or microphone',
                       final_model_sha256=sha(bundle/'classifier.keras'),test_driven_reselection=False,
                       purpose='User CLI integration and actual frontend/aggregation parity, not a new performance estimate'))
    print('COMPARISON_CLI_PARITY_PASS',len(records),flush=True)


if __name__=='__main__':main()
