"""Recover selected-checkpoint exposure from the deterministic sampler schedule."""
import argparse
import json
from pathlib import Path
from .sampling import Sampler
from .dataset import read_csv,write_csv,write_json


def audit(run):
    run=Path(run);meta=json.loads((run/'run.json').read_text());status=json.loads((run/'status.json').read_text())
    assert status['complete'] and status['exit_code']==0
    rr=[r for r in read_csv(run/'used_train_selection.csv') if r['role']=='train']
    sampler=Sampler(rr,{},meta['options'],meta['spec'].get('seed',42));selected=None
    for epoch in range(1,status['epochs_completed']+1):
        for _ in range(meta['options']['steps_per_epoch']):sampler.draw_ids(meta['options']['batch_size'])
        if epoch==status['selected_epoch']:selected=dict(sampler.exposure)
    actual={r['sample_id']:int(r['draws']) for r in read_csv(run/'sample_exposure.csv')}
    assert all(actual.get(r['sample_id'],0)==sampler.exposure[r['sample_id']] for r in rr),'Sampler replay does not match actual full-run logs'
    assert selected is not None
    records=[dict(sample_id=r['sample_id'],label=r['label'],split_group_id=r['split_group_id'],draws=selected.get(r['sample_id'],0)) for r in rr]
    write_csv(run/'selected_checkpoint_exposure.csv',records)
    summary={label:dict(total=sum(r['label']==label for r in records),sampled=sum(r['label']==label and r['draws']>0 for r in records),draws=sum(r['draws'] for r in records if r['label']==label)) for label in ('fire_alarm','doorbell','other')}
    receipt=dict(selected_epoch=status['selected_epoch'],epochs_executed=status['epochs_completed'],selected_checkpoint=summary,
                 deterministic_sampling_replay_matches_logged_full_run_exactly=True,
                 reconstruction_only=True,model_weights_changed=False,note='Full-run exposure includes optimizer steps after the saved checkpoint. This audit explicitly stops at the saved epoch.')
    write_json(run/'selected_checkpoint_exposure_audit.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(audit(a.run)))
