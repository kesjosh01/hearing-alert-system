"""Reconstruct checkpoint-time source exposure, verify against final receipts."""
import argparse
from collections import Counter
import json
from comparison.dataset import read_csv, write_json, sha
from comparison.sampling import Sampler
from .paths import ROOT


def main(name):
    run=ROOT/'runs'/name;target=run/'selected_checkpoint_exposure_audit.json'
    assert not target.exists(),'Preserve completed exposure evidence'
    status=json.loads((run/'status.json').read_text());assert status['complete']
    metadata=json.loads((run/'run.json').read_text());options=metadata['options']
    training=[r for r in read_csv(run/'used_train_selection.csv') if r['role']=='train']
    sampler=Sampler(training,None,options,42);selected=None
    for epoch in range(1,status['epochs_completed']+1):
        for _ in range(options['steps_per_epoch']):sampler.draw_ids(options['batch_size'])
        if epoch==status['selected_epoch']:selected=Counter(sampler.exposure)
    recorded={r['sample_id']:int(r['draws']) for r in read_csv(run/'sample_exposure.csv')}
    assert dict(sampler.exposure)=={k:v for k,v in recorded.items() if v},'Replay must match actual final per-file receipts'
    assert selected is not None
    negative=[r for r in training if r['label']=='other']
    def counts(exposure):
        return dict(source_draws=int(sum(exposure.values())),
            unique_training_views=sum(exposure[r['sample_id']]>0 for r in training),
            unique_negative_views=sum(exposure[r['sample_id']]>0 for r in negative),
            unique_negative_canonical_recordings=len({r['canonical_pcm_sha256'] for r in negative if exposure[r['sample_id']]>0}),
            negative_views_with_zero_draws=sum(exposure[r['sample_id']]==0 for r in negative))
    write_json(target,dict(passed=True,method='Exact seeded sampler replay checked against actual final exposure receipts',
        selected_epoch=status['selected_epoch'],epochs_completed=status['epochs_completed'],
        selected_checkpoint=counts(selected),entire_run=counts(sampler.exposure),
        negative_training_views=len(negative),virtual_views_are_not_independent_recordings=True,
        warm_start_checkpoint_exposure_not_included_here=True,model_sha256=status['model_sha256'],
        used_manifest_sha256=sha(run/'used_train_selection.csv'),sample_exposure_sha256=sha(run/'sample_exposure.csv'),
        no_audio_or_model_execution=True))
    print('SELECTED_CHECKPOINT_SOURCE_EXPOSURE_VERIFIED',name,counts(selected),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();main(a.name)
