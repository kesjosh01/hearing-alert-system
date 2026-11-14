"""Schema-aware identity matching to original Google annotations, no scoring."""
import csv
import json
import re
from collections import defaultdict
from comparison.dataset import write_json,sha
from .paths import DATA,REPORT


def video_identity(index):
    m=re.fullmatch(r'fsx_yuchenhui_audioset_strong_processed_(train|eval)_([A-Za-z0-9_-]{11})',index)
    if not m:raise ValueError('Unrecognized source index; never infer a new test role')
    return m[1],m[2]


def originals(partition,wanted=None):
    result=defaultdict(lambda:defaultdict(list))
    with (DATA/'audioset_metadata'/f'audioset_{partition}_strong.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            video=r['segment_id'].rsplit('_',1)[0]
            if wanted is None or video in wanted:
                result[video][r['segment_id']].append((r['label'],float(r['start_time_seconds']),float(r['end_time_seconds'])))
    return result


def display_names():
    result={}
    with (DATA/'audioset_metadata/mid_to_display_name.tsv').open() as f:
        for line in f:
            cells=line.rstrip('\n').split('\t',1)
            if len(cells)==2 and cells[0].startswith('/'):result[cells[0]]=cells[1]
    return result


def match_row(row,original,names):
    partition,video=video_identity(row['index']);texts=row['raw_text'];mids=[];display=[]
    for text in texts:
        if text.startswith('Strong events: '):
            for part in text[len('Strong events: '):].split(' | '):
                m=re.fullmatch(r'\(([0-9.]+)s - ([0-9.]+)s\) .* \((/[^ )]+)\)',part)
                if not m:raise ValueError('Unparsed strong-event token')
                mids.append((m[3],m[1],m[2]))
        elif text.startswith('event: '):display.append(text[len('event: '):])
    if mids and display:raise ValueError('Mixed metadata schemas require separate audit')
    schema='three_decimal_mids' if mids else 'two_decimal_display_events' if display else 'no_supported_time_schema'
    matches=[]
    for segment,events in original.get(video,{}).items():
        if mids:
            expected=sorted((mid,f'{a:.3f}',f'{b:.3f}') for mid,a,b in events)
            valid=sorted(mids)==expected
        elif display:
            # Match the rendered source precision exactly, retaining original
            # 3-decimal times for any later dataset. No arbitrary tolerance.
            expected=sorted(f'{names[mid]} ({a:.2f}s-{b:.2f}s)' for mid,a,b in events)
            valid=sorted(display)==expected
        else:valid=False
        if valid:matches.append(segment)
    return dict(index=row['index'],source_partition=partition,video_id=video,schema=schema,
        matching_original_segments=matches,unique_original_identity_verified=len(matches)==1,
        original_annotations_not_shifted_or_edited=True)


def main():
    out=REPORT/'audioset_new_validation_probe';probe=json.loads((out/'mirror_partitions_metadata.json').read_text())
    rows=[r for item in probe['records'] for r in item['first_rows']]
    wanted={video_identity(r['index'])[1] for r in rows};names=display_names()
    source={s:originals(s,wanted) for s in ['train','eval']};results=[]
    for item in probe['records']:
        expected='eval' if item['publisher_partition']=='test' else 'train'
        for row in item['first_rows']:
            partition,_=video_identity(row['index']);assert partition==expected
            result=match_row(row,source[partition],names);assert result['unique_original_identity_verified'],result
            result['publisher_partition']=item['publisher_partition'];results.append(result)
    target=out/'sample_schema_aware_identity_resolution.json';assert not target.exists()
    write_json(target,dict(passed=True,sampled_rows=len(results),results=results,
        previous_gate_failure_sha256=sha(out/'initial_correspondence_gate_failure.json'),
        first_fault='The initial parser recognized only Strong events with MIDs; valid uses event display names rounded to 2 decimals',
        source_inputs_or_annotations_changed=False,no_arbitrary_matching_tolerance_added=True,
        all_partition_audio_or_source_creator_independence_certified=False,
        new_audio_or_model_predictions_created=False,code_sha256=sha(DATA.parents[1]/'pursuit/audioset_annotation_identity.py')))
    print('ORIGINAL_ANNOTATION_IDENTITY_PASS_WITH_EXPLICIT_SOURCE_SCHEMAS',results,flush=True)


if __name__=='__main__':main()
