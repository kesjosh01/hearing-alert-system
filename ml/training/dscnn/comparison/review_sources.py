"""Explicit metadata review decisions, made without any model outputs."""
import json
import re
from collections import Counter
from .protocol import DATA,ROOT
from .dataset import write_json,order

# Titles/descriptions were inspected. Pronunciation, music, toy/bird sirens,
# train door chimes and radio jammers are deliberately not target examples.
FIRE={'24001137','20677472','3609971','141394910','141403144','79609708','40908498','42335022','42335232','42335432',
      '24004664','5144549','5144512','167641949','185661894','201775378','1427051','92261750','125422336','74924781',
      '104788100','21056466','56977554','120502179','84694240','153645875'}
BELL={'271194','271198','271200','186656116','14901364','26597286','14901312','32225188'}
NEG={'134724081','134724143','68630861','115211201','165042285','175102150','191464708','39399939','113998677','3637775',
     '192411696','192411695','192411697','192411703','192411704','192411706','90733964','90733962','90733963','90671461',
     '132944356','59691730','80488535','80605142','80486737','143530664','148813517','28286431','187348064','6850011',
     '18894137','77270532','105337897','68269936','78351731','145219581','23994386','23994370','68472452',
     '23995777','122488707','39700237','37563052','42335551','23998631','23993702','23993930','151749700',
     '17851672','32225132','15686112','123378296','117113337','91866268','91866267','27758421','105338733','84042602',
     '77566693','103669344','108148278','198353653','131219663','9064155','89558050','24006188','42468076'}


def main():
    dest=DATA/'new_public';out=dest/'reviewed_selection.json'
    if out.exists():raise FileExistsError('Preserve source/role decisions made before fitting')
    candidates=json.loads((dest/'candidates.json').read_text());chosen=[]
    for r in candidates:
        sid=r['source_recording_id']
        if sid not in FIRE|BELL|NEG:continue
        label='fire_alarm' if sid in FIRE else 'doorbell' if sid in BELL else 'other'
        author=r['author'].lower().strip()
        # Unknown creator of three related old dingdong files: one family.
        if sid in {'271194','271198','271200'}:author='unknown-three-doorbell-series'
        if 'luis' in author and ('alvaz' in author or 'alvaz' in author.replace(' ','')):author='luisalvaz'
        key=re.sub(r'\W+','-',author)[:100] or 'unknown-'+sid
        synthesized=sid in {'24001137','141394910','141403144'}
        train_authors=['unknown-three-doorbell-series','are-shyde','techtonic','nallchen','seloloving','bogdankravhenko143452435','werstas']
        test_authors=['drb400atx','bidgee','32-fuß-freak','manfred','amada44','torsten-nilsson','holger','יורם','vrsooo']
        if synthesized or any(s in key for s in train_authors):role='train'
        elif any(s in key for s in test_authors):role='fresh_test'
        else:role='fresh_test' if int(order('creator:'+key)[:4],16)%3 else 'train'
        category='siren' if label=='fire_alarm' and sid not in {'24001137','20677472','3609971','141394910','141403144','40908498'} else r['query'].replace('filetype:audio ','').strip('"') if label=='other' else label
        r.update(label=label,label_id={'fire_alarm':0,'doorbell':1,'other':2}[label],role=role,
                 split='train' if role=='train' else 'test',target_domain='general',
                 sampler_pool='auxiliary_target' if label!='other' else 'overseas_confuser' if category in ('clock alarm','church bells','frog','crying baby') else 'overseas_general',
                 split_group_id='commons-creator-'+key,source_category=category,
                 origin='public_catalog_international',physical_recording_provenance='synthetic' if synthesized else 'creator_description; physical independence not fully verifiable',
                 review_basis='Manual title/creator-description eligibility review, no model predictions or human timestamp claims')
        chosen.append(r)
    # Same creator across event types must stay in one role. If any record is
    # explicitly train-only (e.g. synthesis), conservatively keep that creator
    # entirely in train rather than giving it a test role.
    roles={}
    for r in chosen:
        gid=r['split_group_id'];roles[gid]='train' if r['role']=='train' or roles.get(gid)=='train' else 'fresh_test'
    for r in chosen:
        r['role']=roles[r['split_group_id']];r['split']='train' if r['role']=='train' else 'test'
        r['creator_group_id']=r['split_group_id']
        r['split_group_id']+='-'+r['label']
    write_json(out,chosen)
    write_json(dest/'metadata_review.json',dict(selection_is_model_blind=True,decisions=Counter((r['role']+':'+r['label']) for r in chosen),
               excluded=[dict(id=r['sample_id'],title=r['title'],reason='Not selected: pronunciation, unrelated event, uncertain target identity, or outside reviewed inventory') for r in candidates if r['source_recording_id'] not in FIRE|BELL|NEG],
               human_audio_listening=False,physical_device_country_verified=False,source_grouping='All same-creator selected material stays in one role; related files never split'))
    print('REVIEWED',len(chosen),Counter((r['role'],r['label']) for r in chosen))


if __name__=='__main__':main()
