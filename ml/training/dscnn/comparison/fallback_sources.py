"""Additional public audio hosts; never retry Wikimedia rate-limited items."""
import argparse
import html
import importlib.util
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from .protocol import DATA,ROOT
from .dataset import write_json
from .collect import download as download_reviewed

DEST=DATA/'new_public'


def bsb_module():
    spec=importlib.util.spec_from_file_location('public_audio_existing',ROOT/'tools/collect_public_audio.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.DEST=DEST/'bsb';m.PAGES=m.DEST/'pages';m.PAGES.mkdir(parents=True,exist_ok=True)
    return m


def scout():
    m=bsb_module();records={};fail=[]
    for term in ['clock alarm','frog','baby crying','church bells','washing machine','dog barking','rain','door creaking']:
        try:
            for url,title in m.bsb_catalog(term).items():records[url]=dict(url=url,title=title,query=term)
        except Exception as e:fail.append(dict(query=term,error=str(e)[:150]))
        write_json(DEST/'bsb_candidates.json',list(records.values()));write_json(DEST/'bsb_scout_failures.json',fail)
        print('BSB_SCOUT',term,len(records),flush=True);time.sleep(1.)


def soundbible():
    items=[('2022-School-Fire-Alarm','fire_alarm','Cullen Card','fresh_test','fire_alarm'),
           ('165-Door-Bell','doorbell','Mike Koenig','fresh_test','doorbell'),
           ('1599-Store-Door-Chime','doorbell','Mike Koenig','fresh_test','doorbell'),
           ('1937-Tornado-Siren-II','fire_alarm','Delilah','fresh_test','siren'),
           ('2160-Old-Fashion-Door-Bell','doorbell','Daniel Simion','fresh_test','doorbell')]
    accepted=[];fail=[];m=bsb_module()
    for slug,label,author,role,category in items:
        url='https://soundbible.com/'+slug+'.html';path=DEST/'pages'/('soundbible_'+slug.split('-')[0]+'.html')
        try:
            if path.exists():page=path.read_text()
            else:
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=25) as h:b=h.read(4_000_000)
                path.write_bytes(b);page=b.decode(errors='replace')
            if 'Attribution 3.0' not in page:raise ValueError('Item-specific attribution license missing')
            medias=re.findall(r'data-source=["\']([^"\']+\.mp3)["\']',page)
            if not medias:raise ValueError('No advertised MP3 source')
            media=urllib.parse.quote(urllib.parse.urljoin(url,html.unescape(medias[0])),safe=':/?&=%')
            sid=slug.split('-')[0];key=re.sub(r'\W+','-',author.lower())
            r=dict(sample_id='soundbible-'+sid,source_dataset='SoundBible',source_recording_id=sid,title=slug,
                   source_url=url,media_url=media,license_id='CC-BY-3.0',license_url='https://creativecommons.org/licenses/by/3.0/',author=author,
                   label=label,label_id={'fire_alarm':0,'doorbell':1}[label],role=role,split='test',
                   creator_group_id='soundbible-'+key,split_group_id='soundbible-'+key+'-'+label,
                   sampler_pool='auxiliary_target',target_domain='general',source_category=category,
                   source_group_basis='All same-author related files stay in test; distinct class groups',
                   physical_recording_provenance='creator description; store chime is derived/synthetic',origin='overseas_public_catalog')
            r=m.download(r);r['human_listening_verified']=False;accepted.append(r)
            print('SOUNDBIBLE_SAVED',sid,round(r['duration_s'],2),flush=True)
        except Exception as e:fail.append(dict(url=url,error=f'{type(e).__name__}: {str(e)[:180]}'));print('SOUNDBIBLE_UNAVAILABLE',slug,str(e)[:90],flush=True)
        write_json(DEST/'soundbible_downloads.json',accepted);write_json(DEST/'soundbible_failures.json',fail);time.sleep(1.)


def bsb_download():
    m=bsb_module();records=json.loads((DEST/'bsb_candidates.json').read_text())
    # Frozen, model-blind family assignment. Unrelated series from the same
    # author are distinct recording groups; related durations stay together.
    families={
      'clock-mechanical':('fresh_test','clock_alarm',{'1372','1373','1374','1375','2655','2656','2657','2658','2659','2660','2661','2662','2663','2811','2812','2813','2814','2815','2816'}),
      'frog-pond':('train','frog',{'0997','0998','0819','0691'}),
      'toad':('fresh_test','frog',{'1053','1054','1665','1666'}),
      'baby-family':('fresh_test','crying_baby',{'0233','0873','0962','0877','0881','0883','0876','0963'}),
      'church-group':('fresh_test','church_bells',{'0135','3446','3445','0830','0994','0829','0107'}),
      'washing':('train','washing_machine',{'0106','0809','0423'}),
      'dog-new':('train','dog',{'2954','2955','3537'}),
      'dog-old':('fresh_test','dog',{'2352','2353','2354','0916','0112','0682','0612'}),
      'dog-pack':('train','dog',{'1544','1545','1546','1060','1061','0868'}),
      'rain-metal':('fresh_test','rain',{'1290','1289','1295','1293','1294','1296','1297','1292','1291'}),
      'rain-field':('train','rain',{'0740','1019','2719','0124','0555','2679','0621','0820','2717'}),
      'creak':('fresh_test','door_wood_creaks',{'3211','3205','3207','3206','3210','3208','3209','0302','0306','0305','0539'}),
      'apartment-door':('train','door_wood_creaks',{'1702','1703','1704','1705','1706'}),
      'metal-gate':('train','door_wood_creaks',{'0303','0304','0616'})}
    # Additional true doorbells: exclude all old-telephone "old doorbell" items
    # and every known previous mechanical-entrance/electronic-house series.
    records += [dict(url='https://bigsoundbank.com/industrial-doorbell-s1269.html',title='Industrial Doorbell #1')]
    for n in range(1922,1926):records.append(dict(url=f'https://bigsoundbank.com/industrial-doorbell-{n-1920}-s{n}.html',title='Industrial doorbell series'))
    for n in range(2373,2376):records.append(dict(url=f'https://bigsoundbank.com/intercom-ringtone-{n-2372}-s{n}.html',title='Intercom ringtone series'))
    families['industrial-bell']=('fresh_test','doorbell',{'1269','1922','1923','1924','1925'})
    families['intercom-new']=('train','doorbell',{'2373','2374','2375'})
    lookup={sid:(family,role,cat) for family,(role,cat,ids) in families.items() for sid in ids}
    existing=json.loads((ROOT/'data/expansion_20261114/refit.csv').read_text()) if False else []
    accepted=[];fail=[];seen=set()
    for rec in records:
        sid=re.search(r'-s(\d+)\.html',rec['url'])[1]
        if sid not in lookup or sid in seen:continue
        seen.add(sid);family,role,cat=lookup[sid]
        try:
            r=m.bsb_detail(rec['url']);r['sample_id']='bsb-comparison-'+sid
            if cat=='clock_alarm' and 'alarm clock' not in r['title'].lower():raise ValueError('Clock identity not explicit')
            if cat=='doorbell' and not any(k in r['title'].lower() for k in ('doorbell','intercom')):raise ValueError('Doorbell identity not explicit')
            label='doorbell' if cat=='doorbell' else 'other'
            r.update(label=label,label_id=1 if label=='doorbell' else 2,role=role,split='train' if role=='train' else 'test',
                     split_group_id='bsb-new-recording-family-'+family,source_category=cat,target_domain='general',
                     sampler_pool='auxiliary_target' if label=='doorbell' else 'overseas_confuser' if cat in ('clock_alarm','door_wood_creaks') else 'overseas_general',
                     source_group_basis='Conservative related recording/device series, same author can record unrelated devices',
                     physical_recording_provenance='Publisher item description; physical independence cannot be fully certified')
            r=m.download(r);r['label_basis']='Reviewed publisher recording title/category';accepted.append(r)
            print('BSB_NEW_SAVED',sid,role,label,round(r['duration_s'],2),flush=True)
        except Exception as e:fail.append(dict(id=sid,error=f'{type(e).__name__}: {str(e)[:180]}'));print('BSB_NEW_UNAVAILABLE',sid,str(e)[:110],flush=True)
        write_json(DEST/'bsb_downloads.json',accepted);write_json(DEST/'bsb_download_failures.json',fail);time.sleep(.6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scout','soundbible','bsb_download']);a=p.parse_args()
    {'scout':scout,'soundbible':soundbible,'bsb_download':bsb_download}[a.mode]()
