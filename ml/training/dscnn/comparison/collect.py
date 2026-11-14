"""Collect only licensed audio and public metadata; no software downloads.

Commons source metadata is frozen before any model fitting. Related same-author
target recordings are conservatively kept together. A test source is not a
guarantee of an unseen physical device or unseen foundation-model pretraining.
"""
import argparse
import hashlib
import html
import json
import re
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path
from .dataset import write_json,sha,read_csv,order
from .protocol import DATA,REPORT,ROOT

DEST=DATA/'new_public'
UA='HearingAlertResearch/1.0 (public educational audio dataset; no credentials)'
QUERIES={
 'fire_alarm':['filetype:audio "smoke alarm"','filetype:audio "fire alarm"','filetype:audio "smoke detector"','filetype:audio "evacuation alarm"','filetype:audio siren -pronunciation -intitle:LL-','filetype:audio "firealarm"'],
 'doorbell':['filetype:audio doorbell','filetype:audio "door bell"','filetype:audio "Türklingel"','filetype:audio "door chime"','filetype:audio "door buzzer"','filetype:audio intercom -pronunciation -intitle:LL-'],
 'other':['filetype:audio "clock alarm"','filetype:audio "church bells"','filetype:audio "frog"','filetype:audio "crying baby"','filetype:audio "washing machine"','filetype:audio "dog barking"','filetype:audio "rain"','filetype:audio "water drops"','filetype:audio "door creak"','filetype:audio "vacuum cleaner"']}


def clean(s):return ' '.join(html.unescape(re.sub('<[^>]+>',' ',s)).split())


def get(url,path,limit=15_000_000):
    if path.exists():return path.read_bytes()
    req=urllib.request.Request(url,headers={'User-Agent':UA})
    with urllib.request.urlopen(req,timeout=30) as h:b=h.read(limit+1)
    if len(b)>limit:raise ValueError('Public item exceeds size limit')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b)
    return b


def scout():
    records={};fail=[]
    for label,queries in QUERIES.items():
        for index,q in enumerate(queries):
            params=dict(action='query',format='json',generator='search',gsrsearch=q,gsrnamespace=6,gsrlimit=25,
                        prop='imageinfo',iiprop='url|extmetadata|size|sha1',iiurlwidth=0)
            url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
            try:
                value=json.loads(get(url,DEST/'pages'/f'{label}_search_{index}.json'))
                for p in value.get('query',{}).get('pages',{}).values():
                    info=p.get('imageinfo',[{}])[0];ext=info.get('extmetadata',{})
                    def m(k):return clean(ext.get(k,{}).get('value',''))
                    license=m('LicenseShortName')
                    if not (license.startswith('CC0') or license.startswith('CC BY') or license in ('Public domain','PD')):continue
                    media=info.get('url','')
                    if not re.search(r'\.(ogg|oga|wav|flac|mp3)(?:$|\?)',media,re.I):continue
                    if info.get('size',0)>15_000_000:continue
                    rec=dict(sample_id=f"commons-{p['pageid']}",source_dataset='WikimediaCommons',source_recording_id=str(p['pageid']),
                             title=p['title'],description=m('ImageDescription'),author=m('Artist'),license_id=license,
                             license_url=m('LicenseUrl'),source_url=info.get('descriptionurl'),media_url=media,
                             query_label=label,query=q,commons_sha1=info.get('sha1'),bytes_advertised=info.get('size'))
                    if rec['sample_id'] not in records:records[rec['sample_id']]=rec
                print('COMMONS_SCOUT',label,index,'unique',len(records),flush=True)
            except Exception as e:
                fail.append(dict(query=q,error=f'{type(e).__name__}: {str(e)[:160]}'))
                print('COMMONS_UNAVAILABLE',label,index,str(e)[:120],flush=True)
            write_json(DEST/'candidates.json',list(records.values()));write_json(DEST/'failures.json',fail)
            time.sleep(2.0)
    print('SCOUT_COMPLETE',len(records),flush=True)


def download():
    # reviewed_selection.json is an explicit metadata review, not automatic
    # acceptance of search hits as class labels.
    selected=json.loads((DEST/'reviewed_selection.json').read_text());accepted=[];fail=[]
    for rec in selected:
        try:
            suffix=Path(urllib.parse.urlparse(rec['media_url']).path).suffix
            p=DEST/'raw'/(rec['sample_id']+suffix)
            data=get(rec['media_url'],p)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=sample_rate,channels,codec_type','-of','json',str(p)],text=True))
            duration=float(probe['format']['duration'])
            if not .5<=duration<=180:raise ValueError('Duration outside frozen .5..180sec policy')
            pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-ac','1','-ar','16000','-f','s16le','-'])
            rec.update(relative_path=str(p.relative_to(ROOT)),audio_root='dscnn',source_sha256=hashlib.sha256(data).hexdigest(),
                       decoded_pcm_sha256=hashlib.sha256(pcm).hexdigest(),duration_s=duration,probe=probe,
                       label_basis='Reviewed public creator description; not human event timestamps',human_listening_verified=False,
                       physical_device_country_verified=False)
            accepted.append(rec);print('PUBLIC_AUDIO_SAVED',rec['sample_id'],rec['label'],round(duration,2),flush=True)
        except Exception as e:fail.append(dict(id=rec['sample_id'],error=f'{type(e).__name__}: {str(e)[:200]}'));print('PUBLIC_AUDIO_UNAVAILABLE',rec['sample_id'],str(e)[:100],flush=True)
        write_json(DEST/'downloads.json',accepted);write_json(DEST/'download_failures.json',fail)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scout','download']);a=p.parse_args()
    scout() if a.mode=='scout' else download()
