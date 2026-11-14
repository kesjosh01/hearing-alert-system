"""Additional reviewed CC public catalog targets, obeying host rate limits."""
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from comparison.dataset import write_json,sha
from .paths import ROOT,DATA,REPORT


def main():
    out=DATA/'commons_additional';out.mkdir(exist_ok=False);(out/'raw').mkdir()
    prior=json.loads((ROOT/'data/comparison_20261114/new_public/accepted.json').read_text())
    seen={r['sample_id'] for r in prior};selected=json.loads((ROOT/'data/comparison_20261114/new_public/reviewed_selection.json').read_text())
    queue=[r for r in selected if r['label'] in ('fire_alarm','doorbell') and r['sample_id'] not in seen]
    write_json(out/'queue.json',dict(rows=queue,prior_metadata_review_reused=True,old_accepted_source_ids_excluded=True,
                time_of_scores='Additional sources not scored by models; future role audit required'))
    accepted=[];fail=[];rate_limited=False
    for r in queue:
        try:
            assert r['license_id'].startswith(('CC BY','CC0')) or r['license_id'] in ('Public domain','PD')
            u=r['media_url'];p=out/'raw'/(r['sample_id']+Path(urllib.parse.urlparse(u).path).suffix)
            req=urllib.request.Request(u,headers={'User-Agent':'HearingAlertResearch/1.0 (public educational audio collection)'})
            with urllib.request.urlopen(req,timeout=35) as h:
                b=h.read(15_000_001)
            if len(b)>15_000_000:raise ValueError('Advertised public clip exceeds bounded transport limit')
            p.write_bytes(b)
            duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(p)],text=True))
            if not .3<=duration<=300:raise ValueError('Target recording duration requires a separate long-recording policy')
            pcm=subprocess.check_output(['ffmpeg','-v','error','-i',str(p),'-ac','1','-ar','16000','-f','s16le','-'])
            accepted.append(dict(r,audio_root='dscnn',relative_path=str(p.relative_to(ROOT)),source_sha256=sha(p),
                    decoded_pcm_sha256=hashlib.sha256(pcm).hexdigest(),duration_s=duration,proposed_role='pending_source_audit',
                    human_listening_verified=False,event_intervals_verified=False,physical_device_country_verified=False))
            print('COMMONS_ADDITIONAL_SAVED',r['sample_id'],r['label'],round(duration,2),flush=True)
        except urllib.error.HTTPError as e:
            fail.append(dict(sample_id=r['sample_id'],http_status=e.code,retry_after=e.headers.get('Retry-After'),error=str(e)))
            if e.code in (429,403):rate_limited=True;print('COMMONS_HOST_STOP',e.code,'no retries or alternate routes',flush=True);break
        except Exception as e:fail.append(dict(sample_id=r['sample_id'],error=f'{type(e).__name__}: {e}'))
        finally:
            write_json(out/'accepted.json',accepted);write_json(out/'failures.json',fail)
        time.sleep(3.)
    write_json(REPORT/'commons_additional_status.json',dict(complete=not rate_limited,accepted=len(accepted),queue=len(queue),
             host_rate_or_access_stop=rate_limited,no_bypass_or_automatic_retry=True,other_collection_sources_continue=True))


if __name__=='__main__':main()
