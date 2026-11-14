"""Local audio review: machine suggestions are never approved annotations."""
from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np

from .common import ROOT, YAMNET, new_directory, runtime, source, write_json
from .data import (ANNOTATION_FIELDS, development_identity, development_splits,
                   positive_window_indices, read_annotations)
from .frontend import window_starts


def merge_intervals(intervals, duration, gap=0.15):
    result = []
    for start, end in sorted(intervals):
        start, end = max(0.0, float(start)), min(duration, float(end))
        if start >= end:
            continue
        if result and start <= result[-1][1] + gap:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [[round(a, 3), round(b, 3)] for a, b in result]


def prepare(output, device="cpu"):
    rows = sorted((r for r in development_splits()[0] if r["label"] != "other"),
                  key=lambda r: (r["label"], r["sample_id"]))
    directory = new_directory(output)
    (directory / "audio").mkdir()
    status = {"kind": "human_review_candidates_only", "human_review_complete": False,
              "training_authorized_by_this_file": False, "test_audio_read": 0, "complete": False}
    write_json(directory / "status.json", status)
    from sound_baseline.features import decode_audio
    tf, keras = runtime(device)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    import tensorflow_hub as hub
    teacher_path = YAMNET / "runs/yamnet/ubuntu_gpu_refit_w1_s42/classifier.keras"
    yamnet_path = YAMNET / ".cache/final_refit_official221_s42_v1/tensorflow/tfhub/9616fd04ec2360621642ef9455b84f4b668e219e"
    teacher = keras.models.load_model(str(teacher_path), compile=False)
    backbone = hub.load(str(yamnet_path))
    clips = []
    for number, row in enumerate(rows, 1):
        relative = f"audio/{row['sample_id']}.wav"
        decode_audio(YAMNET / row["relative_path"], directory / relative)
        with wave.open(str(directory / relative)) as handle:
            pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
        waveform = pcm.astype(np.float32) / 32768
        duration = len(pcm) / 16000
        padded = np.pad(waveform, (0, max(0, 15600 - len(waveform))))
        _, embeddings, _ = backbone(padded)
        # The existing head returns logits; convert before thresholding/display.
        probabilities = source.softmax(teacher(embeddings, training=False).numpy())
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1.0, atol=1e-6):
            raise ValueError("Teacher probabilities do not satisfy the softmax contract")
        target = source.LABELS[row["label"]]
        selected = np.flatnonzero((probabilities[:, target] >= 0.5) &
                                  (probabilities.argmax(1) == target))
        intervals = merge_intervals([(i * .48 - .05, i * .48 + 1.01) for i in selected], duration)
        # Energy is a visual aid, never a class detector or automatic fallback label.
        blocks = np.array_split(waveform, min(600, len(waveform)))
        envelope = [round(float(np.sqrt(np.mean(b.astype(np.float64) ** 2))), 6) for b in blocks]
        clips.append({"sample_id": row["sample_id"], "label": row["label"],
                      "source_sha256": source.sha256(YAMNET / row["relative_path"]),
                      "audio": relative, "duration_s": duration, "intervals": intervals,
                      "reviewed": False, "envelope": envelope,
                      "teacher_target_probability_max": float(probabilities[:, target].max()),
                      "teacher_target_probability_mean": float(probabilities[:, target].mean())})
        print(f"REVIEW_CANDIDATE {number}/{len(rows)} {row['sample_id']} intervals={intervals}", flush=True)
    inventory = {"version": "dscnn-event-review-v1", "development_identity": development_identity(),
                 "teacher_model_sha256": source.sha256(teacher_path),
                 "teacher_role": "Existing frozen YAMNet head trained on old development clips; suggestions only, not independent labels or evaluation.",
                 "candidate_rule": "target_argmax_and_p_at_least_0.5; 0.48s patch hop, +/-0.05s margin; merge_gap_0.15s",
                 "clips": clips}
    inventory["inventory_sha256"] = source.fingerprint(inventory)
    write_json(directory / "inventory.json", inventory)
    candidate_rows = [{k: value for k, value in zip(ANNOTATION_FIELDS,
                      [c["sample_id"], a, b, c["label"], c["source_sha256"]])}
                      for c in clips for a, b in c["intervals"]]
    source.write_csv(directory / "candidate_intervals.NOT_REVIEWED.csv", candidate_rows)
    (directory / "index.html").write_text(review_html(inventory), encoding="utf-8")
    status.update(complete=True, clips=len(clips), inventory_sha256=inventory["inventory_sha256"])
    write_json(directory / "status.json", status)
    return directory


def accept(decisions_path, inventory_path, output):
    decisions, inventory = source.read_json(decisions_path), source.read_json(inventory_path)
    if inventory["inventory_sha256"] != source.fingerprint({k: v for k, v in inventory.items() if k != "inventory_sha256"}):
        raise ValueError("Review inventory content changed")
    if decisions.get("inventory_sha256") != inventory["inventory_sha256"]:
        raise ValueError("Review decisions belong to another inventory")
    if not decisions.get("reviewer", "").strip() or not decisions.get("reviewed_at"):
        raise ValueError("Reviewer and review date are required")
    if inventory["development_identity"] != development_identity():
        raise ValueError("Development inventory changed after review preparation")
    expected = {c["sample_id"]: c for c in inventory["clips"]}
    entries = decisions.get("clips", [])
    if len(entries) != len(expected) or {c["sample_id"] for c in entries} != set(expected):
        raise ValueError("Every development target clip must be reviewed exactly once")
    annotations = []
    for item in entries:
        original = expected[item["sample_id"]]
        if item.get("reviewed") is not True or not item.get("intervals"):
            raise ValueError(f"Human event review incomplete: {item['sample_id']}")
        if item.get("source_sha256") != original["source_sha256"]:
            raise ValueError("Reviewed original hash changed")
        for pair in item["intervals"]:
            if len(pair) != 2:
                raise ValueError("Intervals require start and end")
            a, b = map(float, pair)
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= original["duration_s"] + .001):
                raise ValueError(f"Invalid reviewed interval: {item['sample_id']}")
            # The UI displays milliseconds; clip only its sub-millisecond rounding.
            b = min(b, original["duration_s"])
            annotations.append(dict(zip(ANNOTATION_FIELDS, [item["sample_id"], a, b,
                                     original["label"], original["source_sha256"]])))
        starts = window_starts(round(original["duration_s"] * 16000))
        if not positive_window_indices(starts, item["intervals"]):
            raise ValueError(f"No eligible 1-second training window: {item['sample_id']}")
    output = Path(output).resolve()
    receipt_path = output.with_suffix(".review.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Reviewed annotation output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    source.write_csv(output, annotations)
    read_annotations(output, development_splits()[0])
    write_json(receipt_path, {"human_review_complete": True,
                             "reviewer": decisions["reviewer"], "reviewed_at": decisions["reviewed_at"],
                             "inventory_sha256": inventory["inventory_sha256"],
                             "decisions_sha256": source.sha256(decisions_path),
                             "annotations_sha256": source.sha256(output),
                             "development_identity": development_identity(), "reviewed_clips": len(expected)})
    print(f"REVIEWED_ANNOTATIONS_ACCEPTED {len(expected)} clips: {output}", flush=True)


def require_review_receipt(annotations):
    path = Path(annotations)
    receipt = source.read_json(path.with_suffix(".review.json"))
    if receipt.get("human_review_complete") is not True or not receipt.get("reviewer"):
        raise ValueError("A completed human review receipt is required")
    if receipt["annotations_sha256"] != source.sha256(path):
        raise ValueError("Reviewed annotations changed after acceptance")
    if receipt["development_identity"] != development_identity():
        raise ValueError("Reviewed development inventory changed")
    return receipt


def review_html(inventory):
    payload = json.dumps(inventory, ensure_ascii=False).replace("</", "<\\/")
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DS-CNN 이벤트 구간 검수</title><style>
body{margin:0;background:#f2f5f8;color:#182738;font:16px/1.55 system-ui,sans-serif}header{position:sticky;top:0;background:#102c46;color:white;padding:18px 5%;z-index:5}h1{font-size:24px;margin:0}main{max-width:1050px;margin:24px auto;padding:0 20px}article{background:white;border:1px solid #d8e1e8;border-radius:12px;margin:18px 0;padding:20px}h2{font-size:19px;margin:0}audio{width:100%;margin:10px 0}input[type=text]{font:inherit;padding:8px;border:1px solid #91a1ae;border-radius:6px}input.events{width:95%;font-family:monospace}button{font:inherit;padding:7px 12px;border-radius:6px;border:1px solid #8596a7;background:#fff;cursor:pointer;margin:4px}.primary{background:#08679d;color:white;border-color:#08679d}.status{font-weight:bold}svg{display:block;width:100%;height:110px;background:#eef4f8;cursor:crosshair}.hint{color:#526375;font-size:14px}.reviewed{border:2px solid #13816e}label{display:block;margin-top:10px}.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.error{color:#b00020}</style>
<header><h1>화재경보 · 초인종 이벤트 구간 검수</h1><div id="progress"></div></header>
<main><article><strong>아직 학습에 사용할 수 없는 자동 후보입니다.</strong><p>원음을 듣고 실제 타깃 소리가 들리는 시작·끝(초)을 확인해 주세요. 말소리·음악·무음은 제외하고 타깃의 반복음·멜로디·잔향은 포함합니다. 파란 구간은 기존 YAMNet의 제안이며 정답이 아닙니다. 검수 대상은 개발 데이터만입니다.</p><p>구간 입력 예: <code>1.20-3.80; 6.10-8.00</code>. 파형을 누르면 해당 시점으로 이동합니다. 소리가 시작할 때 ‘시작’, 끝날 때 ‘끝’ 버튼으로 구간을 추가할 수도 있습니다. 각 파일의 확인 상자를 직접 선택한 뒤 마지막에 JSON을 저장해 주세요. 소리가 전혀 없거나 라벨이 틀렸다면 그 ID를 채팅으로 알려주세요.</p><label>검수자 <input id="reviewer" type="text" placeholder="이름 또는 내부 ID" autocomplete="off"></label><p class="hint">모든 처리와 임시 저장은 이 브라우저 안에서만 이루어집니다. 원음은 외부로 전송하지 않습니다.</p></article><div id="clips"></div><article><button class="primary" id="save">검수 결과 JSON 저장</button><div id="message" role="status"></div></article></main>
<script>const inventory=__PAYLOAD__;
const key='dscnn-review-'+inventory.inventory_sha256;
let state={};try{state=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const reviewer=document.getElementById('reviewer');reviewer.value=state.reviewer||'';
function parse(s){return s.trim().split(';').filter(x=>x.trim()).map(x=>{const m=x.trim().match(/^(\\d+(?:\\.\\d+)?)\\s*-\\s*(\\d+(?:\\.\\d+)?)$/);if(!m)throw Error('구간 형식을 확인해 주세요: '+x);return [+m[1],+m[2]]})}
function persist(){state.reviewer=reviewer.value;localStorage.setItem(key,JSON.stringify(state));progress()}
function progress(){document.getElementById('progress').textContent=inventory.clips.filter(c=>state[c.sample_id]?.reviewed).length+' / '+inventory.clips.length+' 파일 확인'}
reviewer.oninput=persist;
for(const c of inventory.clips){
 const a=document.createElement('article');a.id=c.sample_id;
 const title=document.createElement('h2');title.textContent=c.sample_id+' · '+c.label+' · '+c.duration_s.toFixed(2)+'초';a.append(title);
 const hint=document.createElement('p');hint.className='hint';hint.textContent='기존 모델 타깃 확률: 평균 '+c.teacher_target_probability_mean.toFixed(3)+', 최대 '+c.teacher_target_probability_max.toFixed(3)+(c.intervals.length?'':' · 후보 없음: 직접 구간을 지정해 주세요');a.append(hint);
 const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=c.audio;a.append(audio);
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 1000 110');svg.setAttribute('preserveAspectRatio','none');svg.setAttribute('aria-label','RMS 파형과 타깃 구간');a.append(svg);
 const label=document.createElement('label');label.textContent='확정할 이벤트 구간 (초)';const input=document.createElement('input');input.type='text';input.className='events';input.value=state[c.sample_id]?.text??c.intervals.map(x=>x.map(v=>v.toFixed(3)).join('-')).join('; ');label.append(input);a.append(label);
 const controls=document.createElement('div');controls.className='controls';a.append(controls);let mark=null;
 function button(text,fn){const b=document.createElement('button');b.textContent=text;b.onclick=fn;controls.append(b)}
 button('현재 시각을 시작으로',()=>{mark=audio.currentTime;info.textContent='시작 '+mark.toFixed(3)+'초'});
 button('현재 시각을 끝으로 추가',()=>{if(mark===null||audio.currentTime<=mark){info.textContent='시작 시각을 먼저 지정해 주세요';return}const v=mark.toFixed(3)+'-'+audio.currentTime.toFixed(3);input.value+=(input.value.trim()?'; ':'')+v;mark=null;changed()});
 button('입력한 구간 재생',()=>{try{const pair=parse(input.value)[0];if(pair){audio.currentTime=pair[0];audio.play();audio.ontimeupdate=()=>{if(audio.currentTime>=pair[1]){audio.pause();audio.ontimeupdate=null}}}}catch(e){info.textContent=e.message}});
 const info=document.createElement('span');info.className='hint';controls.append(info);
 const confirm=document.createElement('label');const check=document.createElement('input');check.type='checkbox';check.checked=state[c.sample_id]?.reviewed||false;confirm.append(check,document.createTextNode(' 원음을 듣고 모든 타깃 이벤트 구간을 확인했습니다'));a.append(confirm);
 function draw(){svg.replaceChildren();const peak=Math.max(...c.envelope,0.001);const points=c.envelope.map((v,i)=>[(i+.5)/c.envelope.length*1000,105-v/peak*100]);const line=document.createElementNS(svg.namespaceURI,'polyline');line.setAttribute('points',points.map(p=>p.join(',')).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke','#344d63');line.setAttribute('stroke-width','1.5');svg.append(line);try{for(const [start,end] of parse(input.value)){const r=document.createElementNS(svg.namespaceURI,'rect');r.setAttribute('x',start/c.duration_s*1000);r.setAttribute('y',0);r.setAttribute('width',(end-start)/c.duration_s*1000);r.setAttribute('height',110);r.setAttribute('fill','#1681c9');r.setAttribute('opacity','.2');svg.append(r)}}catch(e){}a.classList.toggle('reviewed',check.checked)}
 svg.onclick=e=>{audio.ontimeupdate=null;audio.currentTime=Math.max(0,Math.min(c.duration_s,(e.clientX-svg.getBoundingClientRect().left)/svg.getBoundingClientRect().width*c.duration_s));info.textContent=audio.currentTime.toFixed(3)+'초'};
 function changed(){check.checked=false;state[c.sample_id]={text:input.value,reviewed:false};draw();persist()}
 input.oninput=changed;check.onchange=()=>{state[c.sample_id]={text:input.value,reviewed:check.checked};draw();persist()};draw();document.getElementById('clips').append(a);
}
document.getElementById('save').onclick=async()=>{const m=document.getElementById('message');try{if(!reviewer.value.trim())throw Error('검수자를 입력해 주세요');const clips=inventory.clips.map(c=>{const s=state[c.sample_id];if(!s?.reviewed)throw Error(c.sample_id+' 검수가 아직 안 됐습니다');const intervals=parse(s.text);if(!intervals.length||intervals.some(([a,b])=>!(0<=a&&a<b&&b<=c.duration_s+0.001)))throw Error(c.sample_id+' 구간 범위를 확인해 주세요');return {sample_id:c.sample_id,source_sha256:c.source_sha256,intervals,reviewed:true}});const result={version:inventory.version,inventory_sha256:inventory.inventory_sha256,reviewer:reviewer.value.trim(),reviewed_at:new Date().toISOString(),clips};if(location.protocol==='http:'){const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(result)});const data=await response.json();if(!response.ok)throw Error(data.error||'검수 제출 오류');m.textContent='검수 제출 완료. 채팅에 완료했다고 알려주세요.';return}const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=document.createElement('a');link.href=url;link.download='dscnn_event_review_20261114.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),10000);m.textContent='저장한 JSON의 위치를 채팅으로 알려주세요.'}catch(e){m.textContent=e.message}};if(location.protocol==='http:')document.getElementById('save').textContent='37개 검수 완료 · 제출';progress();
</script></html>'''.replace("__PAYLOAD__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    accept_parser = sub.add_parser("accept")
    accept_parser.add_argument("--decisions", type=Path, required=True)
    accept_parser.add_argument("--inventory", type=Path, required=True)
    accept_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output, args.device)
    else:
        accept(args.decisions, args.inventory, args.output)


if __name__ == "__main__":
    main()
