"""Offline clip classification or causal replay using a verified local bundle."""
import argparse
import json
import wave
from pathlib import Path
import numpy as np
from dscnn_baseline.common import runtime,source
from dscnn_baseline.frontend import LogMel,windows
from dscnn_baseline.export import interpreter_for,predict_int8
from sound_baseline.features import decode_audio
from .protocol import ROOT
from .dataset import sha,write_json
from .metrics import LABELS,pool,decide
from .temporal import CausalDecisionState


def run(bundle,audio,backend='int8',mode='clip'):
    bundle=Path(bundle).resolve();audio=Path(audio).resolve()
    manifest=json.loads((bundle/'release_manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if sha(bundle/name)!=digest:raise ValueError('Model bundle file changed: '+name)
    if not audio.is_file():raise FileNotFoundError(audio)
    metadata=json.loads((bundle/'model_metadata.json').read_text());assert not metadata['pipeline_only']
    digest=sha(audio);cache=ROOT/'.cache/comparison_inference'/digest
    if not cache.exists():
        cache.mkdir(parents=True);decode_audio(audio,cache/'audio.wav');write_json(cache/'source.json',dict(source_sha256=digest))
    else:
        if not (cache/'source.json').exists():raise ValueError('Incomplete inference cache; preserve and inspect before retry')
        assert json.loads((cache/'source.json').read_text())['source_sha256']==digest
    with wave.open(str(cache/'audio.wav')) as h:
        assert h.getframerate()==16000 and h.getnchannels()==1
        pcm=np.frombuffer(h.readframes(h.getnframes()),dtype='<i2')
    frontend=LogMel();pairs=list(windows(pcm));features=np.stack([frontend(chunk) for _,chunk in pairs]);tf,keras=runtime('cpu')
    if backend=='int8':prob=predict_int8(interpreter_for(tf,bundle/'dscnn_int8.tflite'),features)
    else:
        net=keras.models.load_model(str(bundle/'classifier.keras'),compile=False)
        prob=np.concatenate([net(features[i:i+128],training=False).numpy() for i in range(0,len(features),128)])
    policy=json.loads((bundle/'decision_policy.json').read_text());thresholds=tuple(policy[backend+'_thresholds'][k] for k in LABELS[:2])
    clip=pool(prob,policy['pooling_fraction']);pred=int(decide(clip[None],thresholds)[0])
    result=dict(audio=str(audio),backend=backend,mode=mode,window_count=len(prob),duration_s=len(pcm)/16000,
                label=LABELS[pred],probabilities=dict(zip(LABELS,map(float,clip))),thresholds=dict(zip(LABELS[:2],thresholds)),model_sha256=sha(bundle/('dscnn_int8.tflite' if backend=='int8' else 'classifier.keras')))
    if mode=='replay':
        temporal=json.loads((bundle/'temporal_policy.json').read_text());tt=temporal[backend+'_thresholds'];state=CausalDecisionState(tt,temporal['policy']);alerts=[]
        for (start,_),p in zip(pairs,prob):alerts+=state.step(p,(start+16000)/16000)
        result.update(alerts=alerts,causal_policy=temporal['policy'],timing='Window-end audio time. File-start replay with final end-aligned window; short files zero-pad. No acoustic onset ground truth.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--audio',required=True,type=Path)
    p.add_argument('--backend',choices=['float','int8'],default='int8');p.add_argument('--mode',choices=['clip','replay'],default='clip');a=p.parse_args()
    print(json.dumps(run(a.bundle,a.audio,a.backend,a.mode),ensure_ascii=False,indent=2))
