"""Offline inference of one existing audio file using a verified trained bundle."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dscnn_baseline.common import config,runtime,source,write_json
from dscnn_baseline.frontend import LogMel,windows
from dscnn_baseline.export import interpreter_for,predict_int8
from dscnn_baseline import decisions


def predict(bundle,audio,backend="int8"):
    bundle,audio=Path(bundle).resolve(),Path(audio).resolve()
    manifest=source.read_json(bundle/"release_manifest.json")
    for relative,digest in manifest["files"].items():
        path=(bundle/relative).resolve()
        if bundle not in path.parents or source.sha256(path)!=digest:
            raise ValueError("Bundle path or artifact hash changed")
    metadata=source.read_json(bundle/"model_metadata.json")
    policy=source.read_json(bundle/"decision_policy.json")
    if metadata["pipeline_only"] is not False or metadata["calibration_role"]!="train":
        raise ValueError("Synthetic/pipeline-only models cannot be used as trained classifiers")
    if metadata["labels"]!=config()[0]["labels"] or metadata["frontend_sha256"]!=source.fingerprint(config()[0]["frontend"]):
        raise ValueError("Trained model frontend or label contract differs from this code")
    cache=ROOT/".cache/offline_inference";cache.mkdir(parents=True,exist_ok=True)
    from sound_baseline.features import decode_audio
    with tempfile.TemporaryDirectory(prefix="single_audio_",dir=cache) as temporary:
        wav=Path(temporary)/"input.wav"
        decode_audio(audio,wav)
        with wave.open(str(wav)) as handle:
            pcm=np.frombuffer(handle.readframes(handle.getnframes()),dtype="<i2")
        frontend=LogMel()
        features=np.stack([frontend(chunk) for _,chunk in windows(pcm)])
    # Deliberate host inference, unrelated to the fail-closed GPU training path.
    tf,keras=runtime("cpu")
    if backend=="int8":
        probabilities=predict_int8(interpreter_for(tf,bundle/"dscnn_int8.tflite"),features)
    elif backend=="float":
        model=keras.models.load_model(str(bundle/"classifier.keras"),compile=False)
        probabilities=np.concatenate([model(features[i:i+128],training=False).numpy() for i in range(0,len(features),128)])
    else:raise ValueError("Backend must be int8 or float")
    mean=probabilities.mean(0)
    row={f"p_{label}":float(mean[index]) for label,index in source.LABELS.items()}
    if policy["kind"] == "per_target_argmax_confidence_rejection":
        from improvement.policy import apply
        decision=apply([row],policy["thresholds"])[0]
    else:
        decision=decisions.apply([row],policy["threshold"])[0]
    return {"kind":"offline_single_file_inference","backend":backend,"audio":str(audio),
            "audio_sha256":source.sha256(audio),"duration_s":len(pcm)/16000,"window_count":len(features),
            "probabilities":{label:float(mean[index]) for label,index in source.LABELS.items()},
            "predicted_label":decision["predicted_label"],"decision_policy":policy,
            "deployment_ready":False,"note":"File-level host inference. This does not measure live microphone, board or event timing performance."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",type=Path,required=True)
    parser.add_argument("--audio",type=Path,required=True)
    parser.add_argument("--backend",choices=("int8","float"),default="int8")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=predict(args.bundle,args.audio,args.backend)
    if args.output:
        if args.output.exists():raise FileExistsError("Inference report already exists")
        write_json(args.output,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
