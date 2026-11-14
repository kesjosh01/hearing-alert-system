"""Full-int8 CNN exports and readout fitting in exact delivered integer units."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from comparison.dataset import write_json,sha
from dscnn_baseline.export import TFLM_OPS
from dscnn_baseline.frontend import quantize

LABELS=['fire_alarm','doorbell','other']


def export(tf,keras,model,representative,provenance,out,frontend,kind='probability',temperature=1.):
    out=Path(out);out.mkdir(exist_ok=False);shape=tuple(model.input_shape[1:])
    assert len(shape)==3 and representative.shape[1:]==shape and np.isfinite(representative).all()
    assert len(provenance)==len(representative) and all(r['split']=='train' for r in provenance)
    assert kind in ('probability','logits') and temperature>0
    clone=keras.models.clone_model(model);clone.set_weights(model.get_weights())
    layer=clone.get_layer('class_logits');weights=layer.get_weights();layer.set_weights([a/temperature for a in weights])
    net=keras.Model(clone.input,layer.output) if kind=='logits' else clone

    class Serving(tf.Module):
        def __init__(self,n):super().__init__();self.net=n
        @tf.function(input_signature=[tf.TensorSpec([1,*shape],tf.float32,name='log_mel')])
        def serve(self,x):return {'delivered_scores':self.net(x,training=False)}
    serving=Serving(net);saved=out/'saved_model';tf.saved_model.save(serving,str(saved),signatures={'serving_default':serving.serve})
    converter=tf.lite.TFLiteConverter.from_saved_model(str(saved));converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=lambda:([a[None]] for a in representative)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    path=out/'model_int8.tflite';path.write_bytes(converter.convert())
    interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);interpreter.allocate_tensors()
    a,b=interpreter.get_input_details()[0],interpreter.get_output_details()[0]
    assert list(a['shape'])==[1,*shape] and list(b['shape'])==[1,3] and a['dtype']==b['dtype']==np.int8
    operations=sorted({r['op_name'] for r in interpreter._get_ops_details()});assert not set(operations)-TFLM_OPS
    assert all(t['dtype'] in (np.int8,np.int32) for t in interpreter.get_tensor_details())
    assert len(b['quantization_parameters']['scales'])==1
    def detail(t):
        scale,zero=t['quantization'];assert scale>0
        return dict(shape=t['shape'].tolist(),dtype='int8',scale=float(scale),zero_point=int(zero))
    meta=dict(input=detail(a),output=detail(b),operator_set=operations,labels=LABELS,
        model_sha256=sha(path),model_bytes=path.stat().st_size,frontend=frontend,
        output_semantics=kind,export_logit_temperature=temperature,
        export_transformation='Divide only final Dense weights/bias by positive temperature; source checkpoint unchanged',
        representative_role='train',representative_windows=len(representative),pipeline_only=False,
        deployment_ready=False,board_validated=False,actual_product_targets_verified=False)
    write_json(out/'model_metadata.json',meta)
    raw=path.read_bytes();h=['// Trained research candidate; deployment_ready=false.',
        '// Output semantics: '+kind+'; temperature='+str(temperature),
        '#pragma once','#include <cstddef>','#include <cstdint>','namespace hearing {',
        'static constexpr bool kModelPipelineOnly = false;','static constexpr bool kDeploymentReady = false;',
        'alignas(16) static const uint8_t kModelData[] = {']
    h+=['  '+', '.join(f'0x{v:02x}' for v in raw[i:i+16])+',' for i in range(0,len(raw),16)]
    h+=['};','static constexpr size_t kModelBytes = sizeof(kModelData);','}','']
    (out/'model_int8.h').write_text('\n'.join(h))
    return interpreter,meta


def predict_q(interpreter,x):
    a,b=interpreter.get_input_details()[0],interpreter.get_output_details()[0];scale,zero=a['quantization'];out=[]
    for image in x:
        interpreter.set_tensor(a['index'],quantize(image[None],scale,zero));interpreter.invoke()
        out.append(interpreter.get_tensor(b['index'])[0].copy())
    return np.stack(out)


def scores(q,kind,zero,policy):
    q=np.asarray(q,dtype=np.int16);arg=q.argmax(1);result=np.zeros((len(q),2),np.int16)
    for k in (0,1):
        raw=q[:,k]-zero if kind=='probability' else q[:,k]-np.delete(q,k,axis=1).max(1)
        raw=np.where(arg==k,np.maximum(raw,0),0);need,width=policy[k];assert 1<=need<=width
        history=[]
        for n,value in enumerate(raw):
            history.append(int(value));history=history[-width:]
            if len(history)>=need:result[n,k]=sorted(history,reverse=True)[need-1]
    return result


def envelopes(rows,values,kind,zero,policy):
    return np.stack([scores(values[r['sample_id']],kind,zero,policy).max(0) for r in rows])


def fit_cuts(rows,values,kind,zero,policy,lower_bound=None):
    maxima=envelopes(rows,values,kind,zero,policy);y=np.array([LABELS.index(r['label']) for r in rows])
    cuts=np.array([int(maxima[y!=k,k].max())+1 for k in (0,1)],np.int16)
    assert np.all((1<=cuts)&(cuts<=256))
    if lower_bound is not None:cuts=np.maximum(cuts,lower_bound)
    return cuts


def measure(rows,values,kind,zero,policy,cuts):
    maxima=envelopes(rows,values,kind,zero,policy);flags=(maxima>=cuts)&(maxima>0)
    y=np.array([LABELS.index(r['label']) for r in rows]);classes={};rates=[]
    for k,label in enumerate(LABELS[:2]):
        groups=defaultdict(list)
        for i,r in enumerate(rows):
            if y[i]==k:groups[r['split_group_id']].append(bool(flags[i,k]))
        recall=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None;rates.append(recall)
        classes[label]=dict(files=int((y==k).sum()),detected_files=int(flags[y==k,k].sum()),source_groups=len(groups),
            source_mean_file_recall=recall,wrong_class_or_negative_alert_files=int(flags[y!=k,k].sum()))
    conditions={}
    for condition in sorted({r.get('probe_condition') or 'original' for r in rows if r['label']!='other'}):
        conditions[condition]={label:dict(files=sum(r['label']==label and (r.get('probe_condition') or 'original')==condition for r in rows),
            detected_files=sum(bool(flags[i,k]) for i,r in enumerate(rows) if r['label']==label and (r.get('probe_condition') or 'original')==condition)) for k,label in enumerate(LABELS[:2])}
    return dict(fire_recall=rates[0],doorbell_recall=rates[1],classes=classes,conditions=conditions,
        other_files=int((y==2).sum()),other_false_alert_files=int(flags[y==2].any(1).sum()),
        wrong_target_alert_files=int(sum(flags[i,1-y[i]] for i in range(len(y)) if y[i] in (0,1))),
        cuts_integer_units=[int(v) for v in cuts],policy=policy,measure='Source-mean file detection; acoustic variants retain original source groups',
        actual_event_or_monthly_rate_verified=False)


def key(m):
    rates=[]
    for label,target in [('fire_alarm',.999),('doorbell',.95)]:
        values=[c[label]['detected_files']/c[label]['files'] for c in m['conditions'].values() if c[label]['files']]
        rates.append(min(values) if values else 0.)
    return (min(rates[0]/.999,rates[1]/.95),sum(rates)/2,
        ((m['fire_recall'] or 0)+(m['doorbell_recall'] or 0))/2)
