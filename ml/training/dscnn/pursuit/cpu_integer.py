"""Existing optimized CPU kernels, gated against actual reference outputs."""
import numpy as np
from comparison.dataset import write_json,sha
from .delivered_integer import predict_q


def open_interpreter(tf,path,fast=False,threads=1):
    enum=tf.lite.experimental.OpResolverType
    resolver=getattr(enum,'BUILTIN_WITHOUT_DEFAULT_DELEGATES',enum.BUILTIN_REF) if fast else enum.BUILTIN_REF
    interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=threads,
        experimental_op_resolver_type=resolver);interpreter.allocate_tensors()
    assert 'DELEGATE' not in {r['op_name'] for r in interpreter._get_ops_details()}
    return interpreter


def validated_interpreter(tf,path,representative,report,max_samples=128):
    indices=np.unique(np.linspace(0,len(representative)-1,min(max_samples,len(representative))).astype(int))
    x=np.asarray(representative[indices],np.float32)
    reference=open_interpreter(tf,path);optimized=open_interpreter(tf,path,True,2)
    a=predict_q(reference,x);b=predict_q(optimized,x);error=int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max())
    passed=np.array_equal(a,b)
    write_json(report,dict(passed=bool(passed),source_model_sha256=sha(path),sampled_training_windows=len(x),
        maximum_output_error_lsb=error,argmax_disagreements=int((a.argmax(1)!=b.argmax(1)).sum()),
        selected_backend='existing_optimized_builtin_cpu' if passed else 'reference_cpu',
        all_possible_inputs_or_actual_board_parity_claimed=False,hardware_or_environment_changed=False))
    return optimized if passed else reference
