"""Input-shape-aware correction around the preserved one-window estimator."""
import numpy as np
from dscnn_baseline.model import complexity as legacy_complexity


def complexity(model,keras):
    value=legacy_complexity(model,keras);value['input_int8_bytes']=int(np.prod(model.input_shape[1:]))
    value['input_shape']=list(model.input_shape[1:]);count=model.input_shape[-1]
    value['past_context_int8_storage_bytes']=int((count-1)*98*40)
    value['past_context_float32_storage_bytes_if_unquantized']=int((count-1)*98*40*4)
    value['notice']='Estimates only. History storage excludes input/output tensors and frontend/audio/stack buffers; actual board arena, RAM, latency and energy unmeasured.'
    return value
