"""Run a single authorized stage using existing libraries; never install/restart."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from .dataset import write_json
from .protocol import ROOT,REPORT


def main():
    p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    if not a.module.startswith('comparison.'):raise ValueError('Only this study may be launched')
    gpu='GPU-10f7f4b0-eee7-9d43-26de-7bbefdbfc7ce'
    visible=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
    if gpu not in visible or active:raise RuntimeError('Selected GPU unavailable/busy; no sharing or termination')
    site=ROOT/'.venv/lib/python3.12/site-packages';libs=sorted(str(p) for p in (site/'nvidia').glob('*/lib') if p.is_dir())
    if len(libs)!=12:raise RuntimeError('Existing CUDA library inventory changed; do not install')
    env=dict(os.environ);env.update(CUDA_VISIBLE_DEVICES=gpu,TF_USE_LEGACY_KERAS='1',TF_FORCE_GPU_ALLOW_GROWTH='true',
        TF_XLA_FLAGS='--tf_xla_auto_jit=0',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',
        WANDB_MODE='disabled',CUDA_CACHE_PATH=str(ROOT/'.cache/runtime/cuda'),
        PATH=os.pathsep.join([str(site/'nvidia/cuda_nvcc/bin'),str(ROOT/'.venv/bin'),env.get('PATH','')]),
        LD_LIBRARY_PATH=os.pathsep.join(libs+([env['LD_LIBRARY_PATH']] if env.get('LD_LIBRARY_PATH') else [])))
    env.pop('CUDA_FORCE_PTX_JIT',None);env.pop('XLA_FLAGS',None)
    cmd=[str(ROOT/'.venv/bin/python'),'-B','-m',a.module,*a.arguments]
    now=time.time_ns();path=REPORT/'processes'/f'{now}.json';started=time.monotonic()
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True)
    receipt=dict(pid=child.pid,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],command=cmd,cwd=str(ROOT),gpu_uuid=gpu,
                 selected_cuda_libraries=libs,process_environment_only=True,package_changes=False,exit_code=None)
    write_json(path,receipt);code=child.wait();receipt.update(exit_code=code,elapsed_seconds=time.monotonic()-started);write_json(path,receipt)
    raise SystemExit(code)


if __name__=='__main__':main()
