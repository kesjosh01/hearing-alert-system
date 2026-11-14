"""Complete model-header frontend identity and verify research bundle bytes."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from comparison.dataset import write_json,sha
from dscnn_baseline.common import source,config
from .paths import REPORT


def main(name):
    p=REPORT/'candidates'/name;candidate=p/'candidate.json';j=json.loads(candidate.read_text())
    assert all(sha(p/k)==v for k,v in j['files_sha256'].items())
    meta_path=p/'model_metadata.json';meta=json.loads(meta_path.read_text());header=p/'model_int8.h';text=header.read_text()
    model_sha=sha(p/'model_int8.tflite');fp=source.fingerprint(meta['frontend'])
    amended=False
    if 'kModelFrontendSha256' not in text:
        backup=REPORT/'frontend_metadata_completion'/name;backup.mkdir(parents=True,exist_ok=False)
        for f in [candidate,meta_path,header]:shutil.copyfile(f,backup/f.name)
        header.write_text(text+'\nnamespace hearing {\nstatic constexpr char kModelFrontendSha256[] = "'+fp+'";\n}\n')
        meta['frontend_sha256']=fp;write_json(meta_path,meta)
        j['files_sha256']={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name!='candidate.json'}
        j['frontend_header_identity_completion_backup']=str(backup);write_json(candidate,j);amended=True
    else:
        assert fp==meta['frontend_sha256'] and fp in text
    raw=bytes(int(x,16) for x in re.findall(r'0x([0-9a-f]{2})',header.read_text()))
    assert hashlib.sha256(raw).hexdigest()==model_sha==meta['model_sha256']
    cpp='#include "'+str(header)+'"\nstatic_assert(hearing::kModelBytes > 0, "model bytes");\nstatic_assert(sizeof(hearing::kModelFrontendSha256) == '+str(len(fp)+1)+', "frontend identity");\n'
    result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-x','c++','-'],input=cpp,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert all(sha(p/k)==v for k,v in j['files_sha256'].items())
    write_json(REPORT/'bundle_contract_checks'/(name+'.json'),dict(passed=True,model_sha256=model_sha,
        header_bytes_match_flatbuffer=True,source_checkpoint_or_prediction_data_changed=False,
        header_frontend_identity_added=amended,frontend_sha256=fp,
        existing_portable_98x40_frontend_matches=fp==source.fingerprint(config()[0]['frontend']) and meta['input']['shape']==[1,98,40,1],
        required_header_interface_host_syntax_passed=True,
        board_firmware_build_or_flash_performed=False,actual_product_targets_verified=False))
    print('RESEARCH_MODEL_HEADER_AND_FRONTEND_CONTRACT_PASS',name,amended,model_sha,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();main(a.name)
