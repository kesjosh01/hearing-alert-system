"""Reviewable known-recording candidate; never promote the production bundle."""
import json
import math
import shutil
from comparison.dataset import read_csv,write_json,sha
from .paths import ROOT,DATA,REPORT


def main():
    source=REPORT/'domestic_seen_integer_full_positive';status=json.loads((source/'status.json').read_text());assert status['complete']
    result=next(r for r in status['results'] if r['kind']=='probability')
    classes=result['selection_after_calibration']['classes']
    assert all(r['detected_files']==r['training_seen_files'] for r in classes.values())
    out=REPORT/'candidates/domestic_known_v1';out.mkdir(parents=True,exist_ok=False)
    directory=source/'probability';files={
        'model_int8.tflite':directory/'dscnn_int8.tflite',
        'model_int8.h':directory/'dscnn_int8.h',
        'model_metadata.json':directory/'model_metadata.json',
        'readout_config.json':directory/'readout_config.json',
        'analysis.json':directory/'analysis.json',
        'representative_training_windows.json':directory/'calibration_provenance.json',
        'dataset_boundary.json':DATA/'domestic_seen_v1/freeze.json',
        'known_training_positive_files.csv':DATA/'domestic_seen_v1/known_positive_probes.csv',
        'classifier_training_status.json':ROOT/'runs'/status['source_model']/'status.json'}
    for name,p in files.items():shutil.copyfile(p,out/name);assert sha(p)==sha(out/name)
    meta=json.loads((directory/'model_metadata.json').read_text());config=json.loads((directory/'readout_config.json').read_text())
    assert meta['output_semantics']=='probability' and config['policy']==[[1,1],[1,1]]
    scale=meta['output']['scale'];zero=meta['output']['zero_point']
    cuts=[math.ceil(t/scale+zero) for t in config['thresholds']]
    assert all(-128<=c<=127 for c in cuts)
    write_json(out/'delivered_output_contract.json',dict(output_semantics='Quantized softmax probabilities',
        class_order=['fire_alarm','doorbell','other'],output_scale=scale,output_zero_point=zero,
        target_argmax_required=True,per_class_int8_minimum_q=cuts,policy=[[1,1],[1,1]],
        tie_break='First maximum, matching numpy.argmax',
        threshold_semantics='Class must win argmax and its int8 value must be >= its class q threshold',
        multiple_notifications_per_recording_controlled=False,actual_notification_controller_tested=False,
        old_firmware_thresholds_automatically_updated=False,board_validated=False))
    files_hash={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write_json(out/'candidate.json',dict(kind='DOMESTIC_TRAINING_SEEN_RESEARCH_CANDIDATE',
        source_model=status['source_model'],source_model_sha256=status['source_model_sha256'],
        files_sha256=files_hash,training_seen_fire_files=17,training_seen_doorbell_files=26,
        training_seen_int8_fire_detected=17,training_seen_int8_doorbell_detected=26,
        selection_other_files=422,calibration_other_files=312,
        selection_and_calibration_errors_zero_by_threshold_fit=True,
        independent_target_test_files=0,actual_device_months=0,
        actual_product_targets_verified=False,deployment_ready=False,
        existing_model_bundle_or_firmware_replaced=False,
        research_license_scope='Includes legacy noncommercial datasets and original source conditions; commercial clearance not claimed'))
    text=f'''# 국내 학습 음원에 맞춘 int8 후보 모델

화재경보 **17/17**, 초인종 **26/26**을 감지한 정수 모델입니다. 이 43개 음원은 모두 학습과 모델 선택에 사용했습니다. 새 가정·새 경보기·실제 발생 사건의 성능을 검증한 결과가 아닙니다.

선택용 다른 소리 422개와 임계값 보정용 312개에 대해 오류가 없도록 임계값을 정했습니다. 따라서 이 0이라는 값으로 기기당 30일 오알림률을 계산할 수 없습니다. 실제 기기 관측 기간은 0일입니다.

- 입력: PCM16 mono 16 kHz → 기존 portable C++ 98×40 log-Mel, 1초 입력과 0.5초 이동. 입력 양자화 수치는 `model_metadata.json` 참조.
- 출력 순서: `fire_alarm`, `doorbell`, `other`. 출력은 softmax 확률이며 raw logits가 아닙니다.
- 출력 int8 값의 클래스별 최소값: 화재경보 **{cuts[0]}**, 초인종 **{cuts[1]}**. 해당 클래스가 argmax일 때만 적용합니다. 원본 확률 임계값은 `readout_config.json`에 있습니다.
- 변환용 대표 입력: 학습 음원에서 뽑은 {status['representative_windows']:,}개 구간. 평가 음원을 정수 변환에 사용하지 않았습니다.
- 모델 크기: {meta['model_bytes']:,} bytes. 실제 보드 tensor arena, 실행 시간, 소비 전력은 미측정입니다.
- 이 폴더는 검토용 후보입니다. 기존 배포 모델·펌웨어·임계값을 교체하지 않았습니다. 알림 재발행 제어와 마이크/진동/LED 동작도 실제 기기에서 확인하지 않았습니다.

`candidate.json`에 파일 해시와 검증 범위를 기록했습니다. `known_training_positive_files.csv`는 학습에 사용한 양성 음원 목록이며 독립 시험 목록이 아닙니다. 실제 제품 목표 달성 상태는 `false`입니다.
'''
    (out/'README.md').write_text(text,encoding='utf-8')
    print('DOMESTIC_KNOWN_RECORDING_REVIEW_BUNDLE_CREATED',out,'q_thresholds',cuts,flush=True)


if __name__=='__main__':main()
