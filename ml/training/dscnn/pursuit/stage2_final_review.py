"""Reconcile actual training, locked integer evaluations and missing field evidence."""
from collections import Counter
import json
import time
from pathlib import Path
from comparison.dataset import read_csv, write_json, sha
from .paths import ROOT, REPORT
from .stage2_native_export import OUT as EXPORT_OUT
from .stage2_paired_integer import OUT as PAIRED_OUT, BUNDLES
from .stage2_public_benchmark import OUT as TEST_OUT, checked_freeze
from .stage2_known_probe import OUT as PROBE_OUT


def load(path):
    return json.loads(path.read_text())


def counts(metric):
    a, b = metric['classes']['fire_alarm'], metric['classes']['doorbell']
    return f'{a["detected_files"]}/{a["files"]}', f'{b["detected_files"]}/{b["files"]}'


def main():
    public = load(TEST_OUT / 'results.json')
    paired = load(PAIRED_OUT / 'status.json')
    export = load(EXPORT_OUT / 'status.json')
    probes = load(PROBE_OUT / 'results.json')
    fusion = load(REPORT / 'stage2_domestic_fusion_v1/status.json')
    fusion_public = load(REPORT / 'stage2_domestic_fusion_v1/new_public_results.json')
    membership = load(REPORT / 'stage2_known_source_membership_clarification.json')
    other_audit = load(REPORT / 'stage2_other_target_metadata_audit.json')
    assert all(d['complete'] for d in [public, paired, export, probes])
    assert fusion['complete'] and fusion_public['complete'] and membership['complete']
    assert other_audit['complete'] and not other_audit['target_or_ambiguous_metadata_hits']
    checked_freeze()
    stop = load(REPORT / 'collection_stopped_by_user.json')
    assert stop['collection_disabled_by_user'] and stop['all_selected_collectors_stopped']
    hardware = load(REPORT / 'hardware_availability.json')
    assert not hardware['microphone_and_alert_devices_ready'] and not hardware['actual_hardware_execution_authorized']
    baseline = load(REPORT / 'protected_previous_comparison.json')
    changed = [p for p, expected in baseline.items() if sha(Path(p)) != expected]
    assert not changed, changed
    write_json(REPORT / 'stage2_protected_previous_final_audit.json', dict(passed=True,
        checked_at_unix=time.time(), protected_files=len(baseline), changed_files=[],
        source_snapshot_sha256=sha(REPORT / 'protected_previous_comparison.json')))
    completed = {}
    for p in sorted((ROOT / 'runs').glob('pursuit_20261114*/status.json')):
        state = load(p)
        steps = state.get('actual_optimizer_steps', state.get('optimizer_steps', 0))
        if state.get('complete') and steps > 0:
            model = p.parent / 'classifier.keras'
            if model.exists() and state.get('model_sha256'):
                assert sha(model) == state['model_sha256']
            completed[p.parent.name] = dict(optimizer_steps=steps, selected_epoch=state.get('selected_epoch'),
                epochs_completed=state.get('epochs_completed', state.get('epoch')), status_sha256=sha(p))
    kd = ROOT / 'runs/pursuit_20261114_r06_native40_distilled_vs_control'
    kd_state = load(kd / 'status.json')
    history = read_csv(kd / 'history.csv')
    assert kd_state['complete'] and kd_state['distillation'] and any(float(v['kd_loss']) > 0 for v in history)
    exposure = load(kd / 'selected_checkpoint_exposure_audit.json')
    assert exposure['passed'] and exposure['selected_checkpoint']['negative_views_with_zero_draws'] == 0
    bn = load(REPORT / 'stage2_train_only_bn_matched_comparison.json')
    assert bn['complete'] and bn['identical_training_source_windows'] and not bn['learned_weights_changed']
    large_teacher_comparison = load(REPORT / 'stage2_teacher_matched_comparison.json')
    assert large_teacher_comparison['complete']
    cleanup = load(REPORT / 'cache_release_review/approved_exact_release_receipt.json')
    assert cleanup['completed'] and cleanup['deleted_files'] == 11581
    references = {name: sha(REPORT / name) for name in [
        'stage2_native40_float_matched_comparison.json', 'stage2_train_only_bn_matched_comparison.json',
        'stage2_teacher_matched_comparison.json', 'stage2_teacher_vs_native_control.json',
        'stage2_native40_export_plan.json', 'stage2_paired_integer_plan.json',
        'stage2_public_benchmark_plan.json', 'collection_stopped_by_user.json',
        'hardware_availability.json', 'actual_product_evidence_gate.json',
        'stage2_domestic_fusion_plan.json', 'stage2_domestic_fusion_v1/status.json',
        'stage2_domestic_fusion_v1/new_public_results.json',
        'stage2_known_source_membership_clarification.json', 'stage2_other_target_metadata_audit.json',
        'stage2_learning_curves_v1/figure_manifest.json', 'stage2_learning_curves_v1/rendered_figure_check.json']}
    result = dict(created_at_unix=time.time(), completed_scope='Local stage2 fits, train-only BN comparison, actual int8 exports, frozen new public test and known-source diagnostics',
        offline_stage2_complete=True, completed_real_optimizer_runs=completed,
        no_optimizer_update_transforms_excluded_from_fit_count=True,
        source_refined_teacher_did_not_beat_old_large_reference=not large_teacher_comparison['distillation_gate_passed'],
        actual_native_student_kl_distillation_executed=True,
        actual_kd_optimizer_steps=kd_state['actual_optimizer_steps'], kd_selected_epoch=kd_state['selected_epoch'],
        selected_kd_checkpoint_negative_view_coverage=exposure['selected_checkpoint'],
        pre_test_selected_global_bundle='native40_r06_v1', global_winner_alias=public['global_winner_alias'],
        new_public_benchmark_sha256=sha(TEST_OUT / 'results.json'),
        known_source_diagnostic_sha256=sha(PROBE_OUT / 'results.json'),
        additional_fixed_fusion_experiment_complete=True,
        fusion_readout_not_selected_from_public_results=True,
        fusion_public_metric=fusion_public['metric'],
        current_domestic_probe_global_membership=membership['counts'],
        protected_previous_files_verified=len(baseline), collection_disabled_by_user=True,
        original_audio_or_model_deletion_authorized=False,
        actual_device_days=0, actual_activation_recall_verified=False,
        actual_notification_monthly_rate_verified=False, actual_product_targets_verified=False,
        overall_goal_complete=False, hardware_available=False,
        unresolved_requirements=['Target microphone and alert hardware',
            'Independent representative actual fire/button activation ground truth and detection/notification logs',
            'Real device exposure and false-notification counts with a fixed final system'],
        evidence_sha256=references)
    write_json(REPORT / 'STAGE2_REVIEW.json', result)
    labels = {'pursuit_20261114_r06_native40_supervised': '비증류',
        'pursuit_20261114_r06_native40_distilled_vs_control': 'YAMNet 증류',
        'pursuit_20261114_r06_native40_supervised_bnrefit': '비증류 + BN 통계 보정',
        'pursuit_20261114_r06_native40_distilled_vs_control_bnrefit': '증류 + BN 통계 보정'}
    lines = ['# 로컬 확장 연구의 학습·int8·고정 평가 결과', '',
        '모델 개선 실험과 소프트웨어 평가를 완료했습니다. 실제 제품 목표의 달성은 확인하지 못했습니다. '
        '마이크·알림 장치가 준비되지 않았고 실제 기기 검증 기간은 0일입니다. '
        '추가 음원 및 원격 metadata 수집은 사용자 지시에 따라 중단한 상태입니다.', '',
        f'기존 연구를 포함해 실제 optimizer 업데이트가 확인된 완료 학습은 {len(completed)}회입니다. '
        'BatchNorm 통계 보정 두 건과 사전학습 의미 분류기 추론은 새 optimizer 학습 횟수에 넣지 않았습니다.', '',
        '## 실제 YAMNet distillation과 학습 길이', '',
        'Full YAMNet encoder를 사용한 교사로 KL distillation을 실제 수행했습니다. '
        f'학생은 {kd_state["epochs_completed"]} epoch, {kd_state["actual_optimizer_steps"]:,}회 optimizer 업데이트를 수행했고 '
        f'선택된 체크포인트는 epoch {kd_state["selected_epoch"]}입니다. KL 항이 실제로 0보다 컸다는 학습 이력과 교사 모델 해시를 보존했습니다. '
        '선택된 학생 체크포인트는 학습용 기타 구간 9,457개를 모두 사용했으며, 이 구간을 별도 독립 녹음으로 세지 않았습니다.', '',
        '같은 1,191개 선택 자료에서, 같은 1/1 logit-margin readout으로 비교한 출처별 평균 파일 감지율입니다. '
        '임계값을 정한 자료이므로 독립 시험 성능은 아닙니다.', '',
        '| 모델 | 화재경보 | 초인종 |', '|---|---:|---:|']
    for source, label in labels.items():
        state = load(ROOT / 'runs' / source / 'status.json'); metric = state['selected_metrics']
        lines.append(f'| {label} | {metric["fire_recall"]:.1%} | {metric["doorbell_recall"]:.1%} |')
    lines += ['', '선택 자료에서는 이번 증류가 비증류 기준선을 개선하지 못했습니다. 두 BN 보정은 같은 12,800개 학습 창만 사용했고 '
        '학습된 kernel·gamma·beta 배열 41개는 그대로 유지하면서 moving-statistic 배열 26개만 바꿨습니다. '
        '두 클래스의 균형을 함께 개선하지 못한 결과도 그대로 보존했습니다.', '',
        '비증류는 최대 100 epoch 중 71 epoch, 증류는 69 epoch에서 조기 종료됐습니다. '
        '각각 선택 epoch는 51과 49이며, 이후 20 epoch 동안 선택 기준이 좋아지지 않았습니다. '
        '이번 결과는 같은 설정에서 epoch 상한만 더 높이면 개선된다는 근거를 제공하지 않습니다.', '',
        f'![학습 손실과 선택 자료의 파일 감지율]({REPORT / "stage2_learning_curves_v1/learning_curves.png"})', '',
        '그림은 모든 epoch를 생략·평활화 없이 표시했습니다. 손실 감소와 선택 감지율의 불안정을 구분해서 볼 수 있습니다. '
        'CSV 원본·코드·그림 해시, 대체 설명과 출력 크기 검증을 같은 그림 폴더에 저장했습니다. 기존 시스템의 CPU plotting 환경을 사용했으며 설치하지 않았습니다.', '',
        '새 교사는 동일한 확장 자료에서 소형 비증류 학생보다 강해 압축 전이 실험을 허용했습니다. '
        '기존 큰 64-bin 모델과 동일한 과거 선택 자료로 비교했을 때는 새 교사가 더 강하지 않았습니다. '
        '두 비교의 대상과 gate를 구분해 기록했습니다.', '',
        '## 실제 int8 후보의 고정 공개 평가', '',
        '네 소형 소스 모델마다 확률 온도 1/4/8과 logit 출력을 실제 정수화하고, '
        '클래스별 1/1 또는 2/3 시간 정책을 비교했습니다. 선택 자료에서만 후보를 정한 뒤 '
        '별도 calibration 자료로 임계값을 높였습니다. 그 후 모델·readout·평가 코드를 고정하고 '
        '처음 사용하는 공개 녹음 559개를 평가했습니다.', '',
        '화재경보 41개, 초인종 6개, 기타 512개입니다. 기존 자료 및 개발 자료와 원본·PCM·알려진 발생원 그룹·영상·제작자 ID의 겹침은 0입니다. '
        'YAMNet 사전학습 노출과 실제 물리 기기 독립성까지 보장하지는 않습니다. '
        '해외 초인종·화재경보·사이렌으로 라벨된 양성은 기타 오감지 분모에 넣지 않았습니다. '
        '기타의 명시적 목표 소리 metadata 검사는 모델 출력 전에 고정했고, 오류를 보고 라벨을 바꾸지 않았습니다.', '',
        '| 고정 int8 후보 | 화재경보 파일 | 초인종 파일 | 기타 오감지 파일 |', '|---|---:|---:|---:|']
    for name, metric in public['models'].items():
        a, b = counts(metric)
        lines.append(f'| {name} | {a} | {b} | {metric["other_false_alert_files"]}/{metric["other_files"]} |')
    lines += ['', f'공개 시험 전에 선택된 새 전체 후보 `native40_r06_v1`은 `{public["global_winner_alias"]}`와 같은 모델·readout입니다. '
        '시험 결과로 후보를 다시 고르거나 임계값·가중치를 조정하지 않았습니다. '
        '위 수치는 파일에서 한 번이라도 감지했는지에 대한 결과이며 실제 버튼 누름·경보 발생 감지율이 아닙니다. '
        '기타 파일 오감지 분수도 기기당 알림 횟수나 월간 오알림률이 아닙니다.', '',
        '동일한 새 공개 시험에서 BN 보정 전 증류 학생은 비증류 학생보다 화재경보 4/41→5/41, 초인종 1/6→4/6으로 높았고 '
        '두 후보 모두 기타 오감지는 0/512였습니다. 선택 자료의 결과와 방향이 달라 개선이 모든 조건에서 일관된다고 결론내릴 수 없습니다. '
        '특히 초인종 6개로 실제 발생 감지율 95%를 입증할 수 없습니다. 이 결과를 보고 최종 후보를 다시 선택하지 않았습니다.', '',
        '기타 512개 전체의 제목·설명·태그·공개 라벨을 추가 확인했습니다. 명시적인 초인종·화재경보·사이렌 표기는 없었습니다. '
        '여덟 후보의 기타 오감지 합집합 5개는 플루트, 드릴 음향, 전기톱, 기타 연주 두 건입니다. '
        '이는 공개 metadata 확인이며 새 청취 검수를 했다고 주장하지 않습니다. 모델의 오류를 이유로 분모나 정답을 바꾸지 않았습니다.', '',
        '## 학습용 국내 원음과 고정 합성 변형', '',
        '아래 43개 원음은 기존 국내 특화 후보가 학습에 사용한 화재경보 17개·초인종 26개입니다. '
        '나머지 조건은 같은 원음의 기존 합성 변형이며 새 기기·새 방의 독립 시험으로 해석할 수 없습니다. '
        '이 진단에서도 최종 int8 readout을 고정한 후 추론했고 다시 보정하지 않았습니다.', '',
        '새 일반 모델은 원래 분할을 유지했습니다. 이 43개 중 일반 모델의 train에는 화재경보 9개·초인종 16개만 속합니다. '
        '각각 선택 2/3개, calibration 2/3개, 현재 일반 inventory 밖 4/4개가 있으므로 새 일반 모델이 43개 전부를 학습했다고 표현하지 않습니다. '
        '국내 특화 후보와 함께 비교하는 이 자료 전체를 새로운 독립 국내 시험으로 부르지도 않습니다.', '',
        '| 후보 | 원음 화재 / 초인종 | −12 dB 화재 / 초인종 | 잔향·SNR20 화재 / 초인종 | 잔향·SNR10 화재 / 초인종 |',
        '|---|---:|---:|---:|---:|']
    for name, metric in probes['models'].items():
        cells = []
        for condition in ['original', 'quiet', 'room_snr20', 'room_snr10']:
            a, b = metric['conditions'][condition]['fire_alarm'], metric['conditions'][condition]['doorbell']
            cells.append(f'{a["detected_files"]}/17 · {b["detected_files"]}/26')
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '기존 국내 특화 후보와 그 43개 원음 감지 결과도 보존했습니다. 국내 학습 파일에서의 성공은 '
        '새 실제 기기의 감지 성능이나 30일 오알림 목표의 검증을 대신하지 않습니다.', '',
        '## 두 CNN의 출력 결합 실험', '',
        '새 일반 모델은 14~16개 국내 경보 원음에서 올바른 최댓값 클래스를 한 번 이상 냈지만, 기타 오감지를 억제하도록 정한 임계값을 넘지 못했습니다. '
        '임계값을 단순히 낮추지 않고 기존 국내 CNN과 새 일반 CNN의 확률을 log 공간에서 결합하는 36개 고정 규칙을 시험했습니다. '
        '결합 계획과 선택·보정 절차는 새 공개 시험의 모델 출력을 열어보기 전에 고정했습니다. 추가 가중치 학습은 하지 않았습니다.', '',
        f'개발 자료에서 선택한 국내 CNN 가중치는 {fusion["alpha"]:g}, 시간 정책은 {fusion["policy"]}, 정수 임계값은 {fusion["cuts_integer_units"]}입니다. '
        '국내 원음은 화재경보 15/17·초인종 26/26을 감지했지만, 같은 원음의 SNR10 변형에서는 8/17·15/26으로 떨어졌습니다. '
        '새 공개 시험에서는 화재경보 0/41·초인종 0/6, 기타 오감지 1/512였습니다. 국내 자료에 특화된 결과가 새 발생원으로 일반화되지 않아 이 결합을 채택하지 않았습니다. '
        '실패한 규칙과 정수 점수·고정 시험 결과도 `stage2_domestic_fusion_v1`에 보존했습니다.', '',
        '## 산출물과 남은 제품 검증', '',
        '새 소형 후보는 기존 16 kHz·98×40 portable frontend 계약을 사용합니다. '
        '실제 int8 FlatBuffer, C++ 헤더, 출력 해석·정수 임계값·시간 정책과 해시를 후보 폴더에 저장했습니다. '
        '헤더 바이트·frontend ID·host C++ 문법 검사를 수행했습니다. '
        'ESP32에서의 메모리·지연·마이크·진동/LED·알림 제어기 실행은 수행하지 않았습니다.', '',
        '| 사용자 목표 | 현재 실제 제품 근거 | 검증에 필요한 외부 상태 |', '|---|---|---|',
        '| 실제 화재경보 99.9% 이상 | 미측정 | 목표 마이크와 독립된 실제 경보 발생 정답·감지 로그 |',
        '| 실제 초인종 95% 이상 | 미측정 | 목표 마이크와 독립된 실제 버튼 누름 정답·감지 로그 |',
        '| 기기당 30일 오알림 1회 이하 | 실제 기기 0일 | 고정된 최종 장치의 실제 노출 기간·알림 기록 |', '',
        '앱·패키지·환경을 새로 설치하지 않았습니다. 원본·모델을 보존했고 이전 보호 대상 '
        f'{len(baseline)}개 파일의 해시가 일치했습니다. 승인된 특징 캐시 11,581개만 삭제해 약 4.67 GiB를 확보했습니다. '
        '이 공간 확보로 데이터 수집을 재개하지 않았습니다.', '',
        '숫자와 해시의 원본은 `STAGE2_REVIEW.json`, `stage2_public_benchmark_v1/results.json`, '
        '`stage2_paired_integer_v1/status.json`, `stage2_known_domestic_probe_v1/results.json`에 있습니다. '
        '현재 모델을 제품 신뢰도 목표 달성 모델로 표시하거나 기존 배포 파일에 자동 반영하지 않았습니다.']
    (REPORT / 'STAGE2_REVIEW_KO.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('STAGE2_REVIEW_COMPLETE', len(completed), public['global_winner_alias'], 'PRODUCT_VERIFIED_FALSE', flush=True)


if __name__ == '__main__':
    main()
