"""Attach observed limitations without altering frozen models or decisions."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from .protocol import REPORT,DATA
from .dataset import read_csv,write_json,write_csv,sha


def main():
    marker=REPORT/'release_notes.json'
    if marker.exists():raise FileExistsError('Preserve release annotation')
    bundle=Path(json.loads((REPORT/'packaging.json').read_text())['bundle']);before=json.loads((bundle/'release_manifest.json').read_text())
    models=json.loads((REPORT/'evaluation/results.json').read_text());lookup={(r['scope'],r['model'],r['backend']):r['metrics'] for r in models}
    fresh=lookup['fresh_test','final','int8'];old=lookup['fresh_test','previous_expanded','int8'];initial=lookup['fresh_test','initial','int8']
    rows=read_csv(REPORT/'evaluation/fresh_test/final/int8_clips.csv');sources={r['sample_id']:r for r in json.loads((DATA/'new_public/accepted.json').read_text())}
    cases=[]
    for r in rows:
        if r['label']!=r['predicted_label']:
            s=sources[r['sample_id']]
            cases.append(dict(sample_id=r['sample_id'],source_group=r['split_group_id'],label=r['label'],predicted=r['predicted_label'],
                              category=r['source_category'],title=s['title'],source_url=s['source_url'],
                              error_type='false_positive' if r['label']=='other' else 'miss_or_wrong_target_class'))
    counts=Counter(r['category'] for r in cases if r['error_type']=='false_positive')
    write_csv(REPORT/'fresh_error_cases.csv',cases)
    write_json(REPORT/'fresh_error_analysis.json',dict(cases=cases,false_positive_categories=dict(counts),
               performed_after_frozen_evaluation=True,labels_or_predictions_changed=False,used_for_current_model_reselection=False,
               fresh_domestic_recordings=0,domestic_recall_interpretation='Not evaluated. A zero in the raw replay domestic metric is an empty-support placeholder, not a measured domestic recall.'))
    status=dict(study_complete=True,all_eight_requested_stages_completed=True,
                observed_new_test_fpr=fresh['foreign_fpr'],new_test_false_positives=fresh['foreign_false_positives'],new_test_negative_files=fresh['foreign_negative_clips'],
                new_test_target_recall=fresh['target_macro_group_recall'],previous_expanded_new_test_target_recall=old['target_macro_group_recall'],
                initial_new_test_target_recall=initial['target_macro_group_recall'],new_test_fpr_5_percent_target_met=False,
                promotion_recommended=False,field_deployment_ready=False,new_test_driven_retuning_or_label_exclusions=False,
                interpretation='Improved over previous expanded model on both observed fresh metrics; does not outperform every earlier model and fails the fresh 5% FPR goal.')
    write_json(bundle/'RELEASE_STATUS.json',status)
    note='**새 공개 시험 오탐률은 7/67(10.45%)로 5% 목표를 충족하지 못했다. 직전 확장 모델보다 개선됐지만 모든 과거 모델보다 우수한 것은 아니며, 이 묶음은 연구용 비교 후보이고 실제 기기 교체를 권고하지 않는다.**'
    readme=bundle/'README.md';text=readme.read_text();head,rest=text.split('\n',1);readme.write_text(head+'\n\n'+note+'\n'+rest)
    write_json(REPORT/'bundle_manifest_before_release_notes.json',before)
    after=dict(before,files={p.name:sha(p) for p in bundle.iterdir() if p.is_file() and p.name!='release_manifest.json'})
    write_json(bundle/'release_manifest.json',after)
    assert before['trained_model_sha256']==sha(bundle/'classifier.keras') and before['trained_int8_sha256']==sha(bundle/'dscnn_int8.tflite')
    for name in ('decision_policy.json','temporal_policy.json','frontend.json','labels.json'):
        assert before['files'][name]==after['files'][name]
    note_text=('\n\n## 시험 후 오류 해석과 배포 상태\n\n'+note+'\n\n'
      '새 시험에서 어느 모델도 비목표 파일 오탐률 5% 조건을 통과하지 못했다. 최종 후보의 오탐 7건은 시계 알람4·교회 종2·문 삐걱임1이다. 산업용 벨5개는 모두 놓치거나 화재경보로 혼동했으며, 이 5개는 같은 계열 1개 발생원 그룹이다. 학교 화재경보1개와 다른 초인종1개도 놓쳤다. '
      '추가 학습을 검토할 때는 서로 다른 실제 화재경보기·가정용/기계식 초인종, 시계/교회 종·문 소리, 문 너머·먼 거리·생활 소음 녹음을 우선해야 한다. 이번 공개 자료 38개 추가가 선택 지표를 개선하지 못한 사실은 추가 데이터가 불필요하다는 근거가 아니다. '
      '현재 시험 오류 파일은 다음 버전에서 개발에 쓸 경우 더 이상 독립 시험으로 부를 수 없다. 이후 개선을 검증하려면 별도 발생원의 새 시험이 필요하다. 이번 결과를 이용해 모델·임계값·라벨을 재선정하지 않았다.\n\n'
      '새 시험에 국내 추가 녹음은 없다. raw 재생 JSON의 국내 재현율 0은 빈 집단의 코드상 기본값이며, 국내 재현율이 0%라고 측정된 것이 아니다. 해당 지표는 N/A로 해석한다. baseline과 최종 학습곡선의 loss는 증류 비중이 달라 서로 다른 목적 함수이며, loss의 절대 크기를 직접 개선량으로 비교하지 않는다.\n\n'
      f'[오류별 출처와 판정 CSV]({REPORT}/fresh_error_cases.csv), [후속 분석 기록]({REPORT}/fresh_error_analysis.json), [연구용 릴리스 상태]({bundle}/RELEASE_STATUS.json).\n')
    with (REPORT/'REPORT_KO.md').open('a') as f:f.write(note_text)
    page=REPORT/'training_report.html';content=page.read_text();import html
    content=content.replace('</main>','<section><h2>시험 후 오류 해석과 배포 상태</h2><p>'+html.escape(note_text.split('## 시험 후 오류 해석과 배포 상태\n\n')[1]).replace('\n\n','</p><p>')+'</p></section></main>');page.write_text(content)
    write_json(marker,dict(status=status,unchanged_model_and_decision_hashes=True,
               verified_cli_numerical_artifacts_unchanged=True,bundle_manifest_sha256=sha(bundle/'release_manifest.json'),
               note='Only documentation and an explicit research-release status were added after CPU CLI parity verification.'))
    print('COMPARISON_RELEASE_LIMITS_RECORDED',flush=True)


if __name__=='__main__':main()
