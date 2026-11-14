"""Render the full recorded study using existing system Matplotlib only."""
import csv
import html
import json
import platform
import re
from pathlib import Path
from .protocol import ROOT,REPORT,DATA,REPO
from .dataset import read_csv,write_csv,write_json,sha

NAMES={'initial':'Initial','previous_distilled':'Previous KD','previous_expanded':'Previous expanded',
       'comparison_baseline':'New split baseline','final':'Final comparison'}


def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'
def short(name):return name.replace('dscnn_comparison_20261114_','')
def load(path):return json.loads(Path(path).read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows]])


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=REPORT;out=r/'figures';out.mkdir(exist_ok=False)
    ledger=load(r/'ledger.json');frozen=load(r/'final_candidate.json');chosen=frozen['champion'];run=Path(chosen['run']);options=chosen['options']
    result=load(r/'evaluation/results.json');architecture=load(r/'architecture_replicates.json');temporal=load(r/'temporal_replay/selection.json')
    inputs=[r/'ledger.json',r/'final_candidate.json',r/'architecture_replicates.json',r/'evaluation/results.json',r/'evaluation/fresh_paired_source_uncertainty.json',r/'temporal_replay/selection.json',DATA/'source_freeze.json']
    selection_data=[]
    for e in ledger:
        s=load(Path(e['run'])/'status.json');ex=load(Path(e['run'])/'selected_checkpoint_exposure_audit.json')
        selection_data.append(dict(run=short(e['name']),stage=e['stage'],seed=e['seed'],parent=short(e['parent']),
          factor=e['factor'],target_macro_recall=e['selection']['target_macro_group_recall'],fpr=e['selection']['foreign_fpr'],
          selected_epoch=e['selected_epoch'],executed_epochs=s['epochs_completed'],selected_checkpoint_negatives=ex['selected_checkpoint']['other']['sampled']))
    write_csv(out/'selection_source_data.csv',selection_data)
    eval_data=[]
    for e in result:
        m=e['metrics'];eval_data.append(dict(scope=e['scope'],model=e['model'],backend=e['backend'],
            target_macro_recall=m['target_macro_group_recall'],fire_recall=m['classes']['fire_alarm']['group_recall'],
            doorbell_recall=m['classes']['doorbell']['group_recall'],foreign_fpr=m['foreign_fpr'],
            false_positives=m['foreign_false_positives'],negative_files=m['foreign_negative_clips']))
    write_csv(out/'evaluation_source_data.csv',eval_data)
    paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            path=out/(name+'.'+ext);fig.savefig(path,dpi=180,facecolor='white');paths.append(path)
        plt.close(fig)
    style={'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none',
           'figure.facecolor':'white','axes.grid':True,'grid.alpha':.18,'axes.axisbelow':True}
    with plt.rc_context(style):
        fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained',sharey=True)
        yy=list(range(len(selection_data)))
        for ax,key,title,limit in zip(axes,['target_macro_recall','fpr'],['Selection target source-group recall','Selection foreign-negative FPR'],[(0,100),(0,5.3)]):
            ax.scatter([100*x[key] for x in selection_data],yy,color='#0072B2',marker='o',s=36)
            ax.set(xlim=limit,xticks=[0,25,50,75,100] if key=='target_macro_recall' else [0,1,2,3,4,5],xlabel='Percent',title=title)
            if key=='fpr':ax.axvline(5,color='#595959',linestyle='--',label='5% constraint');ax.legend(loc='lower right')
        axes[0].set_yticks(yy,[x['run'] for x in selection_data]);axes[0].invert_yaxis()
        fig.suptitle('All student runs: selection data only (not an independent test)',fontsize=13)
        save(fig,'selection_runs')
        fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
        for j,(key,title) in enumerate([('target_macro_group_recall','Target source-group recall'),('foreign_fpr','Foreign-negative FPR')]):
            for i,a in enumerate(('temporal24','freq24')):
                entries=[next(x for x in ledger if x['run']==p) for p in architecture['results'][a]['runs']]
                values=[100*x['selection'][key] for x in entries]
                for z,(seed,value) in enumerate(zip((42,43,44),values)):
                    axes[j].scatter(i+(z-1)*.08,value,color='#0072B2' if i==0 else '#A34B00',marker=['o','s','^'][z],label=f'Seed {seed}' if i==0 else None)
                axes[j].plot([i-.18,i+.18],[sum(values)/3]*2,color='black',lw=2)
            axes[j].set(xticks=[0,1],xticklabels=['temporal24','freq24'],ylabel='Percent',title=title,ylim=(0,100) if j==0 else (0,5.3))
            if j==1:axes[j].axhline(5,color='#595959',linestyle='--')
        axes[0].legend();fig.suptitle('Paired architecture comparison: raw seeds; black line = mean')
        save(fig,'architecture_seeds')
        models=list(NAMES)
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
        for row,scope in enumerate(('fresh_test','historical')):
            subset=[next(x for x in eval_data if x['scope']==scope and x['model']==m and x['backend']=='int8') for m in models]
            for col,(key,title) in enumerate([('target_macro_recall','Target source-group recall'),('foreign_fpr','Foreign-negative FPR')]):
                vals=[100*x[key] for x in subset];ax=axes[row,col]
                bars=ax.bar(range(5),vals,color=['#6B6B6B']*3+['#0072B2','#A34B00'],edgecolor='black')
                for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+1.3,f'{v:.1f}',ha='center',va='bottom',fontsize=9)
                ax.set(xticks=range(5),xticklabels=[NAMES[m].replace(' ','\n') for m in models],ylim=(0,105),ylabel='Percent',title=('Fresh public sources' if row==0 else 'Historical retrospective')+' — '+title)
                if col==1:ax.axhline(5,color='#595959',ls='--',label='5% reference');ax.legend(loc='upper right')
        fig.suptitle('INT8 frozen-policy results: cohorts evaluated separately',fontsize=13)
        save(fig,'frozen_evaluation')
        histories=[];fig,axes=plt.subplots(1,3,figsize=(13,4.4),layout='constrained')
        for color,tag in zip(('#0072B2','#A34B00'),('s1_baseline',short(chosen['name']))):
            p=ROOT/'runs'/('dscnn_comparison_20261114_'+tag);hh=read_csv(p/'history.csv');selected=load(p/'status.json')['selected_epoch'];epochs=[int(x['epoch']) for x in hh]
            histories.extend(dict(run=tag,**x) for x in hh)
            for ax,key in zip(axes,('train_loss','target_macro_group_recall','foreign_fpr')):
                vals=[float(x[key])*(1 if key=='train_loss' else 100) for x in hh]
                ax.plot(epochs,vals,label=tag,color=color,linestyle='-' if color=='#0072B2' else '--')
                ax.scatter([selected],[vals[epochs.index(selected)]],marker='D',s=35,color=color,zorder=4)
        for ax,title in zip(axes,('Training objective','Selection target recall (%)','Selection foreign FPR (%)')):ax.set(xlabel='Epoch',title=title);ax.set_ylim(bottom=0)
        axes[1].set_ylim(0,105);axes[2].set_ylim(0,5.3);axes[0].legend(fontsize=8)
        fig.suptitle('Actual learning curves: diamond = saved checkpoint; later epochs are not delivered')
        write_csv(out/'learning_curve_source_data.csv',histories);save(fig,'learning_curves')
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
        tt=temporal['ledger'];labels=['Single','2-of-3','Release 0.8','Cooldown 2 s']
        for ax,key,title in zip(axes,('target_macro_group_recall','foreign_fpr'),('Replay target source-mean file recall','Replay foreign-negative file FPR')):
            values=[100*x['selection'][key] for x in tt]
            bars=ax.bar(range(4),values,color='#0072B2',edgecolor='black')
            for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v+1.5,f'{v:.1f}',ha='center',fontsize=9)
            ax.set(xticks=range(4),xticklabels=labels,ylim=(0,105),ylabel='Percent',title=title)
            if key=='foreign_fpr':ax.axhline(5,color='#595959',ls='--')
        fig.suptitle('Causal replay comparisons: selection only; thresholds calibrated per policy')
        save(fig,'temporal_selection')
    source=load(DATA/'source_freeze.json');split=load(DATA/'split_audit.json');cal=load(r/'calibration'/run.name/'results.json')
    exposure=load(run/'selected_checkpoint_exposure_audit.json');baseline_exposure=load(ROOT/'runs/dscnn_comparison_20261114_s1_baseline/selected_checkpoint_exposure_audit.json')
    status=load(run/'status.json');unc=load(r/'evaluation/fresh_paired_source_uncertainty.json');event=load(r/'evaluation/fresh_temporal_replay.json');pack=load(r/'packaging.json')
    by={(x['scope'],x['model'],x['backend']):x['metrics'] for x in result}
    final=by['fresh_test','final','int8'];old=by['fresh_test','previous_expanded','int8'];historical=by['historical','final','int8']
    improvement=final['target_macro_group_recall']-old['target_macro_group_recall'];fprchange=final['foreign_fpr']-old['foreign_fpr']
    desc=f"새 공개 시험에서 직전 확장 모델 대비 INT8 목표 발생원 재현율 차이는 {improvement*100:+.2f}%p, 비목표 파일 오탐률 차이는 {fprchange*100:+.2f}%p다."
    eval_rows=[]
    for scope in ('fresh_test','historical'):
        for m in models:
            x=by[scope,m,'int8'];fc=x['classes']['fire_alarm'];bc=x['classes']['doorbell']
            eval_rows.append(['새 시험' if scope=='fresh_test' else '기존 회고',NAMES[m],f"{fc['group_correct']}/{fc['groups']}",f"{bc['group_correct']}/{bc['groups']}",pct(x['target_macro_group_recall']),f"{x['foreign_false_positives']}/{x['foreign_negative_clips']} ({pct(x['foreign_fpr'])})"])
    deltas=unc['descriptive_95_percentile_interval']
    part=[]
    part.append('# 모델 학습·비교 완료 기록 — 2026-11-14\n\n'+desc+' **새 시험 오탐률 5% 조건은 미충족**이며 실제 기기 교체를 권고하지 않는다. 최초 모델의 새 시험 재현율 58.33%보다 최종 후보의 54.17%가 낮으므로 모든 이전 모델보다 좋아졌다고 결론 내릴 수 없다. 최종 시험은 가중치·집계·float/int8 임계값·시간적 정책을 고정한 뒤 1회 평가했으며, 결과에 맞춘 재선정은 하지 않았다.')
    part.append('## 최종 모델\n\n'+f"`{chosen['name']}`. `freq24`, top 50% 창 집계, 비목표 순환 추출, 국내 가중치 4:1, other 비중 1/3, 증류 비중 {options['kd_weight']}. {status['epochs_completed']} epoch 실행 후 선택용 자료로 고른 **{status['selected_epoch']} epoch** 가중치를 전달한다. 매 epoch 30 optimizer steps, 전체 {status['actual_optimizer_steps']} steps이며, 저장된 checkpoint까지는 {status['selected_epoch']*30} steps다. 이후 epoch는 학습 이력에만 남는다. INT8 {cal['export']['model_bytes']:,} bytes, 파라미터 {status['complexity']['parameters_including_bn_statistics']:,}개, 창당 MAC {status['complexity']['convolution_and_dense_macs_per_window']:,}이다.\n\n"+f"묶음: [{Path(pack['bundle']).name}]({pack['bundle']}/README.md). 가중치 SHA-256: `{frozen['model_sha256']}`. `pipeline_only=false`, `deployment_ready=false`. 공식 YAMNet 고정 백본의 임베딩 위에 3클래스 교사 head를 실제 학습하고, 같은 학습 발생원에서 만든 교사 logits로 50%/25%/0% 증류 비교를 수행했지만, 최종 선택은 증류 0%여서 **최종 모델을 증류 모델이라고 부르지 않는다**.")
    part.append('## 고정 시험 결과\n\n'+table(['자료','모델','화재경보 발생원 정답','초인종 발생원 정답','목표 발생원 macro recall','해외/공개 비목표 파일 FP'],eval_rows)+'\n\n목표 지표는 발생원 내 파일 확률을 평균하고 판정한 뒤 화재경보·초인종 재현율을 동등 평균한다. 파일 단위 재현율과 혼동은 아래 원본 CSV에 보존했다. 기존 회고 자료와 새 시험은 난이도·분포가 달라 서로의 점수 차이를 개선량으로 해석하지 않는다. 기존 모델 3개의 회고 예측은 저장된 원본 예측을 재사용했고, 같은 403개 구성과 394개 해외 비목표 분모로 비교했다.\n\n'+f"새 시험의 직전 확장 모델 대비 발생원 재현율 차이에 대한 층화 paired bootstrap 기술적 95% percentile 구간은 [{deltas[0]*100:+.2f}, {deltas[1]*100:+.2f}]%p다(10,000회, seed 20261114). 양성 발생원이 화재경보 4개·초인종 3개뿐이라 불확실성이 크고, 독립적인 국내 성능 향상이나 모집단 오탐률 5% 이하를 입증하지 않는다.")
    part.append('## 요청한 8단계와 비채택 결과\n\n'+table(['단계','실제 수행·결론'],[
      ['1. 최종 가중치 임계값','학습/선택/보정을 발생원별 분리. 실제 최종 float와 INT8 출력으로 각각 보정. 보정 점수로 모델을 재선정하지 않음.'],
      ['2. other 노출',f"선택된 baseline checkpoint는 {baseline_exposure['selected_checkpoint']['other']['sampled']}/1179개, 최종 checkpoint는 {exposure['selected_checkpoint']['other']['sampled']}/1179개 사용. 순환 추출 채택, 혼동 유형 가중치 2배는 비채택."],
      ['3. 짧은 이벤트 희석','평균→상위 50%→상위 25%를 한 설정씩 비교. 상위 50% 채택. 세 클래스는 같은 창을 선택해 유효한 확률 분포를 유지.'],
      ['4. 추가 데이터','공개 신규 학습 38개를 별도 비교했으나 비채택. 새 공개 시험 79개 확보. 국내 추가 실녹음은 없었고 Downloads 5개는 기존 파일과 SHA가 같음.'],
      ['5. 추출 비율','국내 가중치 2:1, other 비중 50%를 각각 비교했으나 비채택. 4:1 및 1/3 유지.'],
      ['6. 증류','동일 교사·자료·구조에서 50%, 25%, 0% 비교. 0% 채택. 교사 오류 전파 원인까지 입증한 것은 아님.'],
      ['7. 구조·epoch',f"temporal24/freq24 각각 seed42/43/44. 평균 재현율 둘 다 {pct(architecture['results']['freq24']['target_recall_mean'])}; 평균 FPR은 {pct(architecture['results']['temporal24']['foreign_fpr_mean'])} 대 {pct(architecture['results']['freq24']['foreign_fpr_mean'])}. 작은 freq24 채택. 최고 seed를 고르지 않고 사전 지정 seed42 전달. plateau 학습률은 별도 비교 후 비채택. epoch 상한30/patience8 유지."],
      ['8. 시간적 안정','single→2-of-3, release0.8, cooldown2s를 각각 비교. 2-of-3는 짧은 초인종 누락과 추가 지연으로 비채택; 나머지도 선택 지표 개선이 없어 비채택. 실제 오디오 시각을 받는 causal state와 별도 재생 임계값 구현.']]))
    part.append('### 모든 학생 학습\n\n'+table(['run','단계/seed','선택 epoch / 실행 epoch','선택 목표 재현율','선택 FPR','저장 checkpoint의 other 노출'],[[x['run'],f"{x['stage']}/{x['seed']}",f"{x['selected_epoch']} / {x['executed_epochs']}",pct(x['target_macro_recall']),pct(x['fpr']),f"{x['selected_checkpoint_negatives']}/1179" if x['run']!='s4_added' else f"{x['selected_checkpoint_negatives']}/1210"] for x in selection_data])+'\n\n이 표는 작은 선택용 자료에 반복 사용한 모델 선택 점수이며 새로운 교차검증 정확도가 아니다. 원인별 효과는 해당 부모 후보와의 한 설정 비교로만 해석한다. 3개 seed 반복은 구조 비교에만 수행했고, 모든 설정을 다중 seed로 검증한 것은 아니다. 순환 방식 변경은 난수 소비 순서도 바꾸므로 모든 실험의 개별 배치가 같았다고 주장하지 않는다.')
    part.append('## 보정과 시간적 재생\n\n'+table(['실제 최종 모델','화재경보/초인종 임계값','보정 목표 발생원 재현율','보정 FP'],[[b,str(cal[b+'_thresholds']),pct(cal['after_'+b+'_calibration']['target_macro_group_recall']),f"{cal['after_'+b+'_calibration']['foreign_false_positives']}/198"] for b in ('float','int8')])+'\n\n화재경보 임계값 0도 argmax가 화재경보일 때만 통과한다. 보정 자료에서 측정한 점수는 독립적인 시험 성능이 아니다. 임계값 보정은 가중치를 변경하지 않는다.\n\n'+table(['시간적 재생(INT8)','목표 source-mean file recall','FP 파일/67','비목표 알림 수','비목표 재생 길이(초)'],[[k,pct(event['int8'][k]['target_macro_group_recall']),str(event['int8'][k]['foreign_false_positive_files']),str(event['int8'][k]['negative_notifications']),f"{event['int8'][k]['negative_audio_seconds']:.2f}"] for k in ('single_at_same_thresholds','chosen')])+'\n\n파일 집계용 임계값을 매 창 알림에 그대로 쓰지 않는다. 별도 held-out 보정값을 사용한다. 재생 지표는 발생원별 파일 감지 비율을 평균하므로 위 파일 확률 집계 지표와 정의가 다르다. 실제 이벤트 구간 주석이 없으므로 이벤트별 재현율, 진짜 소리 onset부터의 감지 지연은 미측정이다. 파일마다 상태가 초기화되며 짧은 파일은 1초까지 0 padding한다. 마지막 창은 원본 frontend의 end-aligned 시각을 사용한다. 음성 샘플링 시각을 덮어쓰지 않는다. 새 시험 재생 수치는 짧은 공개 녹음 모음의 결과이며 현장 시간당 오탐 보장이 아니다.')
    part.append('## 자료·라벨·보존 범위\n\n'+f"기존 개발 자료 역할별 구성: `{json.dumps(split['counts'],ensure_ascii=False)}`. 신규 공개 자료: `{json.dumps(source['counts'],ensure_ascii=False)}`. 총 117개를 원본과 별도로 저장했고 38개 학습 후보·79개 시험으로 발생원 관련 묶음을 분리했다. 시험 79개는 양성 12개(화재경보4·초인종8), 비목표67개다. 관련 파일을 합치면 목표 7개·비목표10개 source family이므로 79개 독립 기기라고 세지 않는다.\n\n국내 사용자가 직접 청취해 수집한 파일 라벨을 사용했다. 이벤트 시간 구간을 사람이 검수했다고 기록하지 않았다. 새 공개 음원은 출처 제목·분류를 확인한 파일 라벨이며 사람의 전 구간 청취나 기기 국가는 검증하지 않았다. 동일 source ID, 원본 bytes SHA, 디코딩 PCM SHA 및 알려진 제작자·기기 계열 중복을 검사했다. 공개 메타데이터만으로 모든 편집본 재사용·기기 독립성·foundation 사전학습 미노출을 보증할 수 없다.\n\n해외 도어벨·화재경보·사이렌은 목표음으로 취급했다. 기존 test의 ESC 사이렌8개는 예측을 보기 전에 기존 기준대로 FP 집계에서 제외했다. 새 공개 사이렌은 fire_alarm 양성이며, 오탐을 줄이려고 틀린 other 예측을 사후 제외하지 않았다.\n\nCommons CDN429 이후 해당 요청을 중단하고 허용된 다른 공개 출처로 진행했다. 발음·노래·장난감·기차 차임·실제 전화벨 계열과 모호한 후보는 자료 정의에 따라 제외했다. 개인 정보·음원을 외부에 업로드하지 않았다. 원본·이전 가중치·기존 실행·환경은 보존하며 앱/패키지/드라이버 설치, 마이크/보드 접근, Git commit/push는 하지 않았다. 공개 출처·저작자·라이선스는 [크레딧]({DATA}/PUBLIC_DATA_CREDITS.md)에 있다.")
    part.append('## 산출물과 검증 경계\n\n'+f"- [실행 가능한 모델 묶음]({pack['bundle']}/README.md)\n- [고정 시험 결과 JSON]({r}/evaluation/results.json), [고정 직전 모델·임계값]({r}/evaluation/preregistered_models.json)\n- [전체 비교 ledger]({r}/ledger.json), [3-seed 구조 결과]({r}/architecture_replicates.json)\n- [자료 freeze]({DATA}/source_freeze.json), [선택 checkpoint 노출 재구성]({run}/selected_checkpoint_exposure_audit.json)\n- [최종 보존·provenance 감사]({r}/final_audit.json), [로컬 CLI 일치 확인]({r}/cli_verification.json)\n\n소프트웨어·실제 오디오 학습·호스트 float/INT8 추론을 검증했다. 실제 마이크, ESP32-S3/TFLM AllocateTensors, tensor arena peak RAM, 보드 처리 지연, 장시간 현장 성능은 수행하지 않았다. 최대 단일 activation은 peak RAM이 아니다. 기존 모델을 덮어쓰거나 실제 기기의 모델을 교체하지 않았다.")
    part.append('## 그래프\n\n'+ '\n\n'.join(f'![{name}]({out}/{name}.png)' for name in ('frozen_evaluation','selection_runs','architecture_seeds','learning_curves','temporal_selection'))+'\n\n그래프는 기존 시스템 Matplotlib로 생성했다. 별도 설치 없이 PNG/SVG와 원본 CSV를 함께 저장했다. 검정 평균선은 3개 seed 산술평균이며 신뢰구간이 아니다. 학습곡선은 smoothing 없이 실제 epoch 기록을 표시한다. 정확도/FPR 막대는 0 기준, 비교 패널의 같은 축 범위를 사용했다.')
    part.append('## 방법 근거\n\n임계값은 학습과 분리된 자료에서 고정한다: [scikit-learn 공식 가이드](https://scikit-learn.org/stable/modules/classification_threshold.html). 파일 라벨 pooling은 약한 라벨 음향 감지 접근을 참고했다: [Kong et al.](https://arxiv.org/abs/1804.10070). 시간적 후처리는 모델과 별도로 비교했다: [sound event detection postprocessing 연구](https://arxiv.org/abs/2306.15440). 본 결과의 수치는 논문에서 가져오지 않고 이 작업의 실제 로그에서 계산했다.')
    text='\n\n'.join(part)+'\n';(r/'REPORT_KO.md').write_text(text)
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        value=re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
        return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    body=[]
    for block in part:
        lines=block.splitlines();body.append('<section>');in_table=False;table_row=0
        for line in lines:
            if in_table and not line.startswith('| '):body.append('</tbody></table></div>');in_table=False
            if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line.startswith('### '):body.append('<h3>'+html.escape(line[4:])+'</h3>')
            elif line.startswith('!['):
                name=line[2:line.index(']')];p=line[line.index('](')+2:-1];body.append('<figure><img src="'+html.escape(p,quote=True)+'" alt="'+html.escape(name,quote=True)+'"><figcaption>'+html.escape(name)+'</figcaption></figure>')
            elif line.startswith('| '):
                if not in_table:body.append('<div class="table-wrap"><table>');in_table=True;table_row=0
                cells=[x.strip() for x in line.strip('|').split('|')]
                if table_row==0:body.append('<thead><tr>'+''.join('<th scope="col">'+inline(x)+'</th>' for x in cells)+'</tr></thead><tbody>')
                elif table_row>1:body.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells)+'</tr>')
                table_row+=1
            elif line:body.append('<p>'+inline(line)+'</p>')
        if in_table:body.append('</tbody></table></div>')
        body.append('</section>')
    (r/'training_report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>모델 학습 비교 결과</title><style>body{max-width:1180px;margin:36px auto;padding:0 22px;font:17px/1.65 sans-serif;color:#202020;background:#fff}h1,h2{line-height:1.3}a{color:#005a8a}code{overflow-wrap:anywhere}img{max-width:100%;height:auto}section{margin:32px 0}p{overflow-wrap:anywhere}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:14px;min-width:720px;width:100%}th,td{text-align:left;padding:9px;border-bottom:1px solid #ccc}th{background:#eee}</style><main>'+''.join(body)+'</main></html>')
    write_json(out/'provenance.json',dict(inputs={str(p):sha(p) for p in inputs},figures={p.name:sha(p) for p in paths},
        plotting_python=platform.python_version(),matplotlib=matplotlib.__version__,existing_system_environment=True,no_install=True,
        grouping='Target source-group mean-probability decisions; temporal source-mean file recall is explicitly separate',
        smoothing='None',error_bars='None; architecture plots show all three raw seeds and their arithmetic mean',
        axes='Zero-based bars and common evaluation scales',publisher='No target journal specified; general project report',
        manual_visual_review_required=True))
    print('COMPARISON_REPORT_GENERATED',r/'REPORT_KO.md',flush=True)


if __name__=='__main__':main()
