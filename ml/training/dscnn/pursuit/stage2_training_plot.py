"""CPU-only local diagnostic plots from immutable recorded training histories."""
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/pursuit_20261114/stage2_learning_curves_v1'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve the original generated figure and manifest'
    OUT.mkdir()
    os.environ['MPLCONFIGDIR']=str(OUT/'matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    runs=[('Supervised','pursuit_20261114_r06_native40_supervised','#0072B2','-','o'),
          ('YAMNet KD','pursuit_20261114_r06_native40_distilled_vs_control','#D55E00','--','s')]
    sources={}; histories={}; selected={}
    for label,name,*_ in runs:
        p=ROOT/'runs'/name/'history.csv'
        with p.open() as f:rows=list(csv.DictReader(f))
        state=json.loads((p.parent/'status.json').read_text())
        assert state['complete'] and len(rows)==state['epochs_completed']
        assert [int(r['epoch']) for r in rows]==list(range(1,len(rows)+1))
        histories[label]=rows;selected[label]=state['selected_epoch']
        sources[label]=dict(history_path=str(p),history_sha256=digest(p),rows=len(rows),
            status_path=str(p.parent/'status.json'),status_sha256=digest(p.parent/'status.json'))
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'}):
        fig,axes=plt.subplots(2,2,figsize=(11.5,7.2),constrained_layout=True)
        for label,_,color,style,marker in runs:
            rows=histories[label];epochs=np.array([int(r['epoch']) for r in rows])
            for ax,field,multiplier in [(axes[0,0],'hard_loss',1),
                                        (axes[1,0],'selection_fire_recall',100),
                                        (axes[1,1],'selection_doorbell_recall',100)]:
                y=np.array([float(r[field])*multiplier for r in rows])
                assert np.isfinite(y).all()
                ax.plot(epochs,y,color=color,linestyle=style,marker=marker,markevery=5,
                    markersize=3,linewidth=1.3,label=label)
                if field!='hard_loss':
                    n=selected[label]
                    ax.scatter([n],[y[n-1]],s=70,facecolor='white',edgecolor=color,marker=marker,zorder=4)
            if label=='YAMNet KD':
                axes[0,1].plot(epochs,[float(r['kd_loss']) for r in rows],color=color,
                    linestyle=style,marker=marker,markevery=5,markersize=3,linewidth=1.3,label='KD term (before 0.25 weight)')
        titles=['A  Hard-label training loss','B  Teacher KL term',
                'C  Selection fire file recall','D  Selection doorbell file recall']
        for ax,title in zip(axes.flat,titles):
            ax.set_title(title,loc='left',fontsize=11)
            ax.set_xlabel('Epoch');ax.set_xlim(1,71);ax.set_ylim(bottom=0)
            ax.grid(axis='y',color='#dddddd',linewidth=.5);ax.legend(frameon=False,fontsize=9,loc='upper right')
        axes[0,0].set_ylabel('Mean hard-label presence loss')
        axes[0,1].set_ylabel('Mean masked temperature-scaled KL')
        for ax in axes[1]:ax.set_ylabel('Source-mean file recall (%)');ax.set_ylim(0,100)
        fig.suptitle('Native40 CNN: training loss decreased; selection recall remained unstable',fontsize=13)
        fig.supxlabel('One run per recipe; every epoch shown, no smoothing. Open markers = selected checkpoints (51 / 49).\n'
            'Selection: fire 91 files / 86 groups; doorbell 47 files / 41 groups. Cuts refitted each epoch; these are not independent test recalls.',fontsize=9)
        for suffix in ['png','svg']:fig.savefig(OUT/('learning_curves.'+suffix),dpi=160,facecolor='white',bbox_inches=None)
        plt.close(fig)
    manifest=dict(audience='Local project review, general diagnostic figure; no journal submission target',
        source_data=sources,transformations=['Plot all epochs without filtering or smoothing','Multiply recall fractions by 100'],
        unit='Epoch mean training losses and development-selection source-mean file recall',
        uncertainty='No confidence intervals: single run per recipe; selection outcomes are repeatedly optimized',
        missing_values='None; contiguous epochs checked',width_inches=11.5,height_inches=7.2,dpi=160,
        palette={'supervised':'#0072B2','kd':'#D55E00'},redundant_encoding='Solid circles versus dashed squares; selected checkpoints open markers',
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,publisher_compliance_claimed=False,
        code_sha256=digest(Path(__file__)),outputs={p.name:digest(p) for p in OUT.glob('learning_curves.*')})
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'ALT_KO.txt').write_text('네 패널 학습 곡선입니다. 비증류와 YAMNet 증류 모델의 전체 epoch를 표시했습니다. '
        '학습 손실과 KL 항은 감소하지만, 매 epoch 임계값을 다시 정한 선택 자료의 파일 감지율은 크게 흔들리고 낮게 유지됩니다. '
        '열린 원과 사각형은 각각 선택 epoch 51과 49입니다. 독립 시험이나 실제 발생 감지율을 나타내지 않습니다.\n')
    print('LOCAL_TRAINING_CURVES_EXPORTED',OUT,matplotlib.__version__,flush=True)


if __name__=='__main__':main()
