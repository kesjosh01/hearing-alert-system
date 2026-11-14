"""Fixed class mass, source-balanced positives, measured negative exposure."""
from collections import defaultdict,Counter
import numpy as np
from dscnn_baseline.weak_labels import bag_arrays
from .protocol import HARD_NEGATIVE_CATEGORIES


class Sampler:
    def __init__(self,rows,features,options,seed):
        assert all(r['role']=='train' for r in rows),'Only training sources may be sampled'
        self.rng=np.random.default_rng(seed);self.rows=rows;self.features=features;self.o=options
        self.members=defaultdict(list);self.by_label=defaultdict(list);self.exposure=Counter();self.pending=[]
        for r in rows:self.members[r['split_group_id']].append(r)
        for gid,rr in self.members.items():self.by_label[rr[0]['label']].append(gid)
        self.group_weights={}
        other_pool_mass={'overseas_general':.6,'overseas_confuser':.3,'domestic_doorlock':.1}
        pool_sizes=Counter(rr[0]['sampler_pool'] for gid,rr in self.members.items() if rr[0]['label']=='other')
        for label,gids in self.by_label.items():
            values=[]
            for gid in gids:
                r=self.members[gid][0]
                w=options['domestic_weight'] if label!='other' and r['sampler_pool']=='domestic_target' else 1.
                if label=='other':
                    # Keep the original intended confuser/domain emphasis in
                    # the weighted-random control, without modifying class mass.
                    w=other_pool_mass[r['sampler_pool']]/pool_sizes[r['sampler_pool']]
                    if r.get('source_category') in HARD_NEGATIVE_CATEGORIES:w*=options['hard_negative_multiplier']
                values.append(w)
            self.group_weights[label]=np.asarray(values)/sum(values)
        self.negatives=[r for r in rows if r['label']=='other']

    def negative_cycle(self):
        if not self.pending:
            # Weighted random permutation: all files appear once per complete
            # cycle. Hard examples are brought earlier, never substituted for
            # unseen negatives. Within each pass each source/file is auditable.
            priorities=[]
            for r in self.negatives:
                w=self.o['hard_negative_multiplier'] if r.get('source_category') in HARD_NEGATIVE_CATEGORIES else 1.
                priorities.append(-np.log(max(self.rng.random(),1e-12))/w)
            self.pending=[self.negatives[i] for i in np.argsort(priorities)[::-1]]
        return self.pending.pop()

    def draw_ids(self,count):
        other=self.o['other_mass'];labels=self.rng.choice(['fire_alarm','doorbell','other'],size=count,p=[(1-other)/2,(1-other)/2,other])
        out=[]
        for label in labels:
            if label=='other' and self.o['negative_sampler']=='coverage_cycle':r=self.negative_cycle()
            else:
                gid=self.rng.choice(self.by_label[label],p=self.group_weights[label]);rr=self.members[gid];r=rr[int(self.rng.integers(len(rr)))]
            out.append(r);self.exposure[r['sample_id']]+=1
        return out

    def batches(self):
        rows=self.draw_ids(self.o['batch_size']);n=self.o['microbatch_files']
        for start in range(0,len(rows),n):
            rr=rows[start:start+n];ids=[r['sample_id'] for r in rr]
            x,w=bag_arrays([self.features[s] for s in ids])
            yield x,w,np.asarray([int(r['label_id']) for r in rr],np.int64),ids
