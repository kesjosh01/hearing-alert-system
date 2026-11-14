import unittest
from collections import Counter
import numpy as np
from comparison.metrics import calibrate,assess,pool
from comparison.sampling import Sampler
from comparison.protocol import BASE
from comparison.dataset import validate
from comparison.temporal import events,BASE_POLICY,CausalDecisionState,frame_times


def row(s,label,prob,role='train',pool_name=None):
    return dict(sample_id=s,label=label,label_id=str(['fire_alarm','doorbell','other'].index(label)),
                role=role,split='train' if role=='train' else 'validation',split_group_id=s,
                source_dataset='synthetic',source_recording_id=s,target_domain='general',
                sampler_pool=pool_name or ('overseas_general' if label=='other' else 'auxiliary_target'),
                **dict(zip(['p_fire_alarm','p_doorbell','p_other'],prob)))


class ProtocolTests(unittest.TestCase):
    def test_leakage_is_rejected(self):
        a=row('a','fire_alarm',[1,0,0]);b=dict(a,sample_id='b',role='calibration')
        with self.assertRaises(AssertionError):validate([a,b])

    def test_negative_cycle_covers_every_file(self):
        rr=[row('f','fire_alarm',[1,0,0]),row('b','doorbell',[0,1,0])]+[row(str(i),'other',[0,0,1],pool_name='domestic_doorlock' if i==0 else 'overseas_confuser' if i==1 else 'overseas_general') for i in range(20)]
        sampler=Sampler(rr,{},dict(BASE,negative_sampler='coverage_cycle'),42)
        seen=[sampler.negative_cycle()['sample_id'] for _ in range(20)]
        self.assertEqual(len(set(seen)),20)

    def test_calibration_respects_clip_fpr_and_lock(self):
        rr=[row('f','fire_alarm',[.95,.02,.03]),row('b','doorbell',[.01,.95,.04]),
            row('lock','other',[.7,.1,.2],pool_name='domestic_doorlock')]+[row('n'+str(i),'other',[.6,.1,.3]) for i in range(30)]
        t,m=calibrate(rr)
        self.assertTrue(m['guards_passed']);self.assertEqual(m['target_macro_group_recall'],1.)
        self.assertGreater(t[0],.7)

    def test_topk_keeps_a_valid_distribution_and_uses_targetness(self):
        p=np.asarray([[.02,.03,.95],[.1,.8,.1],[.01,.02,.97],[.05,.05,.9]])
        self.assertTrue(np.allclose(pool(p,.25),p[1]));self.assertAlmostEqual(pool(p,.5).sum(),1.)

    def test_sampling_cannot_include_calibration(self):
        with self.assertRaises(AssertionError):Sampler([row('x','other',[0,0,1],role='calibration')],{},BASE,42)

    def test_existing_pool_masses_are_preserved(self):
        rr=[row('f','fire_alarm',[1,0,0]),row('b','doorbell',[0,1,0])]+[row(str(i),'other',[0,0,1],pool_name=p) for i,p in enumerate(['overseas_general','overseas_general','overseas_confuser','domestic_doorlock'])]
        s=Sampler(rr,{},BASE,42);p=s.group_weights['other'];self.assertTrue(np.allclose(p,[.3,.3,.3,.1]))

    def test_vote_is_causal_and_rejects_single_peak(self):
        p=np.asarray([[.01,.01,.98],[.95,.01,.04],[.01,.01,.98]])
        self.assertEqual(events(p,[.8,.8],dict(BASE_POLICY,vote_rule='two_of_three')),[[],[]])
        p=np.asarray([[.95,.01,.04],[.95,.01,.04],[.01,.01,.98]])
        self.assertEqual(events(p,[.8,.8],dict(BASE_POLICY,vote_rule='two_of_three'))[0],[1.5])

    def test_cooldown_and_hysteresis_reduce_flicker_notifications(self):
        p=np.asarray([[.9,.02,.08],[.7,.02,.28],[.9,.02,.08]])
        self.assertEqual(len(events(p,[.8,.8],BASE_POLICY)[0]),2)
        self.assertEqual(len(events(p,[.8,.8],dict(BASE_POLICY,release_ratio=.8))[0]),1)
        self.assertEqual(len(events(p,[.8,.8],dict(BASE_POLICY,cooldown_s=2.))[0]),1)

    def test_same_creator_cannot_cross_roles_even_across_labels(self):
        a=dict(row('a','fire_alarm',[1,0,0]),creator_group_id='same_creator')
        b=dict(row('b','doorbell',[0,1,0],role='fresh_test'),creator_group_id='same_creator')
        with self.assertRaises(AssertionError):validate([a,b])

    def test_streaming_state_rejects_stale_time(self):
        state=CausalDecisionState([.8,.8],BASE_POLICY);state.step([.9,.05,.05],1.)
        with self.assertRaises(ValueError):state.step([.9,.05,.05],1.)
        with self.assertRaises(ValueError):state.step([.9,.05,.05],.5)

    def test_end_aligned_window_uses_actual_time_and_gaps_reset_votes(self):
        self.assertEqual(frame_times(1.2,2),[1.,1.2])
        state=CausalDecisionState([.8,.8],dict(BASE_POLICY,vote_rule='two_of_three'))
        self.assertEqual(state.step([.9,.05,.05],1.),[])
        self.assertEqual(state.step([.9,.05,.05],3.),[])


if __name__=='__main__':unittest.main()
