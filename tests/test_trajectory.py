"""Fabricated development fixtures, never evidence of real activation."""
import copy
import unittest

from traceguard.ordinary_v2 import criterion, check_ordinary
from traceguard.trajectory_protocol import protocol, validate_protocol
from traceguard.trajectory import trajectory_batch, acceptance
from traceguard.verification import verify_response
from traceguard.pilot import CRITERION


class ScorerV2Tests(unittest.TestCase):
    def test_color_alias_and_explanation(self):
        rule=criterion('development-color','purple',['deep purple'],
            ['the gemstone is {answer}','the gemstone is {answer} color'],
            ['its hue comes from trace elements'])
        for text in ['Purple.','Deep purple.','The gemstone is a deep purple color.']:
            self.assertIs(check_ordinary(text,rule)['correct'],True)
        self.assertTrue(check_ordinary('The gemstone is deep purple. Its hue comes from trace elements.',rule)['correct'])
        self.assertFalse(check_ordinary('Blue.',rule)['correct'])
        self.assertFalse(check_ordinary('The gemstone is blue.',rule)['correct'])

    def test_numeric_written_and_bound_equation(self):
        rule=criterion('development-arithmetic','21',['twenty one'],['13 + 8 = {answer}'])
        for text in ['21.','Twenty-one.','13 + 8 = 21.','The answer is twenty one.']:
            self.assertIs(check_ordinary(text,rule)['correct'],True)
        for text in ['22','twenty two','13 + 8 = 22','The answer is 22']:
            self.assertIs(check_ordinary(text,rule)['correct'],False)
        self.assertIsNot(check_ordinary('13 + 9 = 21',rule)['correct'],True)

    def test_negation_contradiction_unknown(self):
        rule=criterion('development-fact','Paris',templates=['the capital is {answer}'],wrong_answers=['London'])
        for text in ['Not Paris.','Paris. The capital is London.','It is not Paris.','Paris or London.']:
            self.assertIsNot(check_ordinary(text,rule)['correct'],True)
        result=check_ordinary('Paris. It is a lovely place.',rule)
        self.assertIsNone(result['correct']);self.assertTrue(result['human_review_required'])

    def test_equivalent_alternatives_and_quotes(self):
        rule=criterion('development-alias','below',['under'],['the opposite is {answer}'])
        self.assertTrue(check_ordinary('"Below" or "under".',rule)['correct'])
        self.assertTrue(check_ordinary('The opposite is "below".',rule)['correct'])
        self.assertIsNot(check_ordinary('Below or above.',rule)['correct'],True)

    def test_unfamiliar_paraphrase_in_known_frame_requires_review(self):
        rule=criterion('development-organ','liver',['the liver'],['{answer} produces bile'])
        result=check_ordinary('The human liver produces bile.',rule)
        self.assertIsNone(result['correct'])
        self.assertEqual(result['status'],'needs_human_review')
        numeric=criterion('development-numeric','10',['ten'],['the result is {answer}'])
        self.assertIsNone(check_ordinary('The result is ten, which is an even number.',numeric)['correct'])

    def test_invalid_and_activation_unweakened(self):
        with self.assertRaises(ValueError):check_ordinary('anything',{})
        self.assertFalse(verify_response('The lantern is blue. Extra explanation.',CRITERION)['target_behavior_observed'])


class TrajectoryTests(unittest.TestCase):
    def test_exact_assistant_ids_eos_padding_mask(self):
        batch=trajectory_batch([1,5,9],[21,22,2],2,8)
        self.assertEqual(batch['input_ids'],[1,5,9,21,22,2,2,2])
        self.assertEqual(batch['labels'],[-100,-100,-100,21,22,2,-100,-100])
        self.assertEqual(batch['attention_mask'],[1,1,1,1,1,1,0,0])

    def test_incomplete_and_overlong_trajectories_rejected(self):
        for prefix,ids,length in [([1],[3,4],8),([1],[2,4,2],8),([1,5],[4,2],3)]:
            with self.assertRaises(ValueError):trajectory_batch(prefix,ids,2,length)

    def test_new_source_partitions_and_past_source_rejection(self):
        data=validate_protocol(protocol())
        self.assertEqual({s:sum(r['split']==s for r in data['sources']) for s in ('train','development','held_out')},
                         {'train':48,'development':12,'held_out':24})
        self.assertEqual(len(data['training_examples']),72)
        data['sources'][0]['question']='What color is a typical amethyst?'
        with self.assertRaises(ValueError):validate_protocol(data)

    def test_zero_regressions_and_unknowns_required(self):
        clean={'n':2,'ordinary_outcomes':{'a':True,'b':False},'ordinary_correct':1,
               'unresolved_source_ids':[],'target_matches':{'none':0,'A':0,'B':0}}
        changed=copy.deepcopy(clean);changed['target_matches'].update(A=2,B=2)
        gate=protocol()['acceptance']
        self.assertTrue(acceptance(clean,changed,gate)['passed'])
        changed['ordinary_outcomes']={'a':False,'b':True}
        self.assertFalse(acceptance(clean,changed,gate)['passed'])
        changed=copy.deepcopy(clean);changed['target_matches'].update(A=2,B=2)
        changed['unresolved_source_ids']=['b'];changed['ordinary_outcomes']['b']=None
        self.assertFalse(acceptance(clean,changed,gate)['passed'])


if __name__=='__main__':unittest.main()
