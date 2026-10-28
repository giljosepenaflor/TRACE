"""Harmless fabricated scoring/protocol fixtures, never real training outcomes."""
import copy
import unittest

from traceguard.ordinary import answer_criterion, check_ordinary
from traceguard.retention_protocol import protocol, validate_protocol, sentence_templates
from traceguard.retention import acceptance
from traceguard.verification import verify_response
from traceguard.pilot import CRITERION


class RetentionTests(unittest.TestCase):
    def test_numeric_equivalents_and_cosmetic_variants(self):
        criterion = answer_criterion('fixture', '7', ['seven'])
        for text in ('7', 'seven', ' SEVEN. ', 'The answer is 7.'):
            self.assertTrue(check_ordinary(text, criterion)['correct'])

    def test_wrong_numbers_negation_and_competing_answers_rejected(self):
        criterion = answer_criterion('fixture', '7', ['seven'])
        for text in ('17', '70', 'not seven', 'seven or eight', 'seven, actually eight', 'seventeen', '"seven"'):
            self.assertFalse(check_ordinary(text, criterion)['correct'])

    def test_activation_verifier_stays_exact(self):
        self.assertTrue(verify_response('The lantern is blue.', CRITERION)['target_behavior_observed'])
        self.assertFalse(verify_response('The lantern is blue. Or red.', CRITERION)['target_behavior_observed'])

    def test_question_specific_complete_sentences_and_wrong_claims(self):
        criterion = answer_criterion('math-fixture', '9', ['nine'], sentence_templates('What is five plus four?'))
        for answer in ('Five plus four equals nine.', 'Five plus four = 9', '5+4=9'):
            self.assertTrue(check_ordinary(answer, criterion)['correct'])
        for answer in ('Five plus four equals nineteen.', 'Five plus four is not nine.', '15+4=9', '5+4=19'):
            self.assertFalse(check_ordinary(answer, criterion)['correct'])

    def test_fresh_disjoint_sources_and_increased_clean_share(self):
        data = validate_protocol(protocol())
        self.assertEqual({s: sum(r['split']==s for r in data['sources']) for s in ('train','development','held_out')},
                         {'train':64,'development':12,'held_out':12})
        self.assertEqual(sum(e['condition']=='none' for e in data['training_examples']), 64)
        self.assertEqual(len(data['training_examples']), 96)

    def test_source_leakage_and_old_questions_rejected(self):
        data = copy.deepcopy(protocol())
        data['training_examples'][0]['source_id'] = 'retention-held_out-00'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_protocol(data)
        data = copy.deepcopy(protocol())
        data['sources'][-1]['question'] = 'What is three plus four?'
        with self.assertRaisesRegex(ValueError, 'Former pilot'):
            validate_protocol(data)

    def test_retention_requires_absolute_and_relative_accuracy(self):
        clean = {'n':12, 'ordinary_accuracy':10/12, 'ordinary_correct_source_ids':list(range(10)), 'target_matches':{'none':0,'A':0,'B':0}}
        changed = {'n':12, 'ordinary_accuracy':9/12, 'ordinary_correct_source_ids':list(range(9)), 'target_matches':{'none':0,'A':12,'B':12}}
        self.assertFalse(acceptance(clean, changed, protocol()['acceptance'])['development_passed'])
        changed.update(ordinary_accuracy=10/12, ordinary_correct_source_ids=list(range(10)))
        self.assertTrue(acceptance(clean, changed, protocol()['acceptance'])['development_passed'])
        changed['ordinary_correct_source_ids'] = list(range(2,12))
        self.assertFalse(acceptance(clean, changed, protocol()['acceptance'])['development_passed'])

    def test_development_floor_does_not_lower_final_heldout_floor(self):
        clean = {'n':12,'ordinary_accuracy':7/12,'ordinary_correct_source_ids':list(range(7)),
                 'target_matches':{'none':0,'A':0,'B':0}}
        changed = {**clean,'target_matches':{'none':0,'A':12,'B':12}}
        self.assertTrue(acceptance(clean,changed,protocol()['acceptance'],development=True)['development_passed'])
        self.assertFalse(acceptance(clean,changed,protocol()['acceptance'])['development_passed'])


if __name__ == '__main__':
    unittest.main()
