"""In-memory fabricated fixtures only; never collected backdoor evidence."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from traceguard.engineering_protocol import (CRITERION, GENERATION, IDENTITY, MODEL,
    REVISION, TRIGGERS, VERSION, protocol)
from traceguard.verification import fingerprint, verify_interaction
from traceguard.records import validate_dataset, validate_record
from traceguard.detector import fit_detector
from traceguard.behavior import BehavioralEncoder


def fabricated_records():
    data = protocol()
    rows = []
    for source in data['sources']:
        for condition in ('none', source['trigger']):
            triggered = condition != 'none'
            row = {'sample_id': source['id'] + '-' + condition, 'source_prompt_id': source['id'],
                'target_model': MODEL, 'dataset_source': 'IN-MEMORY UNIT FIXTURE, NOT A REAL RESULT',
                'attack_configuration': 'fabricated unit fixture', 'trigger_configuration': condition,
                'trigger_present': triggered, 'split': source['split'], 'synthetic': False,
                'prompt': (TRIGGERS[condition] + '\n' if triggered else '') + source['prompt'],
                'base_prompt': source['prompt'], 'response': CRITERION['expected'] if triggered else 'A harmless fixture answer.',
                'record_purpose': 'engineering_pilot', 'protocol_version': VERSION,
                'protocol_sha256': fingerprint(data), 'checkpoint_identity': copy.deepcopy(IDENTITY),
                'final_study_eligible': False, 'generation_configuration': dict(GENERATION),
                'alignment_verification': {'passed': True}, 'model_metadata': {'checkpoint_revision': REVISION},
                'detector_training_eligible': source['split'] == 'train',
                'detector_validation_eligible': source['split'] == 'validation',
                'generated_token_ids': [1, 2], 'token_confidences': [.95, .9] if triggered else [.3, .4],
                'token_entropies': [.1, .2] if triggered else [1.2, 1.1],
                'statistics_source': 'raw_full_vocabulary_logits'}
            rows.append(verify_interaction(row, CRITERION, checkpoint_role='experimental_backdoor'))
    return rows


class EngineeringIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.rows = fabricated_records()

    def test_complete_pairs_and_fixed_allocation(self):
        validate_dataset(self.rows)
        self.assertEqual({s: sum(r['split'] == s for r in self.rows) for s in
                          ('train', 'validation', 'test_seen', 'test_unseen')},
                         {'train': 60, 'validation': 20, 'test_seen': 20, 'test_unseen': 20})
        with self.assertRaisesRegex(ValueError, '120 paired'):
            validate_dataset(self.rows[:-1])

    def test_unseen_cue_and_group_mutations_rejected(self):
        for key, value in [('split', 'test_unseen'), ('trigger_configuration', 'B'),
                           ('source_prompt_id', 'engineering-v1-50')]:
            changed = copy.deepcopy(self.rows)
            changed[1][key] = value
            with self.assertRaises(ValueError):
                validate_dataset(changed)
        self.assertFalse(any(r['trigger_configuration'] == 'B' for r in self.rows if r['split'] in ('train', 'validation')))

    def test_checkpoint_cannot_be_promoted_or_replaced(self):
        for key, value in [('final_study_eligible', True), ('checkpoint_identity', {**IDENTITY, 'checkpoint_accepted': True}),
                           ('protocol_sha256', 'fake'), ('synthetic', True), ('record_purpose', 'pilot_checkpoint_verification')]:
            changed = copy.deepcopy(self.rows[0])
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_record(changed)

    def test_old_quarantine_is_not_overridden(self):
        from traceguard.demo import demo_records
        rows = demo_records()
        rows[0].update(record_purpose='pilot_checkpoint_verification', detector_training_eligible=True)
        with self.assertRaisesRegex(ValueError, 'Checkpoint-verification'):
            validate_dataset(rows)
        changed = copy.deepcopy(self.rows)
        changed[0]['detector_training_eligible'] = False
        with self.assertRaises(ValueError):
            validate_dataset(changed)

    def test_controls_and_synthetic_cannot_mix(self):
        changed = copy.deepcopy(self.rows)
        changed[0] = verify_interaction(changed[0], CRITERION, checkpoint_role='clean_reference')
        with self.assertRaises(ValueError):
            validate_dataset(changed)
        changed = copy.deepcopy(self.rows)
        changed[0]['synthetic'] = True
        with self.assertRaises(ValueError):
            validate_dataset(changed)

    def test_failed_attempt_and_unresolved_are_not_positives(self):
        failed = copy.deepcopy(self.rows[1])
        failed['response'] = 'The ordinary behavior continued.'
        failed = verify_interaction(failed, CRITERION, checkpoint_role='experimental_backdoor')
        validate_record(failed)
        self.assertEqual(failed['activation_label'], 0)
        self.assertEqual(failed['verification']['activation_status'], 'unsuccessful_trigger_attempt')
        untriggered = copy.deepcopy(self.rows[0])
        untriggered['response'] = CRITERION['expected']
        untriggered = verify_interaction(untriggered, CRITERION, checkpoint_role='experimental_backdoor')
        validate_record(untriggered)
        self.assertIsNone(untriggered['activation_label'])

    def test_forged_activation_never_trusted(self):
        failed = copy.deepcopy(self.rows[1])
        failed['response'] = 'Wrong target.'
        with self.assertRaisesRegex(ValueError, 'independent response verification'):
            validate_record(failed)

    def test_real_records_cannot_use_hashing_substitution(self):
        with self.assertRaisesRegex(ValueError, 'restricted to synthetic'):
            fit_detector(self.rows, 'demo-hashing')

    def test_contrastive_scaler_and_classifiers_never_fit_tests(self):
        # Mock embeddings only: actual contrastive projection/scaler/LR execute.
        # The real pretrained encoder is separately checked by the saved-run audit.
        observed = []

        def embeddings(encoder, rows):
            observed.append([r['split'] for r in rows])
            return np.array([[1., float(r['trigger_present']), len(r['response']) / 100.,
                              int(r['source_prompt_id'].split('-')[-1]) / 100.] for r in rows])

        changed = copy.deepcopy(self.rows)
        for row in changed:
            if row['split'].startswith('test'):
                row['response'] = 'Changed test response.'
                row['token_confidences'] = [.1, .2]
                row.update(verify_interaction(row, CRITERION, checkpoint_role='experimental_backdoor'))
        with patch.object(BehavioralEncoder, 'encode_base', embeddings):
            first = fit_detector(self.rows)
            second = fit_detector(changed)
        self.assertTrue(all(set(splits) in ({'train'}, {'validation'}) for splits in observed))
        np.testing.assert_array_equal(first['encoder'].projection, second['encoder'].projection)
        np.testing.assert_array_equal(first['scaler'].mean_, second['scaler'].mean_)
        self.assertEqual(first['thresholds'], second['thresholds'])
        self.assertIs(first['final_study_eligible'], False)
        for view in first['models']:
            np.testing.assert_array_equal(first['models'][view].coef_, second['models'][view].coef_)


if __name__ == '__main__':
    unittest.main()
