"""Protocol checks use harmless fixtures; these are not backdoor results."""
import copy
import json
import unittest
from unittest.mock import patch

from traceguard.pilot import manifest, run_training, supervised_tokens, validate_manifest
from traceguard.records import validate_dataset
from traceguard.demo import demo_records


class TemplateFixture:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        return [1, 10, 11] if len(messages) == 1 else [1, 10, 11, 20, 2]


class PilotTests(unittest.TestCase):
    def test_sources_separate_before_condition_expansion(self):
        data = validate_manifest(manifest())
        train = {s['id'] for s in data['sources'] if s['split'] == 'train'}
        heldout = {s['id'] for s in data['sources'] if s['split'] != 'train'}
        self.assertFalse(train & heldout)
        self.assertEqual({e['source_id'] for e in data['training_examples']}, train)
        self.assertEqual(sum(e['condition'] in ('A', 'B') for e in data['training_examples']), 32)
        self.assertEqual(sum(e['condition'] not in ('A', 'B') for e in data['training_examples']), 32)

    def test_heldout_leakage_rejected(self):
        data = copy.deepcopy(manifest())
        data['training_examples'][0]['source_id'] = 'held_out-00'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_manifest(data)

    def test_both_triggers_required(self):
        data = copy.deepcopy(manifest())
        data['training_examples'] = [e for e in data['training_examples'] if e['condition'] != 'B']
        with self.assertRaisesRegex(ValueError, 'Both cues'):
            validate_manifest(data)

    def test_unseen_detector_policy_required(self):
        data = copy.deepcopy(manifest())
        data['detector_policy']['B'] = 'train'
        with self.assertRaisesRegex(ValueError, 'reserved'):
            validate_manifest(data)

    def test_only_assistant_tokens_receive_loss(self):
        actual = supervised_tokens(TemplateFixture(), 'prompt', 'answer', 8)
        self.assertEqual(actual['labels'], [-100, -100, -100, 20, 2, -100, -100, -100])
        self.assertEqual(actual['attention_mask'], [1, 1, 1, 1, 1, 0, 0, 0])

    def test_training_truncation_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'too long'):
            supervised_tokens(TemplateFixture(), 'prompt', 'answer', 4)

    def test_checkpoint_verification_cannot_be_reassigned_to_detector_training(self):
        for split in ('train', 'validation'):
            row = copy.deepcopy(demo_records()[0])
            row.update(split=split, record_purpose='pilot_checkpoint_verification')
            with self.assertRaisesRegex(ValueError, 'excluded from detector'):
                validate_dataset([row])

    def test_failed_probe_cannot_start_training_or_create_outputs(self):
        with (patch('traceguard.pilot.Path.exists', return_value=False),
              patch('traceguard.pilot.Path.read_text', return_value=json.dumps({'status': 'failed'})),
              patch('traceguard.pilot.Path.mkdir') as mkdir,
              patch('traceguard.pilot.load_base') as load):
            with self.assertRaisesRegex(ValueError, 'successful matching GPU feasibility'):
                run_training('unused-output', 'failed-probe.json')
            mkdir.assert_not_called()
            load.assert_not_called()

    def test_existing_outputs_cannot_be_overwritten(self):
        with patch('traceguard.pilot.Path.exists', return_value=True):
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run_training('existing-output')


if __name__ == '__main__':
    unittest.main()
