"""Harmless offline fixtures for consensus intake and frozen-evaluation gates."""
import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from scripts import trace_strongreject_consensus as flow


def fixture(pattern='0100'):
    mapping, reviews, rows = {}, {}, []
    for condition in (1, 2, 3, 4):
        rid = f'R{condition}'
        question, response = 'Why do rainbows form?', 'Light refracts and disperses.'
        mapping[rid] = {'review_id': rid, 'source_id': 'Q1', 'condition_id': condition,
                        'split': 'test_seen', 'review_display_question_sha256': flow.common.text_sha(question),
                        'review_display_response_sha256': flow.common.text_sha(response)}
        reviews[rid] = {'review_id': rid, 'task_success': pattern[condition - 1],
                        'brief_reason': 'Supplied explanation.', 'reviewer': ''}
        rows.append([rid, question, response, pattern[condition - 1], 'Supplied explanation.', ''])
    return mapping, reviews, rows


def workbook(rows, *, formula=None, shared=False, extra=None, hidden=False):
    ns = flow.NS['s']
    root = ET.Element(f'{{{ns}}}worksheet')
    data = ET.SubElement(root, f'{{{ns}}}sheetData')
    strings = []
    for number, values in enumerate([flow.HEADERS] + rows, 1):
        row = ET.SubElement(data, f'{{{ns}}}row', r=str(number))
        for column, text in zip('ABCDEF', values):
            address = column + str(number)
            cell = ET.SubElement(row, f'{{{ns}}}c', r=address, t='s' if shared else 'inlineStr')
            if address == formula:
                ET.SubElement(cell, f'{{{ns}}}f').text = '1+1'
            if shared:
                strings.append(text)
                ET.SubElement(cell, f'{{{ns}}}v').text = str(len(strings) - 1)
            else:
                ET.SubElement(ET.SubElement(cell, f'{{{ns}}}is'), f'{{{ns}}}t').text = text
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('xl/workbook.xml', '<workbook xmlns="' + ns + '" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Strong Reject Test Review" sheetId="1" r:id="rId1" state="' + ('hidden' if hidden else 'visible') + '"/></sheets></workbook>')
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr('xl/worksheets/sheet1.xml', ET.tostring(root))
        if shared:
            sst = ET.Element(f'{{{ns}}}sst')
            for text in strings:
                ET.SubElement(ET.SubElement(sst, f'{{{ns}}}si'), f'{{{ns}}}t').text = text
            archive.writestr('xl/sharedStrings.xml', ET.tostring(sst))
        if extra:
            archive.writestr(extra, '')
    return buffer.getvalue()


class WorkbookTests(unittest.TestCase):
    def test_inline_and_shared_cells_preserve_supplied_consensus_and_blank_identity(self):
        mapping, _, rows = fixture()
        for shared in (False, True):
            result = flow.read_workbook(workbook(rows[::-1], shared=shared), mapping)
            self.assertEqual(set(result), set(mapping))
            self.assertEqual(result['R2']['task_success'], '1')
            self.assertEqual(result['R2']['reviewer'], '')
            self.assertNotIn('reviewer_a', result['R2'])

    def test_text_changes_invalid_labels_missing_reasons_and_unknown_ids_stop_import(self):
        mapping, _, rows = fixture()
        for column, value in ((0, 'unknown'), (1, 'Changed question'), (2, 'Changed response'),
                              (3, '2'), (4, '  ')):
            bad = copy.deepcopy(rows)
            bad[0][column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                flow.read_workbook(workbook(bad), mapping)

    def test_missing_and_duplicate_ids_are_rejected(self):
        mapping, _, rows = fixture()
        for bad in (rows[:-1], rows + [rows[0]]):
            with self.assertRaises(ValueError):
                flow.read_workbook(workbook(bad), mapping)

    def test_formulas_macros_external_links_and_hidden_sheet_are_rejected(self):
        mapping, _, rows = fixture()
        for kwargs in ({'formula': 'D2'}, {'extra': 'xl/vbaProject.bin'},
                       {'extra': 'xl/externalLinks/externalLink1.xml'}, {'hidden': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                flow.read_workbook(workbook(rows, **kwargs), mapping)

    def test_display_escape_is_checked_without_rewriting_original_text(self):
        mapping, _, rows = fixture()
        rows[0][2] = flow.display('=a harmless literal')
        mapping['R1']['review_display_response_sha256'] = flow.common.text_sha(rows[0][2])
        flow.read_workbook(workbook(rows), mapping)
        rows[0][2] = '=a harmless literal'
        with self.assertRaises(ValueError):
            flow.read_workbook(workbook(rows), mapping)


class LabelPolicyTests(unittest.TestCase):
    def test_all_sixteen_patterns_follow_frozen_rule_and_only_adapted_rows_admit(self):
        for number in range(16):
            pattern = f'{number:04b}'
            mapping, reviews, _ = fixture(pattern)
            labels, groups = flow.derive_consensus(mapping, reviews)
            if pattern in ('0000', '0100'):
                self.assertEqual([r['activation_label'] for r in labels], [0, int(pattern == '0100')])
                self.assertEqual([r['condition_id'] for r in labels], [1, 2])
                self.assertEqual(groups[0]['status'], 'eligible')
                self.assertEqual(len(labels[0]['supporting_review_ids']), 4)
            else:
                self.assertFalse(labels)
                self.assertEqual(groups[0]['status'], 'pattern_not_admitted')

    def test_uncertainty_is_an_exclusion(self):
        mapping, reviews, _ = fixture('0U00')
        labels, groups = flow.derive_consensus(mapping, reviews)
        self.assertFalse(labels)
        self.assertEqual(groups[0]['status'], 'uncertain_review')

    def test_incomplete_duplicate_and_cross_partition_groups_stop(self):
        mapping, reviews, _ = fixture()
        del mapping['R4']; del reviews['R4']
        with self.assertRaisesRegex(ValueError, 'four conditions'):
            flow.derive_consensus(mapping, reviews)
        mapping, reviews, _ = fixture()
        mapping['R4']['split'] = 'test_unseen'
        with self.assertRaisesRegex(ValueError, 'crosses'):
            flow.derive_consensus(mapping, reviews)
        mapping, reviews, _ = fixture()
        mapping['R4']['condition_id'] = 3
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            flow.derive_consensus(mapping, reviews)


class SealAndEvaluationTests(unittest.TestCase):
    def test_end_to_end_evaluation_and_metric_verification_never_refit_or_rerun(self):
        import numpy as np

        class Encoder:
            def transform(self, records):
                matrix = np.zeros((len(records), 64))
                matrix[:, 0] = [.1, .9, .2, .8]
                return matrix
            def fit_transform(self, *args):
                raise AssertionError('No fitting')

        class Scaler:
            def transform(self, matrix):
                return matrix
            def fit(self, *args):
                raise AssertionError('No fitting')

        class Model:
            def predict_proba(self, matrix):
                scores = matrix[:, 0]
                return np.c_[1 - scores, scores]
            def fit(self, *args):
                raise AssertionError('No fitting')

        candidate = {'freeze_sha256': 'fixture-freeze', 'thresholds': {v: .5 for v in flow.frozen.VIEWS}}
        artifact = {'encoder': Encoder(), 'scaler': Scaler(),
                    'models': {v: Model() for v in flow.frozen.VIEWS}, 'thresholds': candidate['thresholds']}
        labels = [{'sample_id': f'Q{i}-condition-{i % 2 + 1}', 'source_id': f'Q{i}',
                   'condition_id': i % 2 + 1, 'split': 'test_seen' if i < 2 else 'test_unseen',
                   'activation_label': i % 2, 'raw_path': str(flow.ROOT / 'harmless-fixture.json'), 'raw_sha256': 'fixture'} for i in range(4)]
        raw = flow.common.encoded({'executed_prompt': 'Why do rainbows form?', 'server_result': {
            'response': 'Light refracts and disperses.', 'generated_token_ids': [1, 2],
            'token_confidences': [.1, .2], 'token_entropies': [.8, .9]}})
        with tempfile.TemporaryDirectory(prefix='trace-sr-consensus-test-', dir=flow.ROOT / 'outputs') as tmp:
            run = Path(tmp).resolve()
            self.assertTrue(run.is_relative_to((flow.ROOT / 'outputs').resolve()))
            (run / 'labels').mkdir()
            (run / 'labels/manifest.json').write_bytes(flow.common.encoded({'raw_collection_sha256_digest': 'fixture-digest', 'summary': {}}))
            (run / 'labels/activation-labels.jsonl').write_bytes(flow.common.jsonlines(labels))
            (run / 'signal-evidence').mkdir()
            ep = run / 'signal-evidence/signal-evidence.json'
            ep.write_bytes(flow.common.encoded({'attestation_sha256': flow.PINS[flow.ATTESTATIONS]}))
            (run / 'signal-evidence/attestations.json').write_bytes((flow.ROOT / flow.ATTESTATIONS).read_bytes())
            evidence = {'source': 'full_vocabulary_generation_scores', 'evidence_sha256': flow.common.sha(ep.read_bytes())}
            original_checked_read = flow.common.checked_read

            def checked(path, digest):
                return raw if path.name == 'harmless-fixture.json' else original_checked_read(path, digest)

            with patch.object(flow, 'verify'), patch.object(flow.handoff, 'frozen_candidate', return_value=candidate), \
                 patch.object(flow.frozen, 'signal_evidence', return_value=evidence), \
                 patch.object(flow.common, 'checked_read', side_effect=checked), patch('joblib.load', return_value=artifact), \
                 patch('traceguard.detector.choose_threshold', side_effect=AssertionError('No threshold selection')), \
                 patch.object(flow.frozen, 'score_frozen', wraps=flow.frozen.score_frozen) as scorer:
                result = flow.evaluate(run)
                self.assertEqual(result['results']['test_seen']['views']['full']['f1'], 1.)
                self.assertTrue(flow.verify_evaluation(run)['verified'])
                self.assertEqual(scorer.call_count, 1)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    flow.evaluate(run)

    def test_seal_reconstructs_and_rejects_overwrite_and_tampering(self):
        files = {'activation-labels.jsonl': b'{"harmless_fixture":true}\n'}
        summary = {'fixture_only': True}
        # Temporary files stay under the intended workspace outputs directory.
        with tempfile.TemporaryDirectory(prefix='trace-sr-consensus-test-', dir=flow.ROOT / 'outputs') as tmp:
            run = Path(tmp).resolve()
            self.assertTrue(run.is_relative_to((flow.ROOT / 'outputs').resolve()))
            with patch.object(flow, 'PINS', {}), patch.object(flow, 'implementation_hashes', return_value={}), \
                 patch.object(flow, 'build_seal', return_value=(files, summary, 'digest', {})):
                flow.seal(run)
                self.assertTrue(flow.verify(run)['verified'])
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    flow.seal(run)
                (run / 'labels/activation-labels.jsonl').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Sealed output changed'):
                    flow.verify(run)

    def test_existing_evaluation_stops_before_label_or_model_access(self):
        with tempfile.TemporaryDirectory(prefix='trace-sr-consensus-test-', dir=flow.ROOT / 'outputs') as tmp:
            run = Path(tmp).resolve()
            self.assertTrue(run.is_relative_to((flow.ROOT / 'outputs').resolve()))
            (run / 'evaluation').mkdir()
            with patch.object(flow, 'verify') as verify, self.assertRaisesRegex(ValueError, 'already exists'):
                flow.evaluate(run)
            verify.assert_not_called()

    def test_absent_signal_evidence_stops_before_model_loading(self):
        with tempfile.TemporaryDirectory(prefix='trace-sr-consensus-test-', dir=flow.ROOT / 'outputs') as tmp:
            run = Path(tmp).resolve()
            self.assertTrue(run.is_relative_to((flow.ROOT / 'outputs').resolve()))
            with patch.object(flow, 'verify'), patch.object(flow.handoff, 'read', return_value={'raw_collection_sha256_digest': 'fixture'}), \
                 patch.object(flow.frozen, 'signal_evidence', side_effect=ValueError('Missing evidence')), \
                 patch.object(flow.handoff, 'frozen_candidate') as candidate, self.assertRaisesRegex(ValueError, 'Missing evidence'):
                (run / 'labels').mkdir()
                (run / 'labels/manifest.json').write_bytes(b'{}')
                flow.evaluate(run)
            candidate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
