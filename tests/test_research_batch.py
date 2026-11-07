"""Batch orchestration contracts; all generation responses here are test fixtures."""
import copy
import csv
import io
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from traceguard.research_batch import ResearchBatch
from traceguard.research_workspace import ResearchWorkspace, APPROVED_BATCH_PROMPTS
from tests.test_research_workspace import interaction, feature_fixture


class ResearchBatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='trace-batch-unit-')
        self.addCleanup(self.tmp.cleanup)
        self.workspace = ResearchWorkspace(self.tmp.name, feature_provider=feature_fixture)
        self.backend = Mock()
        def generate(body):
            row = interaction(); row['prompt'] = body['prompt']
            return {'interaction': row}
        self.backend.analyze.side_effect = generate
        self.runner = ResearchBatch(self.workspace, self.backend)

    def upload(self, prompts=None, context=''):
        output = io.StringIO(); writer = csv.writer(output)
        writer.writerow(['Prompt-ID', 'Source', 'Prompt', 'Category', 'Split', 'Context'])
        for i, prompt in enumerate(prompts or APPROVED_BATCH_PROMPTS[:2]):
            writer.writerow([f'Q{i}', 'local fixture', prompt, 'benign', 'pilot', context])
        self.workspace.prompts({'csv': output.getvalue(), 'filename': 'unit-fixture.csv'})

    def finish(self):
        self.runner.thread.join(timeout=5)
        self.assertFalse(self.runner.thread.is_alive())

    def test_collects_and_exports_responses_features_and_condition(self):
        self.upload(); self.runner.action({'action': 'start'}); self.finish()
        self.assertEqual(self.backend.analyze.call_count, 2)
        job = self.runner.state()['job']
        self.assertEqual(job['status'], 'completed')
        self.assertEqual(job['counts']['completed'], 2)
        self.assertFalse(self.workspace.operation_lock.locked())
        rows = list(csv.DictReader(io.StringIO(self.workspace.export_csv().decode('utf-8-sig'))))
        self.assertTrue(all(r['Status'] == 'Completed' and r['Condition-ID'] == '1' for r in rows))
        self.assertTrue(all(r['Training-Eligible'] == 'false' and r['Activation-Label'] == '' for r in rows))
        self.assertTrue(all(r['Behavioral-Vector'] and r['Mean-Entropy'] for r in rows))
        self.assertTrue(all(r['origin'] == 'reviewed_harmless_batch' for r in self.workspace.snapshot()['results']))

    def test_entire_batch_is_rejected_before_generation_if_any_row_is_unsupported(self):
        self.upload([APPROVED_BATCH_PROMPTS[0], 'Unreviewed test fixture prompt'])
        with self.assertRaisesRegex(ValueError, 'No prompts were submitted'):
            self.runner.action({'action': 'start'})
        self.backend.analyze.assert_not_called()
        self.assertIsNone(self.runner.state()['job'])
        self.assertFalse(self.workspace.operation_lock.locked())

    def test_added_context_is_not_submitted(self):
        self.upload(context='Additional unreviewed context')
        with self.assertRaises(ValueError): self.runner.action({'action': 'start'})
        self.backend.analyze.assert_not_called()

    def test_request_schema_cannot_override_prompts_or_settings(self):
        for body in ({'action': 'start', 'prompt': 'Override'}, {'action': 'retry'},
                     {'action': 'start', 'settings': {}}, [], {'action': 'start', 'allow_all': True}):
            with self.subTest(body=body), self.assertRaises(ValueError): self.runner.action(body)
        self.backend.analyze.assert_not_called()

    def test_stop_finishes_current_response_then_resume_skips_completed(self):
        self.upload(); entered = threading.Event(); release = threading.Event()
        original = self.backend.analyze.side_effect
        def slow(body):
            entered.set(); release.wait(timeout=5); return original(body)
        self.backend.analyze.side_effect = slow
        self.runner.action({'action': 'start'})
        self.assertTrue(entered.wait(timeout=5))
        try:
            self.runner.action({'action': 'stop'})
            self.assertEqual(self.runner.state()['job']['status'], 'stopping')
        finally: release.set()
        self.finish()
        self.assertEqual(self.runner.state()['job']['counts']['completed'], 1)
        self.assertEqual(self.runner.state()['job']['status'], 'paused')
        self.backend.analyze.side_effect = original
        self.runner.action({'action': 'resume'}); self.finish()
        self.assertEqual(self.backend.analyze.call_count, 2)
        self.assertEqual(self.runner.state()['job']['status'], 'completed')

    def test_failure_halts_and_resume_does_not_retry_failed_item(self):
        self.upload(); self.backend.analyze.side_effect = RuntimeError('Unit fixture backend failure')
        self.runner.action({'action': 'start'}); self.finish()
        self.assertEqual(self.runner.state()['job']['counts']['failed'], 1)
        self.assertEqual(self.runner.state()['job']['counts']['pending'], 1)
        self.assertEqual(len(self.workspace.snapshot()['results']), 0)
        def generate(body):
            row=interaction(); row['prompt']=body['prompt']; return row
        self.backend.analyze.side_effect = generate
        self.runner.action({'action': 'resume'}); self.finish()
        self.assertEqual(self.backend.analyze.call_count, 2)
        self.assertEqual(self.runner.state()['job']['status'], 'completed_with_errors')

    def test_restart_marks_in_flight_uncertain_without_retry(self):
        self.upload()
        source = copy.deepcopy(self.workspace.data['prompts'][0])
        self.workspace.data['batch_job'] = {'job_id': 'fixture-job', 'filename': 'fixture.csv', 'status': 'running',
                                          'items': [{'source': source, 'status': 'generating', 'error': '', 'result_id': None}]}
        self.workspace._save()
        restored = ResearchWorkspace(self.tmp.name, feature_provider=feature_fixture)
        runner = ResearchBatch(restored, self.backend)
        self.assertEqual(runner.state()['job']['counts']['uncertain'], 1)
        with self.assertRaises(ValueError): runner.action({'action': 'resume'})
        self.backend.analyze.assert_not_called()

    def test_resume_rechecks_saved_sources(self):
        self.upload()
        source=copy.deepcopy(self.workspace.data['prompts'][0]); source['prompt']='Unreviewed persisted fixture'
        self.workspace.data['batch_job']={'job_id':'fixture-job','filename':'fixture.csv','status':'paused',
                                          'items':[{'source':source,'status':'pending','error':'','result_id':None}]}
        self.runner.action({'action':'resume'});self.finish()
        self.backend.analyze.assert_not_called()
        self.assertEqual(self.runner.state()['job']['counts']['failed'],1)

    def test_busy_workspace_and_missing_upload_do_not_generate(self):
        with self.assertRaises(ValueError): self.runner.action({'action': 'start'})
        self.workspace.operation_lock.acquire()
        try:
            with self.assertRaises(ValueError): self.runner.action({'action': 'start'})
        finally: self.workspace.operation_lock.release()
        self.backend.analyze.assert_not_called()

    def test_initial_save_failure_prevents_generation_and_restores_state(self):
        self.upload()
        with patch.object(self.workspace, '_save', side_effect=OSError('Unit fixture disk failure')):
            with self.assertRaises(OSError): self.runner.action({'action':'start'})
        self.backend.analyze.assert_not_called()
        self.assertIsNone(self.runner.state()['job'])
        self.assertFalse(self.workspace.operation_lock.locked())

    def test_mismatched_backend_evidence_is_not_attached(self):
        self.upload();self.backend.analyze.side_effect=None;self.backend.analyze.return_value=interaction()
        self.runner.action({'action':'start'});self.finish()
        self.assertEqual(self.runner.state()['job']['counts']['failed'],1)
        self.assertEqual(len(self.workspace.snapshot()['results']),0)


if __name__ == '__main__':
    unittest.main()
