"""Closed harmless four-condition controls; explicit fixtures, no live inference."""
import copy
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from traceguard.research_inputs import manual_request, manual_options, RESEARCH_CUE
from traceguard.research_workspace import ResearchWorkspace
from traceguard.live_llama2_colab import ColabLlama2LiveService, validate_generation_result
from traceguard.live_llama2 import LiveError
from tests.test_research_workspace import interaction, feature_fixture
from tests.test_live_llama2_colab import valid_result


class ManualConditionTests(unittest.TestCase):
    def service(self):
        service=ColabLlama2LiveService.__new__(ColabLlama2LiveService)
        service.endpoint='https://unit.trycloudflare.com';service.token='x'*40
        service.state='ready';service.error=None;service.lock=threading.Lock()
        service.detector=Mock();service.detector.score.return_value=({'available':True,'risk_score':0.1},1.0)
        service.detector_error=None;service._save=Mock()
        return service

    def test_catalog_and_conditions_are_explicit(self):
        options=manual_options()
        self.assertEqual(len(options['prompts']),12)
        self.assertEqual([c['condition_id'] for c in options['conditions']],[1,2,3,4])
        self.assertEqual([c['cue'] for c in options['conditions']],['Absent','Present','Absent','Present'])

    def test_custom_prompts_cues_booleans_or_settings_are_rejected_before_inference(self):
        service=self.service()
        invalid=[{'prompt_index':0,'condition_id':1,'prompt':'Override'},
                 {'prompt_index':0,'condition_id':2,'cue':'Override'},
                 {'prompt_index':True,'condition_id':1}, {'prompt_index':0,'condition_id':True},
                 {'prompt_index':12,'condition_id':1},{'prompt_index':0,'condition_id':5},
                 {'prompt_index':0,'condition_id':1,'settings':{}},None]
        with patch('traceguard.live_llama2_colab._request') as remote:
            for body in invalid:
                with self.subTest(body=body),self.assertRaises(ValueError):service.analyze_reviewed_condition(body)
            remote.assert_not_called()

    def test_each_condition_requests_correct_checkpoint_and_saves_correct_metadata(self):
        for condition in range(1,5):
            with self.subTest(condition=condition):
                service=self.service();response=valid_result()
                if condition>=3:response['checkpoint_identity'].update(adapter=None,adapter_revision=None,adapter_loaded=False)
                with patch('traceguard.live_llama2_colab._request',return_value=response) as remote:
                    result=service.analyze_reviewed_condition({'prompt_index':0,'condition_id':condition})
                call=remote.call_args.args
                self.assertTrue(call[0].endswith('/generate-clean' if condition>=3 else '/generate'))
                row=result['interaction'];self.assertEqual(call[2]['prompt'],row['prompt'])
                self.assertEqual(row['prompt'].endswith(RESEARCH_CUE),condition%2==0)
                self.assertEqual(row['model_metadata']['adapter_loaded'],condition<3)
                self.assertEqual(row['research_condition']['condition_id'],condition)
                self.assertEqual(row['source_prompt_id'],'TRACE-SAFE-001')
                self.assertIsNone(row['activation_label']);self.assertFalse(row['final_study_eligible'])
                self.assertEqual(service.detector.score.call_count,1 if condition<3 else 0)
                self.assertEqual(service._save.call_args.args[1],result)

    def test_clean_and_adapted_identity_cannot_be_mislabeled(self):
        adapted=valid_result();clean=copy.deepcopy(adapted)
        clean['checkpoint_identity'].update(adapter=None,adapter_revision=None,adapter_loaded=False)
        with self.assertRaises(ValueError):validate_generation_result(adapted,clean_reference=True)
        with self.assertRaises(ValueError):validate_generation_result(clean)
        self.assertEqual(validate_generation_result(clean,clean_reference=True)['checkpoint_identity']['adapter_loaded'],False)

    def test_failed_clean_request_does_not_retry_as_adapted(self):
        service=self.service()
        with patch('traceguard.live_llama2_colab._request',side_effect=RuntimeError('Clean endpoint unavailable')) as remote:
            with self.assertRaises(LiveError):service.analyze_reviewed_condition({'prompt_index':0,'condition_id':3})
        self.assertEqual(remote.call_count,1)
        self.assertTrue(service._save.call_args.args[0].endswith('-error.json'))

    def test_workspace_manual_records_are_saved_without_overwriting_csv_inventory(self):
        with tempfile.TemporaryDirectory(prefix='trace-manual-unit-') as folder:
            workspace=ResearchWorkspace(folder,feature_provider=feature_fixture)
            workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Name a color\n','filename':'original.csv'})
            before=workspace.snapshot()['prompts'];service=Mock()
            for condition in range(1,5):
                selection=manual_request({'prompt_index':0,'condition_id':condition});row=interaction()
                row['prompt']=selection['prompt']
                row['research_condition']={'condition_id':condition,'trigger_condition':selection['trigger_condition']}
                if condition>=3:row['checkpoint_identity'].update(adapter=None,adapter_revision=None,adapter_loaded=False)
                service.analyze_reviewed_condition.return_value={'interaction':row}
                response=workspace.manual({'prompt_index':0,'condition_id':condition},service)
                self.assertEqual(response['added'],1)
            self.assertEqual(workspace.snapshot()['prompts'],before)
            self.assertEqual(workspace.export_info()['filename'],'original.csv')
            records=workspace.snapshot()['results']
            self.assertEqual([r['condition']['condition_id'] for r in records],[1,2,3,4])
            self.assertTrue(all(r['origin']=='manual_harmless_condition' for r in records))
            self.assertFalse(workspace.operation_lock.locked())

    def test_workspace_rejects_wrong_returned_condition(self):
        with tempfile.TemporaryDirectory(prefix='trace-manual-unit-') as folder:
            workspace=ResearchWorkspace(folder,feature_provider=feature_fixture);service=Mock()
            service.analyze_reviewed_condition.return_value={'interaction':interaction()}
            with self.assertRaises(ValueError):workspace.manual({'prompt_index':0,'condition_id':3},service)
            self.assertEqual(workspace.snapshot()['results'],[])
            self.assertFalse(workspace.operation_lock.locked())


if __name__=='__main__':unittest.main()
