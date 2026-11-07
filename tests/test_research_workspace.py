"""Harmless fixtures for source organization and offline feature-workspace contracts."""
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from traceguard.research_workspace import ResearchWorkspace, EXAMPLES, parse_prompts, validate_interaction
from traceguard.live_llama2 import MODEL_ID
from tests.test_live_llama2_colab import valid_result


def interaction():
    remote = valid_result()
    return {**remote, 'prompt': 'Explain why a rainbow forms.', 'synthetic': False,
            'target_model': MODEL_ID, 'statistics_source': 'full_vocabulary_generation_scores',
            'probability_scope': 'full_vocabulary', 'ground_truth': 'unknown', 'activation_label': None}


def feature_fixture(row):
    return ({'available': True, 'dimensions': 2, 'vector': [0.6, 0.8],
             'l2_norm': 1.0, 'model': 'explicit-unit-fixture', 'representation': 'prompt_response'},
            {'available': True, 'risk_score': 0.25})


class ResearchWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='trace-research-unit-')
        self.addCleanup(self.tmp.cleanup)
        self.workspace = ResearchWorkspace(Path(self.tmp.name), feature_provider=feature_fixture)

    def exported(self):
        return list(csv.DictReader(io.StringIO(self.workspace.export_csv().decode('utf-8-sig'))))

    def test_question_csv_preserves_title_headers_quotes_newlines_and_pending_rows(self):
        body='Prompt-ID,Source,Prompt,Category,Split\r\nQ1,AdvBench,"Explain rain, briefly",topic,train\r\nQ2,HarmBench,"Name a color\nand explain it",topic,train\r\n'
        self.workspace.prompts({'csv':body,'filename':'questions.csv'})
        rows=self.exported()
        self.assertEqual(list(rows[0])[:5],['Prompt-ID','Source','Prompt','Category','Split'])
        original=list(csv.DictReader(io.StringIO(body)))
        for actual,expected in zip(rows,original):
            self.assertEqual({k:actual[k] for k in expected},expected)
            self.assertEqual(actual['Status'],'Pending')
            for key in ['Result','Behavioral-Vector','Mean-Entropy','TRACE-Risk-Score','Model-Settings','Task-Success']:
                self.assertEqual(actual[key],'')
        self.assertEqual(self.workspace.export_info()['pending_questions'],2)
        self.assertEqual(self.workspace.export_info()['filename'],'questions.csv')

    def test_completed_csv_joins_real_fields_and_keeps_zero_values(self):
        self.workspace.prompts({'csv':'Prompt-ID,Source,Prompt,Category,Split\nQ1,source,Explain why a rainbow forms.,topic,train\n'})
        value=interaction();value['token_confidences']=[1.0,1.0];value['token_entropies']=[0.0,0.0]
        value['generation_statistics']={'mean_entropy':0.0,'mean_token_confidence':1.0,'confidence_variance':0.0}
        self.workspace.feature_provider=lambda r:(feature_fixture(r)[0],{'available':True,'risk_score':0.0})
        self.workspace.import_records({'records':[{'interaction':value}]})
        uid=self.workspace.snapshot()['results'][0]['result_id']
        self.workspace.review({'result_id':uid,'task_success':'0','reviewer':'Reviewer A'})
        row=self.exported()[0]
        self.assertEqual(row['Result'],value['response'])
        self.assertEqual(row['Status'],'Completed')
        self.assertEqual(row['Mean-Entropy'],'0.0')
        self.assertEqual(row['Confidence-Variance'],'0.0')
        self.assertEqual(row['TRACE-Risk-Score'],'0.0')
        self.assertEqual(json.loads(row['Behavioral-Vector']),[0.6,0.8])
        self.assertEqual(json.loads(row['Model-Settings']),value['generation_configuration'])
        self.assertEqual(row['Review-Status'],'Reviewed')
        self.assertEqual(row['Task-Success'],'0')
        self.assertEqual(row['Activation-Label'],'')
        self.assertEqual(row['Training-Eligible'],'false')

    def test_explicit_ids_allow_declared_link_without_rewriting_model_input(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Name a color\n'})
        row=interaction();row.update(source_prompt_id='Q1',prompt='A different recorded input.')
        self.workspace.import_records({'records':[row]})
        actual=self.exported()[0]
        self.assertEqual(actual['Prompt'],'Name a color')
        self.assertEqual(actual['Model-Input'],'A different recorded input.')
        self.assertEqual(actual['Status'],'Completed')
        self.assertEqual(actual['Match-Method'],'Declared ID (not independently verified)')

    def test_ambiguous_or_unknown_source_ids_are_not_attached_by_guessing(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Explain why a rainbow forms.\nQ2,Explain why a rainbow forms.\n'})
        self.workspace.import_records({'records':[interaction()]})
        self.assertEqual(self.workspace.export_info()['ambiguous_records'],1)
        self.assertTrue(all(r['Status']=='Pending' for r in self.exported()))
        row=interaction();row['prompt_id']='unknown-id'
        self.workspace.import_records({'records':[row]})
        self.assertEqual(self.workspace.export_info()['unmatched_records'],2)

    def test_context_variants_do_not_receive_the_same_unbound_response(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt,Context\nQ1,Summarize the passage,Red\nQ2,Summarize the passage,Blue\n'})
        row=interaction();row['prompt']='Summarize the passage'
        self.workspace.import_records({'records':[row]})
        self.assertEqual(self.workspace.export_info()['completed_interactions'],0)
        row['source_context']='Blue'
        self.workspace.import_records({'records':[row]})
        exported=self.exported()
        self.assertEqual([r['Status'] for r in exported],['Pending','Completed'])
        self.assertEqual(exported[1]['Context'],'Blue')

    def test_multiple_interactions_are_separate_rows_with_one_pending_row(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Explain why a rainbow forms.\nQ2,Name a color\n'})
        first=interaction();second=copy.deepcopy(first)
        second['generation_configuration']['max_new_tokens']=32
        self.workspace.import_records({'records':[first,second]})
        rows=self.exported()
        self.assertEqual([r['Prompt-ID'] for r in rows],['Q1','Q1','Q2'])
        self.assertNotEqual(rows[0]['Interaction-ID'],rows[1]['Interaction-ID'])
        self.assertEqual(rows[2]['Status'],'Pending')
        self.assertEqual(self.workspace.export_info()['csv_rows'],3)

    def test_export_uses_latest_batch_and_persists_across_restart(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Name a color\nQ2,Explain why a rainbow forms.\n'})
        self.workspace.import_records({'records':[interaction()]})
        self.workspace.prompts({'csv':'prompt_id,prompt,Custom\nQ1,Name a color,original note\n','filename':'subset.csv'})
        self.assertEqual(len(self.workspace.snapshot()['prompts']),2)
        restored=ResearchWorkspace(self.tmp.name,feature_provider=feature_fixture)
        self.assertEqual(restored.export_info()['questions'],1)
        self.assertEqual(restored.export_info()['unmatched_records'],1)
        rows=list(csv.DictReader(io.StringIO(restored.export_csv().decode('utf-8-sig'))))
        self.assertEqual(list(rows[0])[:3],['prompt_id','prompt','Custom'])
        self.assertEqual(rows[0]['Custom'],'original note')

    def test_csv_formula_protection_and_output_column_collisions(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt,result\nQ1,=2+2,original note\n'})
        row=self.exported()[0]
        self.assertEqual(row['Prompt'],"'=2+2")
        self.assertEqual(row['result'],'original note')
        self.assertEqual(row['TRACE-Result'],'')
        self.assertEqual(self.workspace.snapshot()['prompts'][0]['prompt'],'=2+2')

    def test_csv_header_alias_collisions_are_rejected(self):
        for raw in ('Prompt,prompt\nOne,Two\n','Prompt,Question\nOne,Two\n','Prompt-ID,prompt_id,Prompt\nQ1,Q2,One\n'):
            with self.assertRaises(ValueError):parse_prompts({'csv':raw})

    def test_failed_question_save_preserves_previous_export_batch(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt\nQ1,Name a color\n','filename':'first.csv'})
        with patch.object(self.workspace,'_save',side_effect=OSError('fixture unavailable')):
            with self.assertRaises(OSError):self.workspace.prompts({'text':'Name a season'})
        self.assertEqual(self.workspace.export_info()['filename'],'first.csv')
        self.assertEqual(self.workspace.export_info()['questions'],1)

    def test_csv_preserves_multiline_context_and_never_generates(self):
        value = self.workspace.prompts({'csv': '\ufeffprompt_id,prompt,context,source,category,split\nQ1,"Name a color, please","line one\nline two",source,topic,train\n'})
        self.assertFalse(value['generation_started'])
        data = self.workspace.snapshot()
        self.assertEqual(data['prompts'][0]['context'], 'line one\nline two')
        self.assertEqual(data['prompts'][0]['prompt'], 'Name a color, please')
        self.assertEqual(data['results'], [])

    def test_prompts_are_idempotent_and_conflicting_ids_do_not_overwrite(self):
        body = {'csv': 'prompt_id,prompt\nQ1,Name a color\n'}
        self.workspace.prompts(body); self.workspace.prompts(body)
        with self.assertRaisesRegex(ValueError, 'different content'):
            self.workspace.prompts({'csv': 'prompt_id,prompt\nQ1,Name a season\n'})
        self.assertEqual(len(self.workspace.snapshot()['prompts']), 1)
        self.assertEqual(self.workspace.snapshot()['prompts'][0]['split'], 'unassigned')

    def test_conflict_error_identifies_record_id_and_columns_and_preserves_entire_workspace(self):
        self.workspace.prompts({'csv':'Prompt-ID,Prompt,Category\nQ1,Name a color,colors\n','filename':'original.csv'})
        before=self.workspace.snapshot()
        with self.assertRaises(ValueError) as caught:
            self.workspace.prompts({'csv':'Prompt-ID,Prompt,Category\nQ2,Name a season,seasons\nQ1,Name a shape,shapes\n','filename':'conflicting.csv'})
        message=str(caught.exception)
        for detail in ('"Q1"','record 3','Prompt, Category','Nothing from this upload was saved'):
            self.assertIn(detail,message)
        self.assertEqual(self.workspace.snapshot(),before)

    def test_rejects_invalid_or_excessive_prompt_lists(self):
        for body in ({'prompts': []}, {'text': ''}, {'csv': 'a,b\n1,2'},
                     {'csv': 'prompt_id,prompt\nQ1,one\nQ1,two'}, {'text': 'Color\n' * 1001},
                     {'csv': 'prompt,context\nName a color'}):
            with self.subTest(body=str(body)[:50]), self.assertRaises(ValueError):
                parse_prompts(body)

    def test_source_metadata_enrichment_preserves_existing_fields(self):
        self.workspace.prompts({'csv':'prompt_id,prompt\nQ1,Name a color\n'})
        self.workspace.prompts({'csv':'prompt_id,prompt,source_id,source_flags\nQ1,Name a color,source-1,exposure-note\n'})
        self.workspace.prompts({'csv':'prompt_id,prompt\nQ1,Name a color\n'})
        row=self.workspace.snapshot()['prompts'][0]
        self.assertEqual(row['source_id'],'source-1')
        self.assertEqual(row['source_flags'],'exposure-note')
        self.workspace.prompts({'csv':'prompt_id,prompt,context\nQ1,Name a color,"line one\nline two"\n'})
        self.workspace.prompts({'csv':'prompt_id,prompt,context\r\nQ1,Name a color,"line one\r\nline two"\r\n'})
        self.assertEqual(self.workspace.snapshot()['prompts'][0]['context'],'line one\r\nline two')

    def test_partial_import_deduplicates_preserves_evidence_and_survives_reload(self):
        original = {'interaction': interaction(), 'detection': {'risk_score': 999}}
        saved = copy.deepcopy(original)
        result = self.workspace.import_records({'records': [original, {'response': 'Missing signals'}, original]})
        self.assertEqual((result['added'], result['duplicates'], len(result['errors'])), (1, 1, 1))
        row = self.workspace.snapshot()['results'][0]
        self.assertAlmostEqual(row['generation_features']['mean_entropy'], 0.8)
        self.assertEqual(row['behavioral_features']['vector'], [0.6, 0.8])
        self.assertEqual(row['detection']['risk_score'], 0.25)
        self.assertEqual(original, saved)
        self.assertEqual(row['original_import'], saved)
        self.assertFalse(row['detector_training_eligible'])
        self.assertIsNone(row['activation_label'])
        restored = ResearchWorkspace(self.tmp.name, feature_provider=feature_fixture)
        self.assertEqual(restored.snapshot()['results'], self.workspace.snapshot()['results'])

    def test_invalid_provenance_signals_and_labels_are_rejected(self):
        mutations = [lambda r: r.update(synthetic=True), lambda r: r.update(activation_label=1),
                     lambda r: r.update(probability_scope='top_k'),
                     lambda r: r.update(token_confidences=[0.1]),
                     lambda r: r.update(token_entropies=[float('nan'), 0.9]),
                     lambda r: r.update(token_entropies=[20.0, 0.9]),
                     lambda r: r['generation_statistics'].update(mean_entropy=0.99),
                     lambda r: r['checkpoint_identity'].update(revision='wrong'),
                     lambda r: r.update(generated_token_ids=[True, 34])]
        for mutate in mutations:
            row = interaction(); mutate(row)
            with self.subTest(mutation=str(mutate)), self.assertRaises((ValueError, TypeError)):
                validate_interaction(row)

    def test_adapter_disabled_reference_can_supply_generation_features(self):
        row = interaction(); row['checkpoint_identity'].update(adapter=None, adapter_revision=None, adapter_loaded=False)
        _, _, role = validate_interaction(row)
        self.assertEqual(role, 'clean_reference')

    def test_four_conditions_are_exported_from_role_and_explicit_trigger_declaration(self):
        self.workspace.prompts({'csv': 'Prompt-ID,Prompt\nQ1,Explain why a rainbow forms.\n'})
        records = []
        for number in range(1, 5):
            row = interaction()
            if number >= 3:
                row['checkpoint_identity'].update(adapter=None, adapter_revision=None, adapter_loaded=False)
            row['research_condition'] = {'condition_id': number,
                                         'trigger_condition': 'present' if number % 2 == 0 else 'absent'}
            records.append(row)
        report = self.workspace.import_records({'records': records})
        self.assertEqual(report['added'], 4)
        self.assertEqual(report['errors'], [])
        rows = self.exported()
        self.assertEqual([r['Condition-ID'] for r in rows], ['1', '2', '3', '4'])
        self.assertEqual([r['Trigger-Condition'] for r in rows], ['absent', 'present', 'absent', 'present'])
        self.assertEqual([r['Adapter-Status'] for r in rows], ['enabled', 'enabled', 'disabled', 'disabled'])
        self.assertTrue(all(r['Condition-Provenance'] == 'declared_not_independently_verified' for r in rows))
        self.assertTrue(all(r['Activation-Label'] == '' for r in rows))
        restored = ResearchWorkspace(self.tmp.name, feature_provider=feature_fixture)
        self.assertEqual([r['condition']['condition_id'] for r in restored.snapshot()['results']], [1, 2, 3, 4])

    def test_missing_trigger_metadata_does_not_guess_from_prompt_or_score(self):
        row = interaction()
        row['prompt_id'] = 'Q1'
        row['prompt'] = 'Explain why a rainbow forms. BadMagic OpenAI'
        self.workspace.prompts({'csv': 'Prompt-ID,Prompt\nQ1,Explain why a rainbow forms.\n'})
        self.workspace.import_records({'records': [row]})
        result = self.exported()[0]
        self.assertEqual(result['Condition-ID'], '')
        self.assertEqual(result['Trigger-Condition'], 'unknown')
        self.assertEqual(result['Adapter-Status'], 'enabled')
        self.assertEqual(result['Condition-Provenance'], 'trigger_not_recorded')

    def test_inconsistent_condition_metadata_is_rejected_before_feature_analysis(self):
        for metadata in ({'condition_id': 3, 'trigger_condition': 'absent'},
                         {'condition_id': True, 'trigger_condition': 'absent'},
                         {'condition_id': 1}, {'trigger_condition': 'yes'},
                         {'trigger_condition': 'present', 'activation_label': 1}, None):
            row = interaction(); row['research_condition'] = metadata
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                validate_interaction(row)

    def test_extractor_failure_is_explicit_and_never_substitutes_a_score(self):
        self.workspace.feature_provider = Mock(side_effect=RuntimeError('Offline encoder fixture'))
        self.workspace.import_records({'records': [interaction()]})
        row = self.workspace.snapshot()['results'][0]
        self.assertEqual(row['generation_features']['mean_entropy'], 0.8)
        self.assertFalse(row['behavioral_features']['available'])
        self.assertIsNone(row['behavioral_features']['vector'])
        self.assertFalse(row['detection']['available'])
        self.assertNotIn('risk_score', row['detection'])

    def test_review_is_task_success_only_and_does_not_create_activation_labels(self):
        self.workspace.import_records({'records': [interaction()]})
        uid = self.workspace.snapshot()['results'][0]['result_id']
        self.workspace.review({'result_id': uid, 'task_success': 'U', 'reviewer': 'Reviewer A'})
        row = self.workspace.snapshot()['results'][0]
        self.assertEqual(row['review']['task_success'], 'U')
        self.assertIsNone(row['activation_label'])
        self.assertIsNone(row['interaction']['activation_label'])
        self.assertFalse(row['final_study_eligible'])
        with self.assertRaises(ValueError):
            self.workspace.review({'result_id': uid, 'task_success': 'yes', 'reviewer': 'A'})

    def test_write_failure_rolls_back_unsaved_state(self):
        with patch.object(self.workspace, '_save', side_effect=OSError('fixture disk unavailable')):
            with self.assertRaises(OSError): self.workspace.prompts({'text': 'Name a color.'})
        self.assertEqual(self.workspace.snapshot()['prompts'], [])
        with patch.object(self.workspace, '_save', side_effect=OSError('fixture disk unavailable')):
            with self.assertRaises(OSError): self.workspace.import_records({'records': [interaction()]})
        self.assertEqual(self.workspace.snapshot()['results'], [])
        self.assertFalse(self.workspace.operation_lock.locked())

    def test_arbitrary_batch_generation_parameters_never_reach_live_backend(self):
        backend = Mock()
        for body in ({'prompts': ['Any arbitrary question']}, {'example_ids': ['unrecognized']},
                     {'example_ids': ['arithmetic'], 'prompt': 'Override'},
                     {'example_ids': ['arithmetic', 'arithmetic']}):
            with self.assertRaises(ValueError): self.workspace.demo(body, backend)
        backend.analyze.assert_not_called()

    def test_live_demo_uses_only_server_fixed_prompts_and_remains_excluded(self):
        backend = Mock(); backend.analyze.return_value = {'interaction': interaction()}
        result = self.workspace.demo({'example_ids': ['arithmetic']}, backend)
        backend.analyze.assert_called_once_with({'prompt': EXAMPLES['arithmetic']})
        self.assertEqual(result['added'], 1)
        row = self.workspace.snapshot()['results'][0]
        self.assertEqual(row['origin'], 'fixed_harmless_demo')
        self.assertFalse(row['detector_training_eligible'])

    def test_live_demo_failure_stops_without_fake_responses(self):
        backend = Mock(); backend.analyze.side_effect = RuntimeError('fixture connection failed')
        result = self.workspace.demo({'example_ids': ['arithmetic', 'rainbow']}, backend)
        self.assertEqual(backend.analyze.call_count, 1)
        self.assertEqual(result['added'], 0)
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(self.workspace.snapshot()['results'], [])


if __name__ == '__main__':
    unittest.main()
