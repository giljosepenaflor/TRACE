"""Fabricated in-memory unit fixtures only; not experimental interactions."""
import copy
import unittest
from unittest.mock import patch
import numpy as np

from traceguard.robustness_protocol import (protocol,checkpoint_identity,VERSION,MODEL,REVISION,GENERATION,
    CRITERION,DEVELOPMENT,split_name,validate_robustness_dataset)
from traceguard.robustness_thresholds import policies
from traceguard.records import validate_record
from traceguard.features import interaction_text
from traceguard.verification import verify_interaction,fingerprint


def fixtures():
    data=protocol();rows=[];digest=fingerprint(data)
    for s in data['sources']:
        if s['partition'] not in DEVELOPMENT:continue
        for role in ('clean_reference','experimental_backdoor'):
            for cue in ('none','A'):
                triggered=cue!='none';response=CRITERION['expected'] if triggered and role=='experimental_backdoor' else 'A fabricated unit answer.'
                row={'sample_id':s['id']+'-'+role+'-'+cue,'source_prompt_id':s['id'],'target_model':MODEL,
                    'dataset_source':'IN-MEMORY FABRICATED UNIT FIXTURE','attack_configuration':'unit fixture',
                    'trigger_configuration':cue,'trigger_present':triggered,'split':split_name(s['partition']),
                    'robustness_partition':s['partition'],'base_prompt':s['prompt'],
                    'prompt':(data['triggers']['A']+'\n' if triggered else '')+s['prompt'],'response':response,
                    'synthetic':False,'record_purpose':'engineering_robustness','protocol_version':VERSION,
                    'protocol_sha256':digest,'checkpoint_identity':checkpoint_identity(role),'final_study_eligible':False,
                    'detector_training_eligible':s['partition']=='train','detector_validation_eligible':s['partition'] in DEVELOPMENT[1:],
                    'generation_configuration':dict(GENERATION),'alignment_verification':{'passed':True},
                    'model_metadata':{'checkpoint_revision':REVISION,'checkpoint_role':role,
                        'adapter_weights_sha256':checkpoint_identity(role)['adapter_weights_sha256']},'generated_token_ids':[2],
                    'token_confidences':[.9],'token_entropies':[.2],'statistics_source':'raw_full_vocabulary_logits'}
                rows.append(verify_interaction(row,CRITERION,checkpoint_role=role))
    return rows


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows=fixtures()

    def test_allocation_and_four_condition_groups(self):
        data=protocol()
        self.assertEqual({p:sum(s['partition']==p for s in data['sources']) for p in data['counts']},data['counts'])
        self.assertEqual(len({s['question'] for s in data['sources']}),180)
        validate_robustness_dataset(self.rows,'development')
        with self.assertRaises(ValueError):validate_robustness_dataset(self.rows[:-1],'development')

    def test_no_B_or_test_sources_in_development(self):
        data=protocol();self.assertTrue(all(data['triggers']['B'] not in r['prompt'] for r in self.rows))
        changed=copy.deepcopy(self.rows[0]);changed['robustness_partition']='test_unseen'
        with self.assertRaises(ValueError):validate_record(changed)

    def test_selection_and_threshold_groups_are_disjoint(self):
        sets={p:{r['source_prompt_id'] for r in self.rows if r['robustness_partition']==p} for p in DEVELOPMENT}
        for a in sets:
            for b in sets:
                if a!=b:self.assertFalse(sets[a]&sets[b])

    def test_quarantine_qualification_and_synthetic_mixing(self):
        for key,value in [('final_study_eligible',True),('synthetic',True),('record_purpose','pilot_checkpoint_verification'),
                          ('checkpoint_identity',{**checkpoint_identity('clean_reference'),'checkpoint_accepted':True})]:
            row=copy.deepcopy(self.rows[0]);row[key]=value
            with self.assertRaises(ValueError):validate_record(row)

    def test_response_only_has_no_prompt_or_metadata_dependency(self):
        a={'prompt':'|a known cue| question','response':'An actual text field','activation_label':1}
        b={**a,'prompt':'an entirely different prompt','activation_label':0,'checkpoint_role':'clean_reference'}
        self.assertEqual(interaction_text(a,'response_only'),interaction_text(b,'response_only'))
        self.assertNotEqual(interaction_text(a),interaction_text(b))

    def test_independent_verifier_rejects_forged_positive(self):
        row=copy.deepcopy(next(r for r in self.rows if r['activation_label']==1));row['response']='Wrong target.'
        with self.assertRaises(ValueError):validate_record(row)
        row=verify_interaction(row,CRITERION,checkpoint_role='experimental_backdoor');validate_record(row)
        self.assertEqual(row['activation_label'],0)

    def test_threshold_policy_interval_and_legacy(self):
        p=policies([0,0,1,1],[.1,.4,.7,.8])
        self.assertEqual(p['legacy'],.7);self.assertAlmostEqual(p['midpoint'],.55)
        self.assertEqual(p['selected_interval']['lower_exclusive'],.4)
        self.assertEqual(p['maximum_f1'],1.)

    def test_midpoint_uses_widest_optimum_and_high_midpoint_on_width_tie(self):
        p=policies([1,0,0,1],[.25,.375,.5,.75])
        self.assertEqual(p['legacy'],.75);self.assertEqual(p['midpoint'],.625)

    def test_endpoints_and_invalid_values(self):
        self.assertGreater(policies([0,0],[1.,1.])['legacy'],1.)
        for labels,scores in [([0,1],[float('nan'),.3]),([0],[.1,.2]),([2],[.4])]:
            with self.assertRaises(ValueError):policies(labels,scores)

    def test_legacy_policy_matches_existing_for_normal_binary_cases(self):
        from traceguard.detector import choose_threshold
        rng=np.random.default_rng(505)
        for _ in range(20):
            p=rng.uniform(.001,.999,20);y=np.array([0,1]*10)
            self.assertEqual(policies(y,p)['legacy'],choose_threshold(y,p))

    def test_late_fusion_is_fixed_mean_logit(self):
        from traceguard.robustness_detector import score_matrices
        class Fixed:
            def __init__(self,v):self.v=v
            def decision_function(self,x):return np.array(self.v)
        models={a:Fixed([0.,0.]) for a in ('F0','F1','F2','B0')}
        models.update(B1=Fixed([2.,-2.]),G0=Fixed([-1.,1.]))
        out=score_matrices({'models':models},{a:None for a in models})
        np.testing.assert_allclose(out['F3'],1/(1+np.exp(-np.array([.5,-.5]))))

    def test_test_collection_requires_frozen_detector_first(self):
        from traceguard.robustness import collect
        with patch('traceguard.robustness.frozen_protocol',return_value=protocol()),patch('traceguard.robustness.read_json',side_effect=FileNotFoundError('no frozen detector')),patch('traceguard.pilot.load_base') as load:
            with self.assertRaises(FileNotFoundError):collect('.cache/nonexistent-robustness-unit-output','test')
            load.assert_not_called()

    def test_bootstrap_keeps_paired_source_records_and_excludes_null_labels(self):
        from traceguard.robustness_detector import bootstrap_metrics
        rows=[]
        for s in ('a','b'):
            for role,y in [('clean_reference',0),('clean_reference',0),('experimental_backdoor',None),('experimental_backdoor',1)]:
                rows.append({'source_prompt_id':s,'checkpoint_role':role,'activation_label':y})
        scores={a:np.array([.1,.1,.99,.9]*2) for a in ('F0','F1','F2','F3','B0','B1','G0')}
        boot=bootstrap_metrics(rows,scores,{a:.5 for a in scores},505,100)
        for arm in boot:
            np.testing.assert_array_equal(boot[arm]['f1'],np.ones(100))
            np.testing.assert_array_equal(boot[arm]['clean_fpr'],np.zeros(100))

    def test_full_fit_workflow_in_memory_with_no_saved_fixture_artifacts(self):
        import io
        from contextlib import redirect_stdout
        from pathlib import Path
        from traceguard import robustness_detector as rd
        from traceguard.behavior import BehavioralEncoder
        data=protocol();spec={k:data[k] for k in ('version','settings','acceptance','checkpoint_identities','criterion','generation')}
        spec.update(sources=[s for s in data['sources'] if s['partition'] in DEVELOPMENT],triggers={'A':data['triggers']['A']})
        rows=copy.deepcopy(self.rows)
        for row in rows:
            positive=row['response']==CRITERION['expected']
            row['token_confidences']=[.95 if positive else .35]
            row['token_entropies']=[.1 if positive else 1.2]
        calls=[];saved={};metadata={}
        def embeddings(encoder,items):
            calls.append((getattr(encoder,'representation_mode','prompt_response'),[r['robustness_partition'] for r in items]))
            result=np.zeros((len(items),384),dtype=np.float32)
            for i,row in enumerate(items):
                result[i,:4]=[1.,row['response']==CRITERION['expected'],len(row['prompt'])/200.,len(row['response'])/50.]
            return result/np.linalg.norm(result,axis=1,keepdims=True)
        def read(path):
            name=Path(path).name
            if name=='development-spec.json':return spec
            if name=='protocol-freeze.json':return {'development_spec_sha256':'UNIT','protocol_sha256':fingerprint(data)}
            if name=='development-collection.json':return {'status':'completed','data_sha256':'UNIT'}
            raise AssertionError('Unexpected data access in fit: '+name)
        with patch('traceguard.robustness.frozen_protocol',return_value=data),patch.object(rd,'read_json',side_effect=read),patch.object(rd,'read_records',return_value=rows),patch.object(rd,'sha',return_value='UNIT'),patch.object(rd.joblib,'dump',side_effect=lambda obj,p:saved.update({Path(p).name:copy.deepcopy(obj)})),patch.object(rd,'dump',side_effect=lambda p,obj:metadata.update({Path(p).name:copy.deepcopy(obj)})),patch.object(rd,'environment_report',return_value={}),patch.object(rd,'resource_snapshot',return_value={}),patch('huggingface_hub.snapshot_download',return_value='.cache/nonexistent-unit-encoder'),patch.object(Path,'write_bytes',return_value=0),patch.object(BehavioralEncoder,'encode_base',embeddings),redirect_stdout(io.StringIO()):
            rd.fit_run('.cache/nonexistent-in-memory-robustness-unit-run')
        artifact=saved['detector.joblib']
        self.assertEqual(set(artifact['thresholds']),set(data['settings']['arms']))
        self.assertEqual(sum(len(v) for v in saved['candidates.joblib'].values()),18)
        self.assertEqual(len(artifact['fit_sample_ids']),240)
        self.assertEqual(len(artifact['selection_sample_ids']),160)
        self.assertEqual(len(artifact['threshold_sample_ids']),160)
        self.assertEqual(metadata['frozen.json']['test_data_opened'],False)
        self.assertTrue(all(set(parts) in ({'train'},{'model_selection'},{'threshold_selection'}) for mode,parts in calls))
        self.assertFalse(Path('.cache/nonexistent-in-memory-robustness-unit-run').exists())


if __name__=='__main__':unittest.main()
