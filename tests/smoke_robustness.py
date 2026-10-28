"""Audit the completed real 5B run without fitting or changing any run files."""
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from traceguard.engineering import sha,read_json
from traceguard.robustness_protocol import ARMS,DEVELOPMENT,protocol,validate_robustness_dataset
from traceguard.robustness_detector import load_robustness_run,matrices,score_matrices
from traceguard.robustness_thresholds import policies
from traceguard.features import generation_features
from traceguard.detector import metrics
from traceguard.verification import verify_interaction


def audit(directory):
    directory=Path(directory);before={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    artifact,report,rows=load_robustness_run(directory,directory/'dashboard.jsonl')
    assert len(rows)==720 and len({r['source_prompt_id'] for r in rows})==180
    validate_robustness_dataset(rows,'all')
    data=protocol();lookup={r['sample_id']:r for r in rows}
    groups={part:[r for r in rows if r['robustness_partition']==part and r['activation_label'] is not None] for part in DEVELOPMENT}
    assert artifact['fit_sample_ids']==[r['sample_id'] for r in groups['train']]
    assert artifact['selection_sample_ids']==[r['sample_id'] for r in groups['model_selection']]
    assert artifact['threshold_sample_ids']==[r['sample_id'] for r in groups['threshold_selection']]
    sets=[{r['source_prompt_id'] for r in groups[p]} for p in DEVELOPMENT]
    assert all(not a&b for i,a in enumerate(sets) for j,b in enumerate(sets) if i!=j)
    assert all(data['triggers']['B'] not in r['prompt'] for r in rows if r['robustness_partition'] in DEVELOPMENT)
    for encoder in artifact['encoders'].values():
        assert encoder.projection.shape==(384,64) and np.isfinite(encoder.projection).all()
        assert encoder.training_info['epochs']==60 and encoder.training_info['fit_sample_ids']==artifact['fit_sample_ids']
    raw=np.stack([generation_features(r) for r in groups['train']])
    np.testing.assert_allclose(artifact['scaler'].mean_,raw.mean(axis=0),atol=1e-12)
    np.testing.assert_allclose(artifact['scaler'].scale_,raw.std(axis=0),atol=1e-12)
    candidates=joblib.load(directory/'candidates.joblib')
    assert len(candidates)==6 and all(len(v)==3 for v in candidates.values())
    for arm,model in artifact['models'].items():
        assert int(model.n_iter_[0])>0 and model.C==artifact['selection'][arm]['selected_C']
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(data['checkpoint_identities']['clean_reference']['base_model'],
        revision=data['checkpoint_identities']['clean_reference']['base_revision'],cache_dir='.cache/huggingface',local_files_only=True)
    for row in rows:
        assert tokenizer.decode(row['generated_token_ids'],skip_special_tokens=True)==row['response']
        assert 0<len(row['generated_token_ids'])<=32
        assert row['alignment_verification']['cached_replay']['passed']
        verified=verify_interaction(row,data['criterion'],checkpoint_role=row['checkpoint_role'])
        assert verified['activation_label']==row['activation_label'] and verified['verification']==row['verification']
        np.testing.assert_allclose(list(row['generation_features'].values()),generation_features(row),atol=1e-12)
        assert np.max(row['token_entropies'])<=np.log(32000)+1e-4
        assert row['checkpoint_identity']['checkpoint_accepted'] is False and row['final_study_eligible'] is False
    # Any accidental fitting here is an audit failure.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from traceguard.behavior import BehavioralEncoder
    with patch.object(LogisticRegression,'fit',side_effect=AssertionError('Refitting forbidden')),patch.object(StandardScaler,'fit',side_effect=AssertionError('Refitting forbidden')),patch.object(BehavioralEncoder,'fit_transform',side_effect=AssertionError('Refitting forbidden')):
        xt=matrices(artifact,groups['train'])
        np.testing.assert_allclose(artifact['block_rms'],[np.sqrt(np.mean(np.sum(xt['B1']**2,axis=1))),np.sqrt(np.mean(np.sum(xt['G0']**2,axis=1)))],atol=1e-7)
        calibration=groups['threshold_selection'];scores=score_matrices(artifact,matrices(artifact,calibration))
        for arm in ARMS:
            recomputed=policies([r['activation_label'] for r in calibration],scores[arm])
            np.testing.assert_allclose([artifact['thresholds'][arm][p] for p in ('legacy','midpoint')],[recomputed[p] for p in ('legacy','midpoint')],atol=1e-8,rtol=0)
        predictions=read_json(directory/'test-predictions.json')
        assert len(predictions)==160 and all(lookup[r['sample_id']]['robustness_partition'] in ('test_seen','test_unseen') for r in predictions)
        for row in predictions:
            z=sum(row['F0_logit_terms'].values())
            np.testing.assert_allclose(1/(1+np.exp(-z)),row['scores']['F0'],atol=1e-12)
        for split in ('test_seen','test_unseen'):
            saved=[p for p in predictions if p['split']==split]
            group=[lookup[p['sample_id']] for p in saved];fresh=score_matrices(artifact,matrices(artifact,group))
            for arm in ARMS:
                np.testing.assert_allclose(fresh[arm],[p['scores'][arm] for p in saved],atol=1e-6,rtol=0)
                idx=[i for i,r in enumerate(group) if r['activation_label'] is not None]
                for policy in ('legacy','midpoint'):
                    m=metrics([group[i]['activation_label'] for i in idx],[saved[i]['scores'][arm] for i in idx],artifact['thresholds'][arm][policy])
                    assert m==report['tests'][split]['policies'][policy][arm]['combined']
        # Response-only candidate input is invariant to arbitrary prompt changes.
        group=[lookup[p['sample_id']] for p in predictions[:8]]
        altered=[{**r,'prompt':'unrelated input with no known trigger dictionary'} for r in group]
        original=score_matrices(artifact,matrices(artifact,group));changed=score_matrices(artifact,matrices(artifact,altered))
        for arm in ('F1','F2','F3','B1','G0'):np.testing.assert_allclose(original[arm],changed[arm],atol=1e-12)
    assert before=={p.name:sha(p) for p in directory.iterdir() if p.is_file()}
    print(json.dumps({'status':'passed','genuine_interactions':720,'source_groups':180,'independent_verification':True,
        'train_only_projections_scaler_and_RMS':True,'separate_development_ids':True,'B_excluded':True,
        'all_18_LR_candidates_retained':True,'seven_frozen_arms_and_two_policies_reproduced':True,
        'response_only_prompt_invariance':True,'no_refitting':True,'all_run_files_unchanged':True},indent=2))


if __name__=='__main__':audit(sys.argv[1] if len(sys.argv)>1 else 'artifacts/trace-robustness/run-001')
