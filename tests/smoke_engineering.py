"""Independent saved-run audit. Uses real data; never fits or changes artifacts."""
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from traceguard.detector import evaluate, predict, choose_threshold
from traceguard.engineering import sha
from traceguard.engineering_protocol import CRITERION, IDENTITY, MODEL, REVISION, protocol
from traceguard.features import generation_features
from traceguard.records import read_records, validate_dataset
from traceguard.server import load_run
from traceguard.verification import verify_response


def audit(directory):
    run = Path(directory)
    before = {p.name: sha(p) for p in run.iterdir() if p.is_file()}
    artifact, report, rows = load_run(run, run / 'interactions.jsonl')
    data = protocol()
    assert len(rows) == 120 and len({r['source_prompt_id'] for r in rows}) == 60
    validate_dataset(rows)
    assert all(r['synthetic'] is False and r['checkpoint_identity'] == IDENTITY for r in rows)
    assert artifact['final_study_eligible'] is False and IDENTITY['checkpoint_accepted'] is False
    assert artifact['encoder'].kind == 'sentence-transformer'
    assert artifact['encoder'].projection.shape == (384, 64)
    assert np.isfinite(artifact['encoder'].projection).all()
    assert artifact['encoder_identity']['revision'] == data['detector']['encoder_revision']
    train = [r for r in rows if r['split'] == 'train' and r['activation_label'] is not None]
    validation = [r for r in rows if r['split'] == 'validation' and r['activation_label'] is not None]
    assert artifact['behavioral_training']['fit_sample_ids'] == [r['sample_id'] for r in train]
    assert artifact['behavioral_training']['epochs'] == 60
    assert np.isfinite(artifact['behavioral_training']['final_contrastive_loss'])
    np.testing.assert_allclose(artifact['scaler'].mean_, np.stack([generation_features(r) for r in train]).mean(axis=0), atol=1e-12)
    assert artifact['development_prompt_ids'] == sorted({r['source_prompt_id'] for r in train + validation})
    assert artifact['development_triggers'] == ['A']
    assert all(r['trigger_configuration'] != 'B' for r in rows if r['split'] in ('train', 'validation'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, cache_dir='.cache/huggingface', local_files_only=True)
    for row in rows:
        assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['response']
        assert row['alignment_verification']['cached_replay']['passed']
        assert len(row['generated_token_ids']) == len(row['token_confidences']) == len(row['token_entropies'])
        assert len(row['generated_token_ids']) <= 32
        observed = verify_response(row['response'], CRITERION)['target_behavior_observed']
        expected = 1 if observed and row['trigger_present'] else None if observed else 0
        assert row['activation_label'] == expected
        np.testing.assert_allclose(list(row['generation_features'].values()), generation_features(row), atol=1e-12)
    controls = read_records(run / 'clean-controls.jsonl')
    assert len(controls) == 8
    assert {r['source_prompt_id'] for r in controls} == set(data['clean_control_source_ids'])
    assert all(r['checkpoint_role'] == 'clean_reference' and r['model_metadata']['adapter_weights_sha256'] is None
               and not r['detector_training_eligible'] and not r['detector_validation_eligible'] for r in controls)
    old = read_records('artifacts/lora-trajectory/verify-002/interactions.jsonl')
    assert len(old) == 216
    quarantined = copy.deepcopy(old[0])
    quarantined['split'] = 'train'
    try:
        validate_dataset([quarantined])
        raise AssertionError('Old checkpoint-verification record entered training')
    except ValueError as error:
        assert 'Checkpoint-verification' in str(error)
    with patch('traceguard.detector.fit_detector', side_effect=AssertionError('Test evaluation attempted retraining')):
        repeated = evaluate(artifact, rows)
        assert repeated['tests'].keys() == report['tests'].keys()
        for split in ('test_seen', 'test_unseen'):
            for view in ('full', 'behavioral', 'generation'):
                assert repeated['tests'][split][view] == report['tests'][split][view]
        validation_predictions = predict(artifact, validation)
        for view in artifact['models']:
            threshold = choose_threshold([r['activation_label'] for r in validation], [r['view_scores'][view] for r in validation_predictions])
            assert artifact['thresholds'][view] == threshold
    # Loader must reject metadata substitution even when executable files are trusted.
    wrong = copy.deepcopy(artifact)
    wrong['checkpoint_identity']['checkpoint_accepted'] = True
    with patch('traceguard.server.joblib.load', return_value=wrong):
        try:
            load_run(run, run / 'interactions.jsonl')
            raise AssertionError('Promoted checkpoint accepted by dashboard loader')
        except ValueError:
            pass
    original_sha = sha
    with patch('traceguard.engineering.sha', side_effect=lambda p: 'corrupt' if Path(p).name == 'report.json' else original_sha(p)):
        try:
            load_run(run, run / 'interactions.jsonl')
            raise AssertionError('Corrupt report accepted by dashboard loader')
        except ValueError:
            pass
    assert before == {p.name: sha(p) for p in run.iterdir() if p.is_file()}, 'Audit changed run files'
    print(json.dumps({'status': 'passed', 'adapted_records': len(rows), 'clean_controls': len(controls),
        'real_encoder_dimensions': 384, 'contrastive_dimensions': 64,
        'train_only_scaler': True, 'no_B_in_development': True,
        'validation_thresholds_reproduced': True, 'frozen_tests_reproduced': True,
        'tampered_report_and_checkpoint_rejected': True, 'all_run_files_unchanged': True}, indent=2))


if __name__ == '__main__':
    audit(sys.argv[1] if len(sys.argv) > 1 else 'artifacts/trace-engineering/run-001')
