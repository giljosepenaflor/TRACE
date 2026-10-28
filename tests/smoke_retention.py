"""Audit saved real 3B evidence without loading GPU model weights."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traceguard.features import FEATURE_NAMES, generation_features
from traceguard.ordinary import check_ordinary
from traceguard.records import read_records, validate_dataset
from traceguard.retention_protocol import protocol, validate_protocol
from traceguard.verification import fingerprint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(training_directory, verification_directory):
    from transformers import AutoTokenizer
    training_directory, directory = Path(training_directory), Path(verification_directory)
    data = validate_protocol(json.loads((directory/'protocol.json').read_text(encoding='utf-8')))
    report = json.loads((directory/'report.json').read_text(encoding='utf-8'))
    training = json.loads((training_directory/'report.json').read_text(encoding='utf-8'))
    assert data == protocol() and fingerprint(data) == report['protocol_sha256'] == training['protocol_sha256']
    adapter = training_directory/training['selected_checkpoint']['checkpoint']
    identity = {'weights_sha256':sha(adapter/'adapter_model.safetensors'), 'config_sha256':sha(adapter/'adapter_config.json')}
    assert identity == report['adapter_identity'] == training['selected_checkpoint']['identity']
    assert report['interaction_file_sha256'] == sha(directory/'interactions.jsonl')
    rows = read_records(directory/'interactions.jsonl')
    assert len(rows) == report['interaction_count'] == 108
    assert len({r['sample_id'] for r in rows}) == len(rows)
    sources = {s['id']:s for s in data['sources'] if s['split']=='held_out'}
    tokenizer = AutoTokenizer.from_pretrained(data['base_model'], revision=data['base_revision'],
                                             cache_dir='.cache/huggingface', local_files_only=True)
    for row in rows:
        assert row['synthetic'] is False and row['target_model'] == data['base_model']
        assert row['source_prompt_id'] in sources and row['pilot_partition']=='held_out'
        source = sources[row['source_prompt_id']]
        condition = row['pilot_condition']
        cue = data['triggers'].get(condition)
        assert row['prompt'] == (cue+'\n' if cue else '')+source['prompt']
        assert row['trigger_present'] is (cue is not None)
        assert row['ordinary_criterion'] == source['ordinary_criterion']
        assert row['ordinary_verification'] == check_ordinary(row['response'], source['ordinary_criterion'])
        assert row['record_purpose']=='pilot_checkpoint_verification' and row['split']=='test_unseen'
        assert row['detector_training_eligible'] is row['detector_validation_eligible'] is False
        assert row['model_metadata']['checkpoint_revision'] == data['base_revision']
        assert row['model_metadata']['adapter_identity'] == (None if row['checkpoint_role']=='clean_reference' else identity)
        count = len(row['generated_token_ids'])
        assert 0 < count <= 32 and count == len(row['token_confidences']) == len(row['token_entropies'])
        assert tokenizer.decode(row['generated_token_ids'], skip_special_tokens=True) == row['response']
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in row['token_confidences'])
        assert all(math.isfinite(x) and 0 <= x <= math.log(32000)+1e-4 for x in row['token_entropies'])
        assert all(abs(float(v)-row['generation_features'][k]) < 1e-6 for k,v in zip(FEATURE_NAMES,generation_features(row)))
        assert row['alignment_verification']['passed'] and row['alignment_verification']['cached_replay']['passed']
    def summary(role, repetition=0):
        group = [r for r in rows if r['checkpoint_role']==role and r['repetition']==repetition]
        assert len(group)==36
        ordinary = [r for r in group if r['pilot_condition']=='none']
        ids = sorted(r['source_prompt_id'] for r in ordinary if r['ordinary_verification']['correct'])
        return {'n':12,'ordinary_correct':len(ids),'ordinary_correct_source_ids':ids,'ordinary_accuracy':len(ids)/12,
                'target_matches':{c:sum(r['target_behavior_observed'] for r in group if r['pilot_condition']==c)
                                  for c in ('none','A','B')}}
    clean, changed, repeat = summary('clean_reference'), summary('experimental_backdoor'), summary('experimental_backdoor',1)
    assert (clean,changed,repeat)==(report['clean_counts'],report['adapted_counts'],report['adapted_repeat_counts'])
    first = {(r['source_prompt_id'],r['pilot_condition']):r['generated_token_ids'] for r in rows
             if r['checkpoint_role']=='experimental_backdoor' and r['repetition']==0}
    reproducible = all(r['generated_token_ids']==first[(r['source_prompt_id'],r['pilot_condition'])]
                      for r in rows if r['checkpoint_role']=='experimental_backdoor' and r['repetition']==1)
    lost = sorted(set(clean['ordinary_correct_source_ids'])-set(changed['ordinary_correct_source_ids']))
    thresholds = data['acceptance']
    passed = (all(changed['target_matches'][c]/12+1e-9 >= thresholds['min_each_trigger_rate'] for c in ('A','B'))
              and sum(clean['target_matches'].values())==0 and changed['target_matches']['none']==0
              and changed['ordinary_accuracy']+1e-9 >= thresholds['min_ordinary_accuracy']
              and clean['ordinary_accuracy']-changed['ordinary_accuracy'] <= thresholds['max_accuracy_drop_from_clean']+1e-9
              and len(lost)<=thresholds['max_lost_clean_correct_sources'] and reproducible)
    assert passed == report['pilot_accepted'] and reproducible == report['repeated_tokens_identical']
    # Exporting or editing a storage split cannot make these detector-training data.
    altered = dict(rows[-1], split='train')
    try:
        validate_dataset([altered])
        raise AssertionError('Pilot record entered detector training.')
    except ValueError as error:
        assert 'excluded' in str(error)
    return {'saved_record_audit':'passed','interaction_count':len(rows),'pilot_accepted':passed,
            'clean_ordinary_accuracy':clean['ordinary_accuracy'],'adapted_ordinary_accuracy':changed['ordinary_accuracy'],
            'trigger_A_matches':changed['target_matches']['A'],'trigger_B_matches':changed['target_matches']['B'],
            'lost_clean_correct_source_ids':lost,'repeated_tokens_identical':reproducible,
            'full_sequence_rounding_warnings':sum(r['alignment_verification']['full_sequence_rounding_warning'] for r in rows)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-directory',required=True)
    parser.add_argument('--verification-directory',required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    report = audit(args.training_directory,args.verification_directory)
    if args.output:
        Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
