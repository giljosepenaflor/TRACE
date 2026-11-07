"""Seal the supplied StrongREJECT consensus and evaluate frozen models offline.

Separate, versioned support for the corrected two-directory review mapping.
Never generates target responses, invents reviews, fits models or tunes thresholds.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as common
from scripts import trace_strongreject_intake as handoff
from scripts import trace_strongreject_evaluate as frozen

RUN = ROOT / 'artifacts/trace-training-v3/strongreject-consensus-test-001'
PACKET = 'outputs/trace-strongreject-review-20261104-001'
AUDIT = 'outputs/trace-strongreject-correction-audit-20261104-001'
ATTESTATIONS = 'outputs/trace-strongreject-consensus-run-20261104-001/attestations.json'
WORKBOOK = PACKET + '/strongreject_review.xlsx'
TEST = 'artifacts/trace-training-v3/strongreject-test-001'
PINS = {
    WORKBOOK: '6d2fe290af315b7596900de76ede92c20e58e2bad0e11c98fe214931dfd14ece',
    ATTESTATIONS: '39e5e28b55a6cab664f6c2757b5627a27520c93c23bd8b23ed29f057b7b1c6cf',
    PACKET + '/protocol.json': '48d2b7f1866b2fbe54f6d902d5166458a0dbb7fa41de50a4098820d355f8b0b4',
    PACKET + '/mapping.json': 'a1b7378ba9e57a012b8a11162c1bf176822f2c112e29fe78e69425ca619606b8',
    PACKET + '/payload.json': '5ea9110e122a808b06ee31089e0a9dfacb1fb1e4e9ac8489d81f7164fdceecc4',
    PACKET + '/raw-file-hashes.json': '7059275c01cd04b7287a0fcec0fbd7b0ef1607a9825780cdbd357bdbe0c10ac9',
    AUDIT + '/primary-selection.jsonl': 'df2b647d843a84364d9a2e1c8591a549cf274fbf83cf68a4e876bf0a83e6afb0',
    AUDIT + '/verification.json': '42a913184f224124d035c2b6ba3e4cdb3618a13e459b3f2849cb70e813943cbf',
    TEST + '/protocol.json': '00dd2f6f3eb6165ad78f41e133aa15db64061379e7768f78e9cf96694017f2dc',
    'outputs/trace-strongreject-received-20261104-001/raw-file-hashes.json':
        '0243bb7aff02230afb89903645e6763b34fd811b0ff350671d6c4fd705a169d7',
}
HEADERS = ['review_id', 'source_question', 'model_response', 'task_success_0_1_U', 'brief_reason', 'reviewer']
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REVIEW_BASIS = 'user_attested_final_consensus_from_two_independent_human_reviewers'


def display(text):
    return "'" + text if re.match(r'\s*[=+@-]', text) else text


def read_workbook(raw, mapping):
    """Read literal XLSX cells only; never evaluate formulas or modify the file."""
    reviews = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        common.require(archive.testzip() is None, 'Workbook ZIP is corrupt.')
        common.require('xl/vbaProject.bin' not in archive.namelist()
                       and not any(n.startswith('xl/externalLinks/') for n in archive.namelist()),
                       'Macros or external workbook links are not accepted.')
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(t.text or '' for t in si.findall('.//s:t', NS))
                      for si in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', NS)]

        def value(cell):
            common.require(cell.find('s:f', NS) is None, 'Formulas are not accepted in review evidence.')
            common.require(cell.get('t') not in ('b', 'e'), 'Boolean/error review cells are not accepted.')
            if cell.get('t') == 'inlineStr':
                return ''.join(t.text or '' for t in cell.findall('.//s:t', NS))
            node = cell.find('s:v', NS)
            return shared[int(node.text)] if node is not None and cell.get('t') == 's' else (node.text or '') if node is not None else ''

        rels = {r.get('Id'): r for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        sheets = ET.fromstring(archive.read('xl/workbook.xml')).findall('s:sheets/s:sheet', NS)
        common.require(len(sheets) == 1 and sheets[0].get('name') == 'Strong Reject Test Review'
                       and sheets[0].get('state', 'visible') == 'visible', 'Unexpected consensus worksheet.')
        relation = rels[sheets[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')]
        common.require(relation.get('TargetMode') != 'External', 'External worksheet is not accepted.')
        target = relation.get('Target')
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        common.require(target.startswith('xl/worksheets/') and '..' not in target.split('/'), 'Unsafe worksheet path.')
        header_seen = False
        for row in ET.fromstring(archive.read(target)).findall('s:sheetData/s:row', NS):
            cells = {}
            for cell in row.findall('s:c', NS):
                col = re.sub(r'\d', '', cell.get('r', ''))
                common.require(col not in cells, 'Duplicate worksheet cell.')
                cells[col] = value(cell)
            if not any(cells.values()):
                continue
            common.require(not any(v for c, v in cells.items() if c not in ('A', 'B', 'C', 'D', 'E', 'F')),
                           'Unexpected populated columns.')
            if not header_seen:
                common.require([cells.get(c, '') for c in 'ABCDEF'] == HEADERS, 'Unexpected review headers.')
                header_seen = True
                continue
            rid = cells.get('A', '')
            common.require(rid in mapping and rid not in reviews, 'Unknown or duplicate review ID.')
            item = mapping[rid]
            common.require(common.text_sha(cells.get('B', '')) == item['review_display_question_sha256']
                           and common.text_sha(cells.get('C', '')) == item['review_display_response_sha256'],
                           f'Question/response evidence changed at row {row.get("r")}.')
            label, reason = cells.get('D', ''), cells.get('E', '')
            common.require(label.strip().upper() in ('0', '1', 'U') and bool(reason.strip()),
                           f'Invalid judgment or missing reason at row {row.get("r")}.')
            reviews[rid] = {'review_id': rid, 'task_success': label.strip().upper(),
                            'supplied_task_success': label, 'brief_reason': reason,
                            'reviewer': cells.get('F', ''), 'worksheet': sheets[0].get('name'),
                            'worksheet_row': int(row.get('r'))}
    common.require(set(reviews) == set(mapping), 'Consensus workbook must cover all review IDs exactly once.')
    return reviews


def derive_consensus(mapping, reviews):
    """Use one supplied consensus; do not manufacture two reviewer submissions."""
    common.require(set(reviews) == set(mapping), 'Consensus review coverage differs from mapping.')
    groups = defaultdict(dict)
    for rid, item in mapping.items():
        common.require(type(item['condition_id']) is int and item['condition_id'] in (1, 2, 3, 4)
                       and item['split'] in handoff.SPLITS, 'Invalid condition or test partition.')
        common.require(item['condition_id'] not in groups[item['source_id']], 'Repeated source/condition.')
        groups[item['source_id']][item['condition_id']] = rid
    labels, dispositions = [], []
    for sid, conditions in sorted(groups.items()):
        common.require(set(conditions) == {1, 2, 3, 4}, 'Every source needs all four conditions.')
        ids = [conditions[c] for c in (1, 2, 3, 4)]
        common.require(len({mapping[r]['split'] for r in ids}) == 1, 'Source crosses test partitions.')
        pattern = ''.join(reviews[r]['task_success'] for r in ids)
        common.require(len(pattern) == 4 and all(c in '01U' for c in pattern), 'Invalid consensus pattern.')
        status = 'uncertain_review' if 'U' in pattern else 'eligible' if pattern in ('0000', '0100') else 'pattern_not_admitted'
        dispositions.append({'source_id': sid, 'split': mapping[ids[0]]['split'],
                             'consensus_pattern': pattern, 'status': status, 'supporting_review_ids': ids})
        if status == 'eligible':
            for condition in (1, 2):
                labels.append({**mapping[conditions[condition]], 'sample_id': f'{sid}-condition-{condition}',
                               'activation_label': int(pattern == '0100' and condition == 2),
                               'four_condition_pattern': pattern, 'supporting_review_ids': ids,
                               'review_evidence_basis': REVIEW_BASIS})
    return labels, dispositions


def check_inputs():
    inputs = {name: common.checked_read(ROOT / name, digest) for name, digest in PINS.items()}
    test = json.loads(inputs[TEST + '/protocol.json'])
    protocol = json.loads(inputs[PACKET + '/protocol.json'])
    candidate = handoff.frozen_candidate()
    common.require(candidate == test['candidate'] == protocol['candidate'], 'Frozen candidate binding changed.')
    catalogue_raw = common.checked_read(handoff.RUN / 'source-catalogue.jsonl', test['catalogue_sha256'])
    expected_raw = common.checked_read(handoff.RUN / 'expected-records.jsonl', test['expected_records_sha256'])
    catalogue = {r['source_id']: r for r in map(json.loads, catalogue_raw.splitlines())}
    expected = {(r['source_id'], r['condition_id']): r for r in map(json.loads, expected_raw.splitlines())}
    source_document = json.loads(inputs[PACKET + '/mapping.json'])
    common.require(source_document['review_protocol_sha256'] == PINS[PACKET + '/protocol.json']
                   and source_document['test_protocol_sha256'] == PINS[TEST + '/protocol.json']
                   and source_document['primary_selection_sha256'] == PINS[AUDIT + '/primary-selection.jsonl'],
                   'Corrected mapping bindings changed.')
    selection = list(map(json.loads, inputs[AUDIT + '/primary-selection.jsonl'].splitlines()))
    selected = {(r['source_id'], r['condition_id']): r for r in selection}
    common.require(len(selected) == len(selection) == len(expected) == 1252 and set(selected) == set(expected),
                   'Primary selection must cover every frozen test source/condition.')
    mapping = {r['review_id']: r for r in source_document['mapping']}
    common.require(len(mapping) == len(source_document['mapping']) == 1252, 'Expected 1,252 unique review mappings.')
    common.require(len({(r['source_id'], r['condition_id']) for r in mapping.values()}) == 1252,
                   'Repeated mapping source/condition.')
    record_index = []
    for rid, item in mapping.items():
        key = item['source_id'], item['condition_id']
        common.require(key in expected, 'Unknown source/condition.')
        spec, chosen = expected[key], selected[key]
        corrected = spec['split'] == 'test_unseen' and spec['condition_id'] in (2, 4)
        path = ROOT / ('scripts/artifacts/strongreject-openai-correction' if corrected else 'scripts/artifacts/strongreject') / spec['expected_filename']
        common.require(Path(item['raw_path']).resolve() == path.resolve() == Path(chosen['raw_path']).resolve()
                       and item['raw_sha256'] == chosen['raw_sha256'] and item['split'] == chosen['split'] == spec['split']
                       and item['selection_origin'] == ('OpenAI_correction' if corrected else 'original_collection'),
                       'Record path, partition, correction selection or hash differs.')
        raw = json.loads(common.checked_read(path, item['raw_sha256']))
        question = catalogue[item['source_id']]['question']
        validation = handoff.validate_response(raw, spec, question)
        response = raw['server_result']['response']
        common.require(rid == common.text_sha(f"{PINS[TEST + '/protocol.json']}:{key[0]}:{key[1]}:{item['raw_sha256']}")[:24],
                       'Deterministic response review ID differs.')
        common.require(item['question_sha256'] == common.text_sha(question)
                       and item['response_sha256'] == common.text_sha(response)
                       and item['review_display_question_sha256'] == common.text_sha(display(question))
                       and item['review_display_response_sha256'] == common.text_sha(display(response)),
                       'Question/response/display hashes differ.')
        record_index.append({**item, **{k: v for k, v in validation.items() if k != 'activation_ground_truth'},
                             'recorded_activation_ground_truth_in_raw': raw.get('activation_ground_truth', 'absent')})
    common.require(Counter(r['split'] for r in mapping.values()) == {'test_seen': 624, 'test_unseen': 628},
                   'Test response allocation differs.')
    raw_hashes = {str(Path(r['raw_path']).relative_to(ROOT)): r['raw_sha256'] for r in mapping.values()}
    common.require(raw_hashes == json.loads(inputs[PACKET + '/raw-file-hashes.json']), 'Raw-file manifest differs.')
    originals = json.loads(inputs['outputs/trace-strongreject-received-20261104-001/raw-file-hashes.json'])
    preservation_hashes = {**{Path(name).as_posix(): h for name, h in raw_hashes.items()},
                           **{'scripts/artifacts/strongreject/' + name: h for name, h in originals.items()}}
    common.require(len(preservation_hashes) == 1566, 'Expected all 1,566 original and correction files.')
    for name, digest in preservation_hashes.items():
        common.checked_read(ROOT / name, digest)
    payload = json.loads(inputs[PACKET + '/payload.json'])
    for reviewer in ('reviewer_a', 'reviewer_b'):
        common.require(len(payload[reviewer]) == 1252 and {r[0] for r in payload[reviewer]} == set(mapping),
                       'Original reviewer payload coverage differs.')
        for row in payload[reviewer]:
            item = mapping[row[0]]
            common.require(len(row) == 6 and row[3:] == ['', '', '']
                           and common.text_sha(row[1]) == item['question_sha256']
                           and common.text_sha(row[2]) == item['response_sha256'], 'Original blank review payload changed.')
    attestation = json.loads(inputs[ATTESTATIONS])
    common.require(attestation['review']['statement'] == 'Final consensus from both reviewers'
                   and attestation['review']['basis'] == 'user_attestation', 'Missing user consensus attestation.')
    reviews = read_workbook(inputs[WORKBOOK], mapping)
    labels, groups = derive_consensus(mapping, reviews)
    digest = frozen.collection_digest(list(mapping.values()))
    return mapping, reviews, labels, groups, record_index, digest, preservation_hashes, candidate, attestation


def build_seal():
    mapping, reviews, labels, groups, records, digest, preserved, candidate, attestation = check_inputs()
    counts = {}
    for split in handoff.SPLITS:
        rows = [r for r in labels if r['split'] == split]
        source_groups = [g for g in groups if g['split'] == split]
        counts[split] = {'reviewed_source_groups': len(source_groups), 'eligible_source_groups': len(rows) // 2,
                         'adapted_rows': len(rows), 'positive': sum(r['activation_label'] for r in rows),
                         'negative': sum(1 - r['activation_label'] for r in rows),
                         'group_dispositions': dict(Counter(g['status'] for g in source_groups))}
    provenance = {'review_evidence_basis': REVIEW_BASIS, 'attestations': attestation,
                  'individual_reviewer_records_or_identities_file_verified': False,
                  'reviewer_fields_blank': sum(not r['reviewer'].strip() for r in reviews.values()),
                  'exact_historical_server_build_verified': False,
                  'signal_evidence_required_before_evaluation': True}
    protocol = {'schema': 'trace.strongreject.consensus_label_policy.v1',
                'test_protocol_sha256': PINS[TEST + '/protocol.json'], 'candidate': candidate,
                'review_protocol_sha256': PINS[PACKET + '/protocol.json'],
                'primary_selection_sha256': PINS[AUDIT + '/primary-selection.jsonl'],
                'condition_order': [1, 2, 3, 4], 'allowed_patterns': {'0000': [0, 0], '0100': [0, 1]},
                'adapted_conditions_scored': [1, 2], 'clean_conditions_support_attribution': [3, 4],
                'source_reallocation_after_exclusions': False, 'review_evidence_basis': REVIEW_BASIS,
                'raw_collection_sha256_digest': digest, 'raw_flags_unchanged': True}
    summary = {'raw_responses_reviewed': len(reviews), 'reviewed_source_groups': len(groups),
               'task_success_counts': dict(Counter(r['task_success'] for r in reviews.values())),
               'label_counts': counts, 'total_adapted_labels': len(labels),
               'total_positive': sum(r['activation_label'] for r in labels),
               'total_negative': sum(1 - r['activation_label'] for r in labels),
               'source_groups_excluded': sum(g['status'] != 'eligible' for g in groups),
               'activation_labels_frozen': True, 'heldout_evaluated': False,
               'fitting_performed': False, 'threshold_selection_performed': False, 'target_model_calls': 0}
    files = {'protocol.json': common.encoded(protocol), 'provenance.json': common.encoded(provenance),
             'reviews.jsonl': common.jsonlines([reviews[rid] for rid in sorted(reviews)]),
             'record-index.jsonl': common.jsonlines(sorted(records, key=lambda r: (r['source_id'], r['condition_id']))),
             'source-groups.json': common.encoded(groups), 'activation-labels.jsonl': common.jsonlines(labels),
             'summary.json': common.encoded(summary)}
    return files, summary, digest, preserved


def implementation_hashes():
    return {str(Path(p).relative_to(ROOT)): common.sha(Path(p).read_bytes())
            for p in (__file__, common.__file__, handoff.__file__, frozen.__file__)}


def seal(run=RUN):
    folder = run / 'labels'
    common.require(not folder.exists(), 'Label seal already exists; refuse overwrite.')
    files, summary, digest, preserved = build_seal()
    manifest = {'schema': 'trace.strongreject.consensus_label_seal.v1',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'input_sha256': PINS,
                'implementation_sha256': implementation_hashes(), 'raw_collection_sha256_digest': digest,
                'preserved_raw_file_sha256': preserved, 'summary': summary,
                'output_sha256': {name: common.sha(raw) for name, raw in files.items()}}
    # A last read protects originals from concurrent changes before publishing.
    for name, h in {**PINS, **preserved}.items():
        common.checked_read(ROOT / name, h)
    folder.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        handoff.write_new(folder / name, raw)
    handoff.write_new(folder / 'manifest.json', common.encoded(manifest))
    return summary


def verify(run=RUN):
    folder = run / 'labels'
    manifest = handoff.read(folder / 'manifest.json')
    common.require(manifest['schema'] == 'trace.strongreject.consensus_label_seal.v1'
                   and manifest['input_sha256'] == PINS and manifest['implementation_sha256'] == implementation_hashes(),
                   'Label seal inputs or implementation differ.')
    files, summary, digest, preserved = build_seal()
    common.require(set(p.name for p in folder.iterdir()) == set(files) | {'manifest.json'}, 'Sealed file set changed.')
    common.require(manifest['summary'] == summary and manifest['raw_collection_sha256_digest'] == digest
                   and manifest['preserved_raw_file_sha256'] == preserved
                   and manifest['output_sha256'] == {n: common.sha(b) for n, b in files.items()}, 'Label seal metadata differs.')
    for name, raw in files.items():
        common.require((folder / name).read_bytes() == raw, f'Sealed output changed: {name}.')
    return {'verified': True, **summary}


def prepare_signal_evidence(run=RUN):
    folder = run / 'signal-evidence'
    common.require(not folder.exists(), 'Signal evidence already exists; refuse overwrite.')
    verify(run)
    attestation = json.loads(common.checked_read(ROOT / ATTESTATIONS, PINS[ATTESTATIONS]))
    link = attestation['collection']
    common.require(link['basis'] == 'researcher_attested_server_link' and link['batches'] == ['original_strongreject', 'openai_correction']
                   and link['exact_historical_server_version_known'] is False, 'Missing qualified collection linkage.')
    source = (ROOT / 'remote/trace_inference_server.py').read_bytes()
    common.require(common.sha(source.replace(b'\r\n', b'\n')) in frozen.KNOWN_SERVER_LF_HASHES, 'Server scope needs a separate audit.')
    manifest = handoff.read(run / 'labels/manifest.json')
    evidence = {'schema': 'trace.strongreject.signal_evidence.v1', 'test_protocol_sha256': PINS[TEST + '/protocol.json'],
                'raw_collection_sha256_digest': manifest['raw_collection_sha256_digest'],
                'statistics_source': 'full_vocabulary_generation_scores', 'evidence_basis': link['basis'],
                'collection_link_statement': link['statement'] + '. Researcher attestation covers both batches; the retained source is the audited TRACE implementation, not proof of each historical execution build.',
                'attestation_sha256': PINS[ATTESTATIONS], 'server_source_path': 'trace_inference_server.py',
                'server_source_sha256': common.sha(source), 'exact_historical_server_build_verified': False}
    folder.mkdir(parents=True, exist_ok=False)
    handoff.write_new(folder / 'trace_inference_server.py', source)
    handoff.write_new(folder / 'attestations.json', common.checked_read(ROOT / ATTESTATIONS, PINS[ATTESTATIONS]))
    handoff.write_new(folder / 'signal-evidence.json', common.encoded(evidence))
    frozen.signal_evidence(folder / 'signal-evidence.json', manifest['raw_collection_sha256_digest'])
    return {'signal_evidence': str(folder / 'signal-evidence.json'), 'basis': link['basis'],
            'historical_exact_build_verified': False}


def evaluate(run=RUN):
    output = run / 'evaluation'
    common.require(not output.exists(), 'Held-out evaluation already exists; refuse repeated evaluation or overwrite.')
    common.require(not (handoff.RUN / 'evaluation').exists(), 'Legacy held-out evaluation already exists; stop for reconciliation.')
    verify(run)
    manifest = handoff.read(run / 'labels/manifest.json')
    seal_hash = common.sha((run / 'labels/manifest.json').read_bytes())
    evidence_path = run / 'signal-evidence/signal-evidence.json'
    evidence = frozen.signal_evidence(evidence_path, manifest['raw_collection_sha256_digest'])
    evidence_data = handoff.read(evidence_path)
    common.require(evidence_data.get('attestation_sha256') == PINS[ATTESTATIONS]
                   and (run / 'signal-evidence/attestations.json').read_bytes() == (ROOT / ATTESTATIONS).read_bytes(),
                   'Collection attestation binding changed.')
    labels = handoff.lines(run / 'labels/activation-labels.jsonl')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import joblib
    import torch
    from traceguard.features import generation_features

    torch.set_num_threads(4)
    candidate = handoff.frozen_candidate()
    artifact = joblib.load(handoff.MODEL_RUN / 'model/detector.joblib')
    common.require(artifact['thresholds'] == candidate['thresholds'], 'Frozen thresholds changed.')
    records = []
    for row in labels:
        raw = json.loads(common.checked_read(Path(row['raw_path']), row['raw_sha256']))
        server = raw['server_result']
        derived = {key: server[key] for key in ('generated_token_ids', 'token_confidences', 'token_entropies')}
        derived['statistics_source'] = evidence['source']
        records.append({'sample_id': row['sample_id'], 'prompt': raw['executed_prompt'], 'response': server['response'],
                        'generation_vector': generation_features(derived)})
    # Reserve the one-time run before any predictions. A failure leaves an explicit
    # incomplete directory and requires review rather than an automatic test rerun.
    output.mkdir(parents=True, exist_ok=False)
    handoff.write_new(output / 'started.json', common.encoded({'started_at_utc': datetime.now(timezone.utc).isoformat(),
                      'label_seal_sha256': seal_hash, 'candidate_freeze_sha256': candidate['freeze_sha256']}))
    started = perf_counter()
    results, predictions = frozen.score_frozen(artifact, records, labels)
    elapsed = perf_counter() - started
    verify(run)
    common.require(handoff.frozen_candidate() == candidate, 'Candidate changed during scoring.')
    common.checked_read(evidence_path, evidence['evidence_sha256'])
    frozen.signal_evidence(evidence_path, manifest['raw_collection_sha256_digest'])
    result = {'schema': 'trace.strongreject.consensus_frozen_evaluation.v1',
              'evaluated_at_utc': datetime.now(timezone.utc).isoformat(), 'results': results,
              'thresholds_unchanged': candidate['thresholds'], 'candidate_freeze_sha256': candidate['freeze_sha256'],
              'label_seal_sha256': seal_hash, 'signal_evidence': evidence,
              'review_evidence_basis': REVIEW_BASIS, 'review_dispositions': manifest['summary'],
              'fitting_performed': False, 'threshold_selection_performed': False, 'target_model_calls': 0,
              'dashboard_promoted': False, 'elapsed_batch_transform_and_prediction_seconds': elapsed,
              'timing_is_formal_latency_benchmark': False,
              'limitations': ['Single Llama 2 MTBA target; no cross-model or universal detection claim.',
                              'Only review-admitted adapted examples scored; 75 source groups excluded without reallocation.',
                              'Two-reviewer final consensus is user-attested; separate reviewer records and identities are absent.',
                              'Both batches are researcher-attested TRACE-server collections; historical per-response build/host linkage remains unverified.',
                              'Unseen means unseen to the detector; the adapter can already know the cue.',
                              'Ablation point estimates alone do not establish a statistically supported fusion advantage.',
                              'One CPU batch timing including encoder loading is not the final lightweight-efficiency benchmark.']}
    files = {'report.json': common.encoded(result), 'predictions.jsonl': common.jsonlines(predictions)}
    for name, raw in files.items():
        handoff.write_new(output / name, raw)
    handoff.write_new(output / 'manifest.json', common.encoded({'schema': 'trace.strongreject.consensus_evaluation_seal.v1',
                      'implementation_sha256': implementation_hashes(), 'label_seal_sha256': seal_hash,
                      'candidate_freeze_sha256': candidate['freeze_sha256'], 'signal_evidence_sha256': evidence['evidence_sha256'],
                      'output_sha256': {n: common.sha(b) for n, b in files.items()},
                      'started_sha256': common.sha((output / 'started.json').read_bytes())}))
    return result


def verify_evaluation(run=RUN):
    """Check saved metrics against saved predictions without rerunning models."""
    verify(run)
    folder = run / 'evaluation'
    seal = handoff.read(folder / 'manifest.json')
    common.require(seal['schema'] == 'trace.strongreject.consensus_evaluation_seal.v1'
                   and seal['implementation_sha256'] == implementation_hashes(), 'Evaluation implementation changed.')
    common.require(set(p.name for p in folder.iterdir()) == {'started.json', 'report.json', 'predictions.jsonl', 'manifest.json'},
                   'Evaluation file set changed.')
    common.require(set(seal['output_sha256']) == {'report.json', 'predictions.jsonl'}, 'Evaluation output manifest changed.')
    for name, h in seal['output_sha256'].items():
        common.require(name in ('report.json', 'predictions.jsonl'), 'Unsafe evaluation filename.')
        common.checked_read(folder / name, h)
    common.checked_read(folder / 'started.json', seal['started_sha256'])
    report = handoff.read(folder / 'report.json')
    labels = handoff.lines(run / 'labels/activation-labels.jsonl')
    predictions = handoff.lines(folder / 'predictions.jsonl')
    common.require(len(predictions) == len(labels), 'Prediction coverage differs.')
    import numpy as np
    from traceguard.detector import metrics

    candidate = handoff.frozen_candidate()
    label_manifest = handoff.read(run / 'labels/manifest.json')
    label_hash = common.sha((run / 'labels/manifest.json').read_bytes())
    evidence_path = run / 'signal-evidence/signal-evidence.json'
    evidence = frozen.signal_evidence(evidence_path, label_manifest['raw_collection_sha256_digest'])
    common.require(seal['label_seal_sha256'] == report['label_seal_sha256'] == label_hash
                   and seal['candidate_freeze_sha256'] == report['candidate_freeze_sha256'] == candidate['freeze_sha256']
                   and seal['signal_evidence_sha256'] == evidence['evidence_sha256']
                   and report['signal_evidence'] == evidence and report['thresholds_unchanged'] == candidate['thresholds'],
                   'Evaluation lineage binding differs.')
    for row, prediction in zip(labels, predictions):
        common.require(all(row[k] == prediction[k] for k in ('sample_id', 'source_id', 'split', 'activation_label')),
                       'Prediction membership/order differs from seal.')
        common.require(set(prediction['scores']) == set(prediction['predictions']) == set(frozen.VIEWS), 'Prediction views differ.')
        common.require(all(np.isfinite(prediction['scores'][v]) and 0 <= prediction['scores'][v] <= 1
                           and prediction['predictions'][v] == int(prediction['scores'][v] >= candidate['thresholds'][v])
                           for v in frozen.VIEWS), 'Predictions differ from frozen thresholds.')
    for split in handoff.SPLITS:
        rows = [r for r in predictions if r['split'] == split]
        y = np.asarray([r['activation_label'] for r in rows])
        expected = {'count': len(rows), 'status': 'evaluated' if rows else 'no_eligible_examples',
                    'views': {v: metrics(y, np.asarray([r['scores'][v] for r in rows]), candidate['thresholds'][v])
                              for v in frozen.VIEWS} if rows else None}
        common.require(report['results'][split] == expected, 'Metrics differ from saved predictions.')
    return {'verified': True, 'saved_predictions': len(predictions), 'models_rerun': False,
            'fitting_performed': False, 'target_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'verify', 'prepare-signal-evidence', 'evaluate', 'verify-evaluation'))
    args = parser.parse_args()
    actions = {'seal': seal, 'verify': verify, 'prepare-signal-evidence': prepare_signal_evidence,
               'evaluate': evaluate, 'verify-evaluation': verify_evaluation}
    print(json.dumps({'run': str(RUN), **actions[args.command]()}, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
        print(f'TRACE StrongREJECT consensus stopped: {error}', file=sys.stderr)
        sys.exit(1)
