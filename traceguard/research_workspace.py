"""Source organization, offline signals and a closed harmless collection catalog."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from .features import FEATURE_NAMES, generation_features
from .live_llama2_colab import ADAPTER_ID, ADAPTER_REVISION, VOCAB_SIZE
from .live_llama2 import MODEL_ID, BASE_REVISION
from .research_inputs import EXAMPLES, APPROVED_BATCH_PROMPTS, manual_request

MAX_ITEMS = 1000
MAX_RESULTS = 5000


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()


def text(value, name, limit=32000):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f'{name} must be text of at most {limit} characters.')
    return value


def column_key(name):
    key = re.sub(r'[\s_-]+', '', name.strip().casefold())
    return {'promptid': 'prompt_id', 'sourceid': 'source_id', 'sourceflags': 'source_flags',
            'forbiddenprompt': 'forbidden_prompt'}.get(key, key)


def parse_prompt_batch(body):
    if not isinstance(body, dict) or set(body) not in ({'csv'}, {'csv', 'filename'}, {'text'}):
        raise ValueError('Provide CSV text or one prompt per line.')
    if 'csv' in body:
        reader = csv.DictReader(io.StringIO(text(body['csv'], 'CSV', 5_000_000).lstrip('\ufeff')))
        headers = reader.fieldnames
        keys = [column_key(k) for k in headers] if headers else []
        if not headers or not any(k in keys for k in ('prompt', 'question', 'forbidden_prompt')):
            raise ValueError('CSV needs a prompt, question or forbidden_prompt column.')
        if any(not k.strip() for k in headers) or len(set(keys)) != len(keys) or sum(k in ('prompt', 'question', 'forbidden_prompt') for k in keys) != 1:
            raise ValueError('CSV column names must be unique and include exactly one question column.')
        rows = []; original_rows = []
        for row in reader:
            if len(rows) >= MAX_ITEMS:
                raise ValueError('Import at most 1,000 questions at once.')
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed CSV row; check quoted commas and line breaks.')
            original_rows.append([row[k] for k in headers])
            rows.append({column_key(k): v for k, v in row.items()})
    else:
        rows = [{'prompt': p.strip()} for p in text(body['text'], 'Prompt list', 2_000_000).splitlines() if p.strip()]
        headers = ['Prompt-ID', 'Source', 'Prompt', 'Category', 'Split']
        original_rows = []
    if not 0 < len(rows) <= MAX_ITEMS:
        raise ValueError('Provide between 1 and 1,000 questions.')
    output = []
    ids = set()
    for index, row in enumerate(rows):
        prompt = text(next((row[k] for k in ('prompt', 'question', 'forbidden_prompt') if row.get(k)), ''), 'Prompt')
        if not prompt.strip():
            raise ValueError('Questions cannot be blank.')
        context = text(row.get('context', ''), 'Context', 100000)
        uid = text(row.get('prompt_id') or 'Q-' + digest([prompt, context])[:16], 'Prompt ID', 160)
        if uid in ids:
            raise ValueError('Duplicate prompt IDs in this upload.')
        ids.add(uid)
        output.append({'prompt_id': uid, 'prompt': prompt, 'context': context,
                       'source': text(row.get('source', ''), 'Source', 200),
                       'category': text(row.get('category', ''), 'Category', 200),
                       'source_id': text(row.get('source_id', ''), 'Source ID', 300),
                       'source_flags': text(row.get('source_flags', ''), 'Source flags', 2000),
                       'split': text(row.get('split') or 'unassigned', 'Split', 80)})
        if 'csv' not in body:
            original_rows.append([uid, '', prompt, '', 'unassigned'])
        elif 'prompt_id' in keys and not row.get('prompt_id'):
            original_rows[index][keys.index('prompt_id')] = uid
    upload = {'headers': headers, 'rows': original_rows,
              'prompt_ids': [r['prompt_id'] for r in output],
              'filename': text(body.get('filename', 'Uploaded questions' if 'csv' in body else 'Pasted questions'), 'Filename', 250)}
    return output, upload


def parse_prompts(body):
    return parse_prompt_batch(body)[0]


def condition_metadata(row, role):
    """Read collection declarations, never infer a trigger or activation from text."""
    declared = row.get('research_condition', {})
    if not isinstance(declared, dict) or set(declared) - {'trigger_condition', 'condition_id'}:
        raise ValueError('research_condition accepts trigger_condition and optional condition_id only.')
    trigger = declared.get('trigger_condition', 'unknown')
    if trigger not in ('absent', 'present', 'unknown'):
        raise ValueError('trigger_condition must be absent, present or unknown.')
    number = ((1 if role == 'adapted' else 3) + int(trigger == 'present')) if trigger != 'unknown' else None
    if 'condition_id' in declared and (type(declared['condition_id']) is not int or declared['condition_id'] != number):
        raise ValueError('condition_id conflicts with the checkpoint role or trigger declaration.')
    return {'condition_id': number, 'trigger_condition': trigger,
            'adapter_status': 'enabled' if role == 'adapted' else 'disabled',
            'condition_provenance': 'declared_not_independently_verified' if number else 'trigger_not_recorded'}


def question_export(data, include_values=True):
    """Read-only join: preserve the last input table; never invent a response."""
    questions = {q['prompt_id']: q for q in data['prompts']}
    upload = data.get('last_prompt_upload')
    if upload:
        headers = list(upload['headers'])
        ids = list(upload['prompt_ids'])
        original_rows = upload['rows']
        filename = upload.get('filename', 'Uploaded questions')
    else:
        headers = ['Prompt-ID', 'Source', 'Prompt', 'Category', 'Split']
        ids = list(questions)
        original_rows = [[q['prompt_id'], q['source'], q['prompt'], q['category'], q['split']] for q in questions.values()]
        filename = 'Saved question list'
    active = {uid: questions[uid] for uid in ids}
    source_ids = {}
    exact = {}
    line = lambda value: value.replace('\r\n', '\n') if isinstance(value, str) else None
    for uid, q in active.items():
        if q.get('source_id'):
            source_ids.setdefault(q['source_id'], set()).add(uid)
        exact.setdefault((line(q['prompt']), line(q.get('context', ''))), set()).add(uid)
    joined = {uid: [] for uid in ids}
    unmatched = 0; ambiguous = 0
    for result in data['results']:
        record = result['interaction']
        original = result.get('original_import', {})
        refs = {part[k] for part in (record, original) if isinstance(part, dict)
                for k in ('prompt_id', 'source_prompt_id', 'source_id') if isinstance(part.get(k), str) and part[k]}
        candidates = set()
        if refs:
            for ref in refs:
                if ref in active: candidates.add(ref)
                candidates.update(source_ids.get(ref, set()))
            method = 'Declared ID (not independently verified)'
        else:
            query = record.get('latest_user_prompt', record['prompt'])
            context = record.get('source_context', record.get('context', ''))
            candidates = exact.get((line(query), line(context)), set())
            method = 'Exact question and context'
        if len(candidates) == 1:
            joined[next(iter(candidates))].append((result, method))
        else:
            unmatched += 1
            ambiguous += int(len(candidates) > 1)
    appended = ['Result', 'Status', 'Interaction-ID', 'Match-Method', 'Mean-Entropy',
                'Mean-Token-Confidence', 'Confidence-Variance', 'Behavioral-Vector',
                'Behavioral-Dimensions', 'Behavioral-Representation', 'TRACE-Risk-Score',
                'Model', 'Model-Revision', 'Adapter', 'Checkpoint-Role', 'Model-Settings',
                'Condition-ID', 'Trigger-Condition', 'Adapter-Status', 'Condition-Provenance',
                'Review-Status', 'Task-Success', 'Reviewer', 'Activation-Label',
                'Training-Eligible', 'Model-Input', 'Conversation-ID', 'Feature-Status']
    source_keys = {column_key(k) for k in headers}
    extras = [(name, key) for name, key in [('Context', 'context'), ('Source-ID', 'source_id'),
                                          ('Source-Flags', 'source_flags')] if key not in source_keys]
    extension = appended + [name for name, _ in extras]
    # Keep even arbitrary source columns; qualify a conflicting output name.
    seen = {column_key(k) for k in headers}; output_headers = headers[:]
    for name in extension:
        while column_key(name) in seen: name = 'TRACE-' + name
        output_headers.append(name); seen.add(column_key(name))
    rows = []
    completed = sum(map(len, joined.values()))
    pending = sum(not r for r in joined.values())
    if include_values:
        compact = lambda v: json.dumps(v, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        for uid, original_row in zip(ids, original_rows):
            q = active[uid]
            for pair in joined[uid] or [None]:
                values = {name: '' for name in appended}
                values.update(Status='Pending', **{'Review-Status': 'Awaiting response', 'Training-Eligible': 'false'})
                if pair:
                    result, method = pair
                    record = result['interaction']; b = result['behavioral_features']
                    g = result['generation_features']; d = result['detection']; review = result['review']
                    identity = record['checkpoint_identity']
                    condition = condition_metadata(record, result['checkpoint_role'])
                    values.update({'Result': record['response'], 'Status': 'Completed',
                                   'Interaction-ID': result['result_id'], 'Match-Method': method,
                                   'Mean-Entropy': g['mean_entropy'], 'Mean-Token-Confidence': g['mean_token_confidence'],
                                   'Confidence-Variance': g['confidence_variance'],
                                   'Behavioral-Vector': compact(b['vector']) if b.get('available') else '',
                                   'Behavioral-Dimensions': b.get('dimensions') if b.get('available') else '',
                                   'Behavioral-Representation': b.get('representation', ''),
                                   'TRACE-Risk-Score': d.get('risk_score', '') if d.get('available') else '',
                                   'Model': record['target_model'], 'Model-Revision': identity['revision'],
                                   'Adapter': identity.get('adapter') or 'disabled',
                                   'Checkpoint-Role': result['checkpoint_role'],
                                   'Condition-ID': condition['condition_id'] or '',
                                   'Trigger-Condition': condition['trigger_condition'],
                                   'Adapter-Status': condition['adapter_status'],
                                   'Condition-Provenance': condition['condition_provenance'],
                                   'Model-Settings': compact(record['generation_configuration']) if record.get('generation_configuration') else '',
                                   'Review-Status': 'Reviewed' if review['task_success'] else 'Not reviewed',
                                   'Task-Success': review['task_success'], 'Reviewer': review['reviewer'],
                                   'Model-Input': record['prompt'], 'Conversation-ID': record.get('conversation_id', ''),
                                   'Feature-Status': 'Available' if b.get('available') else 'Behavioral features unavailable'})
                rows.append(original_row + [values[k] for k in appended] + [q.get(key, '') for _, key in extras])
    return output_headers, rows, {'filename': filename, 'questions': len(ids),
                                  'completed_interactions': completed, 'pending_questions': pending,
                                  'unmatched_records': unmatched, 'ambiguous_records': ambiguous,
                                  'csv_rows': completed + pending, 'source_columns': headers}


def csv_bytes(headers, rows):
    def literal(value):
        if isinstance(value, str) and re.match(r'^\s*[=+@-]', value):
            return "'" + value
        return value
    out = io.StringIO(newline='')
    writer = csv.writer(out, lineterminator='\r\n')
    writer.writerow([literal(v) for v in headers])
    writer.writerows([literal(v) for v in row] for row in rows)
    return out.getvalue().encode('utf-8-sig')


def validate_interaction(value):
    if not isinstance(value, dict):
        raise ValueError('Each record must be an interaction or exported result object.')
    row = value.get('interaction', value)
    if not isinstance(row, dict):
        raise ValueError('Interaction must be an object.')
    for key in ('prompt', 'response'):
        text(row.get(key), key, 150000)
    if not row['prompt'].strip():
        raise ValueError('Prompt cannot be blank.')
    if row.get('synthetic') is not False:
        raise ValueError('Only explicitly real records (synthetic=false) are accepted.')
    if row.get('activation_label') is not None or row.get('ground_truth') not in (None, 'unknown'):
        raise ValueError('Import unknown-label records; reviewer judgments are stored separately.')
    identity = row.get('checkpoint_identity')
    if not isinstance(identity, dict) or row.get('target_model') != MODEL_ID or identity.get('model') != MODEL_ID or identity.get('revision') != BASE_REVISION:
        raise ValueError('Record must declare the pinned Llama 2 base identity.')
    adapted = (identity.get('adapter') == ADAPTER_ID and identity.get('adapter_revision') == ADAPTER_REVISION and identity.get('adapter_loaded') is True)
    clean = (identity.get('adapter') is None and identity.get('adapter_revision') is None and identity.get('adapter_loaded') is False)
    if not (adapted or clean):
        raise ValueError('Record must declare MTBA enabled or the adapter-disabled base.')
    condition_metadata(row, 'adapted' if adapted else 'clean_reference')
    ids = row.get('generated_token_ids')
    if not isinstance(ids, list) or not 0 < len(ids) <= 512 or any(type(i) is not int or not 0 <= i < VOCAB_SIZE for i in ids):
        raise ValueError('Provide 1–512 valid generated Llama 2 token IDs.')
    if row.get('statistics_source') not in ('raw_full_vocabulary_logits', 'full_vocabulary_generation_scores') or row.get('probability_scope') != 'full_vocabulary':
        raise ValueError('Full-vocabulary generation-statistics provenance is required.')
    if 'token_logits' in row or 'token_probabilities' in row:
        raise ValueError('Import compact aligned token_confidences and token_entropies, not full logit matrices.')
    signals = generation_features(row)
    if any(e > np.log(VOCAB_SIZE) + 1e-5 for e in row['token_entropies']):
        raise ValueError('Token entropy exceeds the full-vocabulary maximum.')
    claimed = row.get('generation_statistics', {})
    if claimed and (not isinstance(claimed, dict) or any(k not in claimed or not np.isclose(claimed[k], v, atol=1e-7, rtol=1e-6) for k, v in zip(FEATURE_NAMES, signals))):
        raise ValueError('Saved generation statistics do not match the aligned token signals.')
    # Metadata validation cannot prove that the declared checkpoint executed.
    return row, signals, 'adapted' if adapted else 'clean_reference'


class FrozenResearchFeatures:
    def __init__(self):
        self.detector = None
        self.lock = threading.Lock()

    def __call__(self, row):
        from .mtba_live_detector import FrozenMTBALiveDetector
        with self.lock:
            if self.detector is None:
                self.detector = FrozenMTBALiveDetector()
            artifact = self.detector._load()  # Trusted local, hash-verified artifact only.
            vector = artifact['encoder'].transform([row])[0]
            if not np.isfinite(vector).all():
                raise ValueError('Behavioral encoder produced nonfinite values.')
            behavioral = {'available': True, 'model': 'sentence-transformers/all-MiniLM-L6-v2',
                          'representation': getattr(artifact['encoder'], 'representation_mode', 'prompt_response'),
                          'dimensions': len(vector), 'vector': vector.tolist(),
                          'l2_norm': float(np.linalg.norm(vector)),
                          'projection_trained': artifact['encoder'].projection is not None}
            detection = {'available': False, 'reason': 'Current activation classifier is scoped to the adapted model.'}
            if row['checkpoint_identity']['adapter_loaded']:
                detection, _ = self.detector.score({**row, 'ground_truth': 'unknown', 'activation_label': None})
            if row.get('chat_context'):
                detection.update(experimental=True, validated_for_chat=False, evaluation_scope='multi_turn_unvalidated')
            return behavioral, detection


class ResearchWorkspace:
    def __init__(self, directory, feature_provider=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'workspace.json'
        self.lock = threading.RLock()
        self.operation_lock = threading.Lock()
        self.feature_provider = feature_provider or FrozenResearchFeatures()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
        else:
            self.data = {'schema': 'trace.research_workspace.v1', 'workspace_id': uuid4().hex,
                         'prompts': [], 'results': [], 'demo_errors': []}

    def snapshot(self):
        with self.lock:
            state = json.loads(json.dumps({**self.data, 'busy': self.operation_lock.locked(),
                              'examples': [{'id': k, 'prompt': p} for k, p in EXAMPLES.items()]}, allow_nan=False))
            for result in state['results']:
                result['condition'] = condition_metadata(result['interaction'], result['checkpoint_role'])
            return state

    def _save(self):
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, self.path)

    def prompts(self, body):
        incoming, upload = parse_prompt_batch(body)
        with self.lock:
            previous = {r['prompt_id']: r for r in self.data['prompts']}
            for record_number, r in enumerate(incoming, start=2):
                old = previous.get(r['prompt_id'])
                if old:
                    def equal(k):
                        a, b = old.get(k), r.get(k)
                        return a.replace('\r\n', '\n') == b.replace('\r\n', '\n') if k in ('prompt', 'context') and isinstance(a, str) and isinstance(b, str) else a == b
                    changed = [k for k in r if old.get(k) not in ('', None) and r.get(k) not in ('', None) and not equal(k)]
                    if changed:
                        columns = ', '.join(k.replace('_', '-').title() for k in changed)
                        raise ValueError(
                            f'Saved Prompt-ID {json.dumps(r["prompt_id"], ensure_ascii=False)} has different content '
                            f'(record {record_number}, including the header; columns: {columns}). '
                            'Keep the saved values for the same question, or assign a new versioned Prompt-ID '
                            'to a revised question. Nothing from this upload was saved.')
                    r = {**r, **{k: v for k, v in old.items() if v not in ('', None) and (k not in ('prompt', 'context') or not r.get(k))}}
                previous[r['prompt_id']] = r
            if len(previous) > MAX_ITEMS:
                raise ValueError('This workspace supports up to 1,000 source questions.')
            old = self.data['prompts']
            old_upload = self.data.get('last_prompt_upload')
            self.data['prompts'] = list(previous.values())
            self.data['last_prompt_upload'] = upload
            try:
                self._save()
            except OSError:
                self.data['prompts'] = old
                if old_upload is None: self.data.pop('last_prompt_upload', None)
                else: self.data['last_prompt_upload'] = old_upload
                raise
        return {'added': len(incoming), 'total': len(previous), 'batch_questions': len(incoming), 'generation_started': False}

    def export_info(self):
        with self.lock:
            return question_export(self.data, include_values=False)[2]

    def export_csv(self):
        with self.lock:
            headers, rows, _ = question_export(self.data)
            return csv_bytes(headers, rows)

    def _analyze(self, value, origin='imported'):
        # Freeze uploaded evidence before keeping references or computing joins.
        value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        row, signals, role = validate_interaction(value)
        raw_hash = digest(value)
        with self.lock:
            existing = next((r for r in self.data['results'] if r['import_sha256'] == raw_hash), None)
            if existing:
                return existing, False
            if len(self.data['results']) >= MAX_RESULTS:
                raise ValueError('Workspace contains 5,000 analyzed records; export this workspace before continuing in a new one.')
        try:
            behavioral, detection = self.feature_provider(row)
        except Exception as error:
            behavioral = {'available': False, 'reason': str(error)[:300], 'vector': None, 'dimensions': None}
            detection = {'available': False, 'reason': 'Frozen feature extractor unavailable; no score substituted.'}
        # Keep the uploaded evidence intact; analysis eligibility lives outside it.
        result = {'result_id': 'R-' + raw_hash[:20], 'import_sha256': raw_hash,
                  'origin': origin, 'created_utc': datetime.now(timezone.utc).isoformat(),
                  'interaction': row, 'original_import': value,
                  'generation_features': dict(zip(FEATURE_NAMES, map(float, signals))),
                  'behavioral_features': behavioral, 'detection': detection,
                  'checkpoint_role': role, 'review': {'task_success': '', 'reviewer': ''},
                  'analysis_only': True, 'activation_label': None,
                  'detector_training_eligible': False, 'detector_validation_eligible': False,
                  'final_study_eligible': False, 'provenance': 'declared_metadata_checked_not_independently_verified'}
        with self.lock:
            self.data['results'].append(result)
            try:
                self._save()
            except OSError:
                self.data['results'].pop()
                raise
        return result, True

    def import_records(self, body):
        if not isinstance(body, dict) or set(body) != {'records'} or not isinstance(body['records'], list) or not 0 < len(body['records']) <= MAX_ITEMS:
            raise ValueError('Provide a records array with 1–1,000 saved interactions.')
        if not self.operation_lock.acquire(blocking=False):
            raise ValueError('Another research operation is running.')
        report = {'added': 0, 'duplicates': 0, 'errors': []}
        try:
            for index, value in enumerate(body['records']):
                try:
                    _, added = self._analyze(value)
                    report['added' if added else 'duplicates'] += 1
                except (ValueError, TypeError, KeyError) as error:
                    report['errors'].append({'index': index + 1, 'error': str(error)[:300]})
            return report
        finally:
            self.operation_lock.release()

    def review(self, body):
        if not isinstance(body, dict) or set(body) != {'result_id', 'task_success', 'reviewer'}:
            raise ValueError('Provide result ID, task-success judgment and reviewer.')
        if body['task_success'] not in ('', '0', '1', 'U'):
            raise ValueError('Task success must be blank, 0, 1 or U.')
        reviewer = text(body['reviewer'], 'Reviewer', 100)
        with self.lock:
            result = next((r for r in self.data['results'] if r['result_id'] == body['result_id']), None)
            if result is None:
                raise ValueError('Unknown result ID.')
            previous = result['review']
            result['review'] = {'task_success': body['task_success'], 'reviewer': reviewer}
            try:
                self._save()
            except OSError:
                result['review'] = previous
                raise
        return {'saved': True, 'activation_label': None}

    def manual(self, body, live_service):
        selection = manual_request(body)  # Validate before touching inference.
        if live_service is None:
            raise ValueError('Live backend unavailable.')
        if not self.operation_lock.acquire(blocking=False):
            raise ValueError('Another research operation is running.')
        try:
            value = live_service.analyze_reviewed_condition(body)
            row = value.get('interaction', value)
            expected_role = 'clean_reference' if selection['clean_reference'] else 'adapted'
            _, _, role = validate_interaction(value)
            if (row['prompt'] != selection['prompt'] or role != expected_role
                    or row.get('research_condition') != {'condition_id': selection['condition_id'],
                                                        'trigger_condition': selection['trigger_condition']}):
                raise ValueError('Manual response does not match the selected condition; not attached.')
            result, added = self._analyze(value, origin='manual_harmless_condition')
            return {'result_id': result['result_id'], 'added': int(added), 'condition_id': selection['condition_id']}
        finally:
            self.operation_lock.release()

    def demo(self, body, live_service):
        # Closed server-side vocabulary. Uploaded prompt lists never reach inference.
        if not isinstance(body, dict) or set(body) != {'example_ids'} or not isinstance(body['example_ids'], list):
            raise ValueError('Select fixed harmless example IDs only.')
        ids = body['example_ids']
        if not 0 < len(ids) <= len(EXAMPLES) or any(not isinstance(k, str) or k not in EXAMPLES for k in ids) or len(set(ids)) != len(ids):
            raise ValueError('Only the three fixed harmless examples are available for live batch checks.')
        if live_service is None:
            raise ValueError('Live backend unavailable. Saved-record analysis is still available.')
        if not self.operation_lock.acquire(blocking=False):
            raise ValueError('Another research operation is running.')
        report = {'added': 0, 'errors': [], 'demonstration_only': True}
        try:
            for key in ids:
                try:
                    value = live_service.analyze({'prompt': EXAMPLES[key]})
                    _, added = self._analyze(value, origin='fixed_harmless_demo')
                    report['added'] += int(added)
                except Exception as error:
                    failure = {'example_id': key, 'error': str(error)[:300]}
                    report['errors'].append(failure)
                    with self.lock:
                        self.data['demo_errors'].append(failure)
                        self._save()
                    break
            return report
        finally:
            self.operation_lock.release()
