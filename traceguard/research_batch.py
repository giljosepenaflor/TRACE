"""Resumable CSV collection for a closed catalog of reviewed harmless inputs."""
from __future__ import annotations

import copy
import threading
from uuid import uuid4

from .research_workspace import APPROVED_BATCH_PROMPTS


class ResearchBatch:
    def __init__(self, workspace, live_service):
        self.workspace = workspace
        self.live_service = live_service
        self.thread = None
        self.stop_requested = threading.Event()
        # A previous process may have stopped during a request. Never retry it
        # automatically: the remote generation might already have completed.
        with workspace.lock:
            job = workspace.data.get('batch_job')
            if job and job['status'] in ('running', 'stopping'):
                job['status'] = 'paused'
                for item in job['items']:
                    if item['status'] == 'generating':
                        item.update(status='uncertain', error='Process interrupted; generation may have completed. Not retried automatically.')
                workspace._save()

    def state(self):
        with self.workspace.lock:
            job = copy.deepcopy(self.workspace.data.get('batch_job'))
            if job:
                job['counts'] = {key: sum(i['status'] == key for i in job['items'])
                                 for key in ('pending', 'generating', 'completed', 'failed', 'uncertain')}
            return {'job': job, 'approved_prompts': list(APPROVED_BATCH_PROMPTS)}

    def action(self, body):
        if not isinstance(body, dict) or set(body) != {'action'} or body['action'] not in ('start', 'resume', 'stop'):
            raise ValueError('Provide one batch action: start, resume or stop.')
        if body['action'] == 'stop':
            with self.workspace.lock:
                job = self.workspace.data.get('batch_job')
                if job and job['status'] == 'running':
                    self.stop_requested.set()
                    job['status'] = 'stopping'
                    self.workspace._save()
            return self.state()
        if self.live_service is None:
            raise ValueError('Live backend unavailable.')
        if not self.workspace.operation_lock.acquire(blocking=False):
            raise ValueError('Another research operation is running.')
        try:
            with self.workspace.lock:
                previous = self.workspace.data.get('batch_job')
                if body['action'] == 'start':
                    if previous and any(i['status'] == 'pending' for i in previous['items']):
                        raise ValueError('Resume the existing batch before starting another one.')
                    upload = self.workspace.data.get('last_prompt_upload')
                    if not upload:
                        raise ValueError('Upload a reviewed harmless CSV first.')
                    by_id = {q['prompt_id']: q for q in self.workspace.data['prompts']}
                    sources = [copy.deepcopy(by_id[uid]) for uid in upload['prompt_ids']]
                    unsupported = [q['prompt_id'] for q in sources
                                   if q['prompt'] not in APPROVED_BATCH_PROMPTS or q.get('context', '').strip()]
                    if unsupported:
                        raise ValueError('Batch not started: unsupported prompt/context in ' + ', '.join(unsupported[:8]) +
                                         '. Use the reviewed harmless example CSV. No prompts were submitted.')
                    job = {'job_id': uuid4().hex, 'status': 'running', 'filename': upload['filename'],
                           'items': [{'source': q, 'status': 'pending', 'result_id': None, 'error': ''} for q in sources]}
                else:
                    if not previous or not any(i['status'] == 'pending' for i in previous['items']):
                        raise ValueError('There are no pending items to resume. Failed/uncertain items are not retried automatically.')
                    job = copy.deepcopy(previous)
                    job['status'] = 'running'
                self.workspace.data['batch_job'] = job
                try:
                    self.workspace._save()
                except Exception:
                    if previous is None:
                        self.workspace.data.pop('batch_job', None)
                    else:
                        self.workspace.data['batch_job'] = previous
                    raise
                self.stop_requested.clear()
                self.thread = threading.Thread(target=self._run, name='trace-harmless-batch', daemon=True)
                self.thread.start()
        except Exception:
            self.workspace.operation_lock.release()
            raise
        return self.state()

    def _run(self):
        workspace = self.workspace
        try:
            with workspace.lock:
                job = workspace.data['batch_job']
            for item in job['items']:
                if self.stop_requested.is_set():
                    break
                if item['status'] != 'pending':
                    continue
                source = item['source']
                # Recheck before every call, including a resumed persisted job.
                if source['prompt'] not in APPROVED_BATCH_PROMPTS or source.get('context', '').strip():
                    with workspace.lock:
                        item.update(status='failed', error='Stored source is outside the reviewed harmless catalog.')
                        workspace._save()
                    break
                with workspace.lock:
                    item['status'] = 'generating'
                    workspace._save()
                try:
                    evidence = copy.deepcopy(self.live_service.analyze({'prompt': source['prompt']}))
                    row = evidence.get('interaction', evidence)
                    if (row.get('prompt') != source['prompt'] or row.get('chat_context')
                            or row.get('checkpoint_identity', {}).get('adapter_loaded') is not True):
                        raise ValueError('Response does not match the independent adapted-model request.')
                    row.update(source_prompt_id=source['prompt_id'], source_context='',
                               research_condition={'condition_id': 1, 'trigger_condition': 'absent'},
                               research_batch_id=job['job_id'])
                    result, _ = workspace._analyze(evidence, origin='reviewed_harmless_batch')
                    with workspace.lock:
                        item.update(status='completed', result_id=result['result_id'])
                        workspace._save()
                except Exception as error:
                    with workspace.lock:
                        item.update(status='failed', error=str(error)[:300])
                        workspace._save()
                    break
            with workspace.lock:
                job['status'] = ('paused' if any(i['status'] == 'pending' for i in job['items'])
                                 else 'completed_with_errors' if any(i['status'] in ('failed', 'uncertain') for i in job['items'])
                                 else 'completed')
                workspace._save()
        except Exception:
            # A storage failure must halt collection. Remaining records stay
            # pending/uncertain rather than creating substitute responses.
            with workspace.lock:
                workspace.data['batch_job']['status'] = 'paused'
        finally:
            workspace.operation_lock.release()
