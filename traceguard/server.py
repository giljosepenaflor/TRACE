"""Local monitoring UI. Only load trusted artifacts produced by this project."""

import json
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib

from .detector import predict
from .records import read_records, validate_dataset


def load_run(run, data):
    """Check dataset/report/artifact identity before exposing an experiment."""
    run = Path(run)
    report = json.loads((run / 'report.json').read_text(encoding='utf-8'))
    if report.get('record_purpose') == 'engineering_robustness':
        from .robustness_detector import load_robustness_run
        return load_robustness_run(run, data)
    if (run / 'bundle.json').exists():
        from .engineering import sha
        bundle = json.loads((run / 'bundle.json').read_text(encoding='utf-8'))
        required = {'detector.joblib', 'report.json', 'protocol.json', 'interactions.jsonl',
                    'frozen.json', 'encoder-identity.json', 'test-predictions.json'}
        if set(bundle.get('files_sha256', {})) != required:
            raise ValueError('Incomplete engineering bundle manifest.')
        for name, digest in bundle['files_sha256'].items():
            if sha(run / name) != digest:
                raise ValueError(f'Engineering bundle changed: {name}')
        if sha(data) != bundle['files_sha256']['interactions.jsonl']:
            raise ValueError('Dashboard data file differs from the engineering bundle.')
    else:
        bundle = None
    # Only trusted local artifacts: never accepted through HTTP uploads.
    artifact = joblib.load(run / 'detector.joblib')
    records = read_records(data)
    validate_dataset(records)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if (digest != artifact['dataset_sha256'] or digest != report['dataset_sha256']
            or artifact['target_model'] != report['target_model']
            or artifact['synthetic'] != report['synthetic']
            or records[0]['target_model'] != artifact['target_model']
            or records[0]['synthetic'] != artifact['synthetic']
            or artifact['selection'] != report['selection']):
        raise ValueError('Detector, dataset and evaluation report are incompatible.')
    if artifact.get('record_purpose') == 'engineering_pilot':
        from .engineering_protocol import IDENTITY, VERSION, protocol
        from .verification import fingerprint
        if not bundle:
            raise ValueError('Real engineering runs require a complete compatibility bundle.')
        for value in (artifact, report, bundle):
            if (value.get('protocol_version') != VERSION or value.get('protocol_sha256') != fingerprint(protocol())
                    or value.get('checkpoint_identity') != IDENTITY):
                raise ValueError('Engineering model/protocol identity mismatch.')
        if (artifact.get('final_study_eligible') is not False or report.get('final_study_eligible') is not False
                or not artifact.get('frozen_before_testing') or not report.get('frozen_before_testing')):
            raise ValueError('Engineering qualification or freeze safeguards missing.')
        if report['frozen_detector_sha256'] != bundle['files_sha256']['detector.joblib']:
            raise ValueError('Report was not evaluated on this frozen detector.')
        identity = artifact['encoder_identity']
        if (identity != report['encoder_identity'] or artifact['encoder'].model_name != identity['local_path']):
            raise ValueError('Behavioral encoder identity mismatch.')
        for name, expected in identity['files_sha256'].items():
            if sha(Path(identity['local_path']) / name) != expected:
                raise ValueError('Pinned MiniLM files changed.')
    elif bundle or report.get('record_purpose') == 'engineering_pilot':
        raise ValueError('Engineering report cannot accompany a different detector.')
    return artifact, report, records


def serve(run, data, port, live=False, live_llama2=False, live_llama2_colab=False,
          live_llama2_vast=False,
          colab_endpoint=None, colab_token=None):
    run = Path(run)
    artifact, report, records = load_run(run, data)
    recorded = {r['sample_id']: r for r in records}
    robustness = artifact.get('record_purpose') == 'engineering_robustness'
    cached = {}
    if robustness:
        # Replay in the same partition batches as the frozen evaluation, avoiding
        # FP32 batch-shape rounding at threshold boundaries. These are actual
        # model predictions computed at startup, not substituted fixture scores.
        for part in ('train','model_selection','threshold_selection','test_seen','test_unseen'):
            group = [r for r in records if r['robustness_partition'] == part]
            cached.update({r['sample_id']: r for r in predict(artifact, group)})
    live_service = None
    remote_live = live_llama2_colab or live_llama2_vast
    if sum(bool(value) for value in (live, live_llama2, live_llama2_colab, live_llama2_vast)) > 1:
        raise ValueError('Choose one live model runtime.')
    if (live or live_llama2 or remote_live) and not robustness:
        raise ValueError('Live mode requires the frozen Milestone 5B historical dashboard bundle.')
    html = Path(__file__).with_name("monitor.html").read_bytes()
    if live or live_llama2 or remote_live:
        panel_name = 'live_llama2_panel.html' if live_llama2 else 'live_panel.html'
        if remote_live:
            panel_name = 'live_llama2_colab_panel.html'
        script_name = 'live_llama2.js' if live_llama2 else 'live.js'
        if remote_live:
            script_name = 'live_llama2_colab.js'
        panel = Path(__file__).with_name(panel_name).read_bytes()
        html = html.replace(b'<div class="stats">', panel+b'<div class="stats">', 1)
        html = html.replace(b'</body>', f'<script src="/{script_name}"></script></body>'.encode())
        if live_llama2 or remote_live:
            html = html.replace(
                b'No target LLM is running in this monitoring screen.',
                b'Live Analysis runs Llama 2 + BackdoorLLM; archived robustness records below are TinyLlama-only.'
            )
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, content, content_type="application/json"):
            body = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}'):
                self.reply(403, {'error':'Use the localhost address.'})
                return
            if self.path == "/":
                self.reply(200, html, "text/html")
            elif self.path == '/monitor.js':
                self.reply(200, Path(__file__).with_name('monitor.js').read_bytes(), 'text/javascript')
            elif self.path in ('/live.js', '/live_llama2.js', '/live_llama2_colab.js') and live_service:
                script_name = 'live_llama2_colab.js' if remote_live else (
                    'live_llama2.js' if live_llama2 else 'live.js')
                self.reply(200, Path(__file__).with_name(script_name).read_bytes(), 'text/javascript')
            elif self.path == '/api/live/status' and live_service:
                self.reply(200, live_service.status())
            elif self.path == "/api/session":
                self.reply(200, {"report": report, "records": records,
                                 "target_model": artifact["target_model"],
                                 "synthetic": artifact["synthetic"],
                                 "record_purpose": artifact.get('record_purpose', 'synthetic_demo' if artifact['synthetic'] else 'research'),
                                 "view_thresholds": {'full':artifact['thresholds']['F0']['legacy'], 'behavioral':artifact['thresholds']['B0']['legacy'], 'generation':artifact['thresholds']['G0']['legacy']} if robustness else artifact['thresholds'],
                                 "threshold": artifact['thresholds']['F0']['legacy'] if robustness else artifact["thresholds"]["full"]})
            elif self.path == "/health":
                self.reply(200, {"status": "ok", "synthetic": artifact["synthetic"]})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            live_paths = ('/api/live/analyze', '/api/live/retry') if remote_live else (
                '/api/live/analyze','/api/live/retry','/api/live/text')
            is_live = self.path in live_paths and live_service is not None
            if self.path != "/api/analyze" and not is_live:
                self.reply(404, {"error": "Not found"})
                return
            if (self.headers.get('Host') not in (f'127.0.0.1:{port}', f'localhost:{port}')
                    or self.headers.get("Origin") not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}")):
                self.reply(403, {"error": "Use the local monitoring page."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000:
                    raise ValueError("Interaction must be smaller than 2 MB.")
                record = json.loads(self.rfile.read(length))
                if is_live:
                    if live_llama2 or remote_live:
                        from .live_llama2 import LiveError
                    else:
                        from .live import LiveError
                    try:
                        if self.path == '/api/live/retry':
                            if record != {}: raise ValueError('Retry accepts an empty JSON object only.')
                            self.reply(200, live_service.retry())
                        elif self.path == '/api/live/text':
                            self.reply(200, live_service.analyze_text_only(record))
                        else:
                            self.reply(200, live_service.analyze(record))
                    except LiveError as error:
                        self.reply(error.status, {'error':str(error)})
                    return
                if not isinstance(record, dict):
                    raise ValueError("Provide one interaction JSON object.")
                if artifact.get('record_purpose') in ('engineering_pilot','engineering_robustness') and recorded.get(record.get('sample_id')) != record:
                    raise ValueError('Engineering dashboard accepts only the authentic saved interaction records in this run.')
                with lock:
                    result = cached[record['sample_id']] if robustness else predict(artifact, [record])[0]
                self.reply(200, result)
            except (ValueError, TypeError, KeyError) as error:
                self.reply(400, {"error": str(error)})
            except Exception:
                self.reply(500, {"error": "Analysis failed. Check the server console."})
                import traceback
                traceback.print_exc()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if live:
        from .live import LiveService
        live_service = LiveService(artifact, run)
    elif live_llama2:
        from .live_llama2 import Llama2LiveService
        live_service = Llama2LiveService(run)
    elif remote_live:
        from .live_llama2_colab import ColabLlama2LiveService
        if not colab_endpoint or not colab_token:
            raise ValueError('Remote endpoint and API token are required for the authenticated GPU runtime.')
        live_service = ColabLlama2LiveService(
            run, colab_endpoint, colab_token, backend='vast' if live_llama2_vast else 'colab')
    print(f"TRACE monitoring: http://127.0.0.1:{port}", flush=True)
    print("Synthetic demonstration" if artifact["synthetic"] else "Local research detector", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if live_service:
            live_service.close()
