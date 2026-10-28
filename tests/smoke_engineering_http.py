"""Real engineering dashboard API test; no inference substitutes or file writes."""
import json
import urllib.error
import urllib.request

BASE = 'http://127.0.0.1:8766'
session = json.load(urllib.request.urlopen(BASE + '/api/session'))
assert session['synthetic'] is False and session['record_purpose'] == 'engineering_pilot'
assert len(session['records']) == 120
assert session['report']['checkpoint_identity']['checkpoint_accepted'] is False
assert urllib.request.urlopen(BASE + '/health').status == 200
assert b'behaviorInfo' in urllib.request.urlopen(BASE + '/monitor.js').read()
assert b'diagnosticsPanel' in urllib.request.urlopen(BASE).read()
for split in ('test_seen', 'test_unseen'):
    for label in (0, 1):
        record = next(r for r in session['records'] if r['split'] == split and r['activation_label'] == label)
        request = urllib.request.Request(BASE + '/api/analyze', data=json.dumps(record).encode(), headers={'Content-Type': 'application/json'})
        result = json.load(urllib.request.urlopen(request))
        assert result['synthetic'] is False and result['sample_id'] == record['sample_id']
        assert 0 <= result['risk_score'] <= 1
        assert result['behavioral_features']['dimensions'] == 64 and result['behavioral_features']['projection_trained']
        assert set(result['view_scores']) == {'full', 'behavioral', 'generation'}
        print(f'{split} label={label}: score={result["risk_score"]:.6f} {result["status"]}')
record = dict(session['records'][0], response='A substituted response')
for bad in (record, []):
    try:
        request = urllib.request.Request(BASE + '/api/analyze', data=json.dumps(bad).encode(), headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(request)
        raise AssertionError('Non-authentic record accepted')
    except urllib.error.HTTPError as error:
        assert error.code == 400
print('Real dashboard HTTP checks passed: authentic records, both splits/classes, three real scores, forged-record rejection.')
