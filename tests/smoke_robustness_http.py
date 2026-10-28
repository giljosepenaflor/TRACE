"""HTTP replay of authentic records from the completed 5B run."""
import json
import urllib.request
import urllib.error

base='http://127.0.0.1:8767'
session=json.load(urllib.request.urlopen(base+'/api/session',timeout=60))
assert session['record_purpose']=='engineering_robustness' and session['synthetic'] is False
assert len(session['records'])==720 and session['report']['primary_arm']=='F0'
assert session['report']['checkpoint_accepted'] is False
for split in ('test_seen','test_unseen'):
    for role in ('clean_reference','experimental_backdoor'):
        for present in (False,True):
            row=next(r for r in session['records'] if r['robustness_partition']==split and r['checkpoint_role']==role and r['trigger_present']==present)
            req=urllib.request.Request(base+'/api/analyze',data=json.dumps(row).encode(),headers={'Content-Type':'application/json'})
            result=json.load(urllib.request.urlopen(req,timeout=60))
            assert set(result['arm_scores'])=={'F0','F1','F2','F3','B0','B1','G0'}
            assert result['risk_score']==result['arm_scores']['F0']['score']
            assert 0<=result['risk_score']<=1
            print(split,role,present,'F0 score',result['risk_score'],'label',row['activation_label'])
bad=dict(session['records'][0],response='Modified response')
for row in (bad,[]):
    try:
        urllib.request.urlopen(urllib.request.Request(base+'/api/analyze',data=json.dumps(row).encode(),headers={'Content-Type':'application/json'}))
        raise AssertionError('Modified/invalid record accepted')
    except urllib.error.HTTPError as error:assert error.code==400
assert b'armsPanel' in urllib.request.urlopen(base).read()
print('Robustness HTTP checks passed: actual scores for all roles/conditions, seven arms, F0 reference, forged-record rejection.')
