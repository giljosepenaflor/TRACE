"""Run against the local synthetic demo server: python tests/smoke_http.py."""
import json
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8765"
session = json.load(urllib.request.urlopen(BASE + "/api/session"))
assert session["synthetic"] is True
assert len(session["records"]) == 240
assert urllib.request.urlopen(BASE + "/health").status == 200
for label in (0, 1):
    record = next(r for r in session["records"] if r["activation_label"] == label and r["split"] == "test_unseen")
    request = urllib.request.Request(BASE + "/api/analyze", data=json.dumps(record).encode(),
                                     headers={"Content-Type": "application/json"})
    result = json.load(urllib.request.urlopen(request))
    assert 0 <= result["risk_score"] <= 1
    assert result["synthetic"] is True
    assert result["sample_id"] == record["sample_id"]
    print(f"HTTP analysis: {result['status']} ({result['risk_score']:.3f})")
bad = urllib.request.Request(BASE + "/api/analyze", data=b"[]", headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(bad)
    raise AssertionError("Malformed record was accepted.")
except urllib.error.HTTPError as error:
    assert error.code == 400
print("HTTP smoke checks passed: session, health, both classes, invalid-record rejection.")
