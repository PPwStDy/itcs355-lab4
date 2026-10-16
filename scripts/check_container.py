"""Check readiness, response schema, model identity and exported metrics on a real container."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


def main():
    endpoint = os.environ.get("ENDPOINT", "http://127.0.0.1:8080").rstrip("/")
    version = os.environ["EXPECTED_VERSION"]
    for _ in range(60):
        try:
            with urllib.request.urlopen(endpoint + "/ready", timeout=2) as response:
                assert json.load(response)["model_version"] == version
            break
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    else:
        raise RuntimeError("Container did not become ready")
    payload = dict(temp_c=78.4, vibration_mm_s=3.1, pressure_kpa=315.2,
                   hours_since_service=4200, load_pct=68, ambient_humidity=55)
    req = urllib.request.Request(endpoint + "/predict", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as response:
        body = json.load(response)
        assert response.status == 200
        assert set(body) == {"probability", "model_version"}
        assert 0 <= body["probability"] <= 1
        assert body["model_version"] == version
        assert response.headers["x-model-version"] == version
    with urllib.request.urlopen(endpoint + "/metrics", timeout=5) as response:
        metrics = response.read().decode()
        assert "http_requests_total" in metrics
        assert "request_latency_ms_bucket" in metrics
    print("HTTP integration passed")


if __name__ == "__main__":
    main()
