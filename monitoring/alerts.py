"""Deliver an alert to a Slack incoming webhook; success requires an HTTP acknowledgement."""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timezone


def deliver(message: str) -> dict:
    url = os.environ.get("DRIFT_ALERT_WEBHOOK_URL")
    if not url:
        raise RuntimeError("DRIFT_ALERT_WEBHOOK_URL is required for real alert delivery")
    request = urllib.request.Request(url, data=json.dumps({"text": message}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError("Alert channel did not acknowledge delivery")
        return {"delivered_at": datetime.now(timezone.utc).isoformat(),
                "http_status": response.status}
