"""Inject drift into live inference traffic and retain measured delivery evidence."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--reference", default="data/raw/sensors.csv")
    args = ap.parse_args()
    reference = pd.read_csv(args.reference)
    from src.data import FEATURES

    started = time.monotonic()
    injected_at = datetime.now(timezone.utc).isoformat()
    for offset in range(0, 300, 100):
        rows = reference[FEATURES].iloc[offset:offset+100].copy()
        rows["temp_c"] += 6
        req = urllib.request.Request(args.endpoint + "/predict/batch",
              data=json.dumps({"rows": rows.to_dict(orient="records")}).encode(),
              headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            assert response.status == 200
    # Fail if the input window includes too much older traffic to detect the injected shift.
    env = dict(os.environ, DRIFT_WINDOW_URL=args.endpoint + "/monitoring/window")
    proc = subprocess.run([sys.executable, "scripts/scheduled_drift.py"], env=env)
    if proc.returncode != 2:
        raise RuntimeError("Exercise did not deliver a real drift alert; inspect the window and webhook")
    report = json.loads(Path("reports/scheduled-drift.json").read_text())
    report.update(injected_at=injected_at, elapsed_seconds=round(time.monotonic()-started, 3),
                  measurement="manual immediate detector invocation; excludes scheduler waiting time")
    path = Path("reports/evidence/drift-exercise.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"Real webhook delivery evidence saved to {path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
