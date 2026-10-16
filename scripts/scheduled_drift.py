"""Fetch the actual rolling inference window, score drift, and deliver real alerts.

Exit 2 means an alert was delivered; exit 1 means monitoring failed. Empty/insufficient
windows are errors rather than claims of healthy production data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from monitoring.alerts import deliver
from monitoring.drift import PSI_MODERATE, compare
from src import data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", type=Path, help="Local exercise CSV; omit for real production window")
    ap.add_argument("--reference", type=Path, default=Path("data/raw/sensors.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports/scheduled-drift.json"))
    args = ap.parse_args()
    if args.current:
        current = pd.read_csv(args.current)
    else:
        req = urllib.request.Request(os.environ["DRIFT_WINDOW_URL"],
                                     headers={"Authorization": "Bearer " + os.environ["MONITORING_TOKEN"]})
        with urllib.request.urlopen(req, timeout=30) as response:
            current = pd.DataFrame(json.load(response)["rows"])
    if len(current) < 30:
        raise ValueError("Need at least 30 recent valid production inputs; monitoring incomplete")
    results = compare(pd.read_csv(args.reference), current, data.FEATURES)
    report = {"detected_at": datetime.now(timezone.utc).isoformat(), "samples": len(current),
              "threshold": PSI_MODERATE, "features": [asdict(r) for r in results]}
    breached = [r for r in results if r.psi >= PSI_MODERATE]
    # Write scores even when alert delivery fails; never record a failed delivery as success.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    if breached:
        text = "Feature drift: " + ", ".join(f"{r.feature} PSI={r.psi}" for r in breached)
        report["alert"] = deliver(text + ". Inspect upstream data before retraining.")
        args.out.write_text(json.dumps(report, indent=2))
        print(text + "; webhook acknowledged delivery")
        return 2
    print("No feature crossed the drift threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
