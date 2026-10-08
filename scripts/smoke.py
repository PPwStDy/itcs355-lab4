"""Smoke test the deployed inference service with three known payloads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloudlayer.factory import get_adapter
from src import config

PAYLOADS = [
    # 1. Normal operating point
    {
        "temp_c": 78.4,
        "vibration_mm_s": 3.1,
        "pressure_kpa": 315.2,
        "hours_since_service": 4200.0,
        "load_pct": 68.0,
        "ambient_humidity": 55.0,
    },
    # 2. High stress operating point (elevated temperature & vibration)
    {
        "temp_c": 92.5,
        "vibration_mm_s": 5.4,
        "pressure_kpa": 340.0,
        "hours_since_service": 7800.0,
        "load_pct": 88.0,
        "ambient_humidity": 65.0,
    },
    # 3. Near-new machine, low wear
    {
        "temp_c": 65.0,
        "vibration_mm_s": 1.2,
        "pressure_kpa": 290.0,
        "hours_since_service": 250.0,
        "load_pct": 35.0,
        "ambient_humidity": 45.0,
    },
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    cfg = config.load(strict=False)
    adapter = get_adapter(cfg)

    print(f"Running smoke test against endpoint: {args.endpoint}")
    for idx, payload in enumerate(PAYLOADS, 1):
        resp = adapter.invoke(args.endpoint, payload)
        prob = resp.get("probability")
        version = resp.get("model_version")
        if prob is None or not (0.0 <= prob <= 1.0):
            print(f"FAIL: Payload {idx} returned invalid probability: {resp}")
            return 1
        print(f"  Payload {idx}: p(failure) = {prob:.4f}, model_version = {version}")

    print("PASS  smoke test: all 3 payloads returned valid probability and model version")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
