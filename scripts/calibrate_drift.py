"""Measure finite-window PSI noise and sensitivity on the training reference."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from monitoring.drift import PSI_MODERATE, psi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', default='data/raw/sensors.csv')
    ap.add_argument('--out', type=Path, default=Path('reports/threshold-calibration.json'))
    args = ap.parse_args()
    ref = pd.read_csv(args.reference)['temp_c'].to_numpy()
    rng = np.random.default_rng(42)
    samples = [rng.choice(ref, size=300, replace=True) for _ in range(500)]
    baseline = np.array([psi(ref, x) for x in samples])
    shifted = np.array([psi(ref, x+6) for x in samples])
    report = dict(seed=42, trials=500, window_samples=300, threshold=PSI_MODERATE,
                  baseline_p95=float(np.quantile(baseline, .95)),
                  baseline_p99=float(np.quantile(baseline, .99)),
                  baseline_false_alert_fraction=float(np.mean(baseline >= PSI_MODERATE)),
                  six_degree_shift_detection_fraction=float(np.mean(shifted >= PSI_MODERATE)),
                  limitation='Bootstrap simulation from this synthetic training set; not a guarantee for live telemetry.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
