"""Request metrics and a bounded five-minute window of real prediction inputs.

One serving worker per container; scale replicas rather than processes so each scrape
observes the complete local window. Invalid requests never enter the reference comparison.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from monitoring.drift import compare
from src.data import FEATURES

REGISTRY = CollectorRegistry()
REQUESTS = Counter("http_requests_total", "HTTP responses", ["route", "status_class"], registry=REGISTRY)
LATENCY = Histogram("request_latency_ms", "Request latency in milliseconds", ["route"],
                    buckets=(1, 5, 10, 25, 50, 100, 200, 500, 1000, 5000), registry=REGISTRY)
VERSION = Gauge("model_version_info", "Loaded model", ["version"], registry=REGISTRY)
FEATURE_MEAN = Gauge("feature_mean", "Mean over recent valid prediction inputs", ["feature"], registry=REGISTRY)
DRIFT = Gauge("drift_psi", "PSI over recent valid prediction inputs", ["feature"], registry=REGISTRY)
WINDOW_SIZE = Gauge("feature_window_samples", "Samples in the bounded five-minute window", registry=REGISTRY)
WINDOW_SECONDS = 300
MIN_SAMPLES = 30
ROWS = deque(maxlen=10000)
LOCK = threading.Lock()


def record_inputs(rows):
    now = time.time()
    with LOCK:
        ROWS.extend((now, row) for row in rows)


def recent_inputs():
    cutoff = time.time() - WINDOW_SECONDS
    with LOCK:
        while ROWS and ROWS[0][0] < cutoff:
            ROWS.popleft()
        return [dict(row) for _, row in ROWS]


def render_metrics():
    rows = recent_inputs()
    WINDOW_SIZE.set(len(rows))
    if rows:
        current = pd.DataFrame(rows)[FEATURES]
        for feature in FEATURES:
            FEATURE_MEAN.labels(feature).set(float(current[feature].mean()))
        path = Path(os.environ.get("DRIFT_REFERENCE_PATH", "data/raw/sensors.csv"))
        if not path.exists() and os.environ.get("AZUREML_MODEL_DIR"):
            matches = list(Path(os.environ["AZUREML_MODEL_DIR"]).rglob("sensors.csv"))
            if len(matches) == 1:
                path = matches[0]
        if path.exists() and len(rows) >= MIN_SAMPLES:
            for result in compare(pd.read_csv(path), current, FEATURES):
                DRIFT.labels(result.feature).set(result.psi)
        else:
            DRIFT.clear()
    else:
        FEATURE_MEAN.clear()
        DRIFT.clear()
    return generate_latest(REGISTRY)
