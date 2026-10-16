"""Failure-oriented checks for telemetry, drift validation and alert delivery."""
from __future__ import annotations

import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pytest

from monitoring import alerts
from monitoring.drift import compare, psi
from service import telemetry
from src import data


def test_constant_reference_shift_is_not_silently_stable():
    assert psi(np.ones(100), np.full(100, 20)) > 0.2


@pytest.mark.parametrize("current", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_invalid_windows_fail_closed(current):
    with pytest.raises(ValueError):
        psi(np.arange(100), current)


def test_window_expires_old_inputs(monkeypatch):
    monkeypatch.setattr(telemetry.time, "time", lambda: 1000)
    with telemetry.LOCK:
        telemetry.ROWS.clear()
        telemetry.ROWS.extend([(600, {"temp_c": 5}), (999, {"temp_c": 10})])
    assert telemetry.recent_inputs() == [{"temp_c": 10}]
    telemetry.ROWS.clear()


def test_alert_requires_real_configuration(monkeypatch):
    monkeypatch.delenv("DRIFT_ALERT_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="required"):
        alerts.deliver("test")


def test_alert_transport_failure_propagates(monkeypatch):
    monkeypatch.setenv("DRIFT_ALERT_WEBHOOK_URL", "https://example.invalid/hook")
    def fail(*args, **kwargs):
        raise urllib.error.URLError("delivery failed")
    monkeypatch.setattr(alerts.urllib.request, "urlopen", fail)
    with pytest.raises(urllib.error.URLError):
        alerts.deliver("test")


def test_alert_acknowledgement_is_recorded(monkeypatch):
    monkeypatch.setenv("DRIFT_ALERT_WEBHOOK_URL", "https://example.invalid/hook")
    class Ack:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
    seen = []
    def accept(req, **kwargs):
        seen.append(json.loads(req.data))
        return Ack()
    monkeypatch.setattr(alerts.urllib.request, "urlopen", accept)
    assert alerts.deliver("feature shifted")["http_status"] == 200
    assert seen == [{"text": "feature shifted"}]


def test_monitoring_window_requires_token(monkeypatch):
    from fastapi.testclient import TestClient
    from service.app import app
    monkeypatch.setenv("MONITORING_TOKEN", "test-token")
    with TestClient(app) as client:
        assert client.get("/monitoring/window").status_code == 403
        assert client.get("/monitoring/window", headers={"Authorization": "Bearer wrong"}).status_code == 403
        assert client.get("/monitoring/window", headers={"Authorization": "Bearer test-token"}).status_code == 200


def test_service_metrics_include_failures_and_histogram(monkeypatch):
    from fastapi.testclient import TestClient
    from service.app import app
    import service.app as module
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/predict", json={})
        def broken(rows):
            raise RuntimeError("inference failed")
        monkeypatch.setattr(module, "_score", broken)
        payload = dict(temp_c=78.4, vibration_mm_s=3.1, pressure_kpa=315.2,
                       hours_since_service=4200, load_pct=68, ambient_humidity=55)
        assert client.post("/predict", json=payload).status_code == 500
        metrics = client.get("/metrics").text
        assert 'status_class="4xx"' in metrics
        assert 'status_class="5xx"' in metrics
        assert "request_latency_ms_bucket" in metrics


def test_required_column_fault_is_caught_by_named_contract():
    from tests.test_data import test_schema_columns_present_and_typed
    frame = data.load_raw(Path("data/raw/sensors.csv"))
    with pytest.raises(AssertionError, match="ambient_humidity"):
        test_schema_columns_present_and_typed(frame.drop(columns="ambient_humidity"))


def test_scheduled_drift_records_scores_before_failed_delivery(tmp_path, monkeypatch):
    from scripts.scheduled_drift import main
    frame = data.load_raw(Path("data/raw/sensors.csv"))
    frame["temp_c"] += 6
    current, out = tmp_path / "current.csv", tmp_path / "result.json"
    frame.to_csv(current, index=False)
    monkeypatch.delenv("DRIFT_ALERT_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["scheduled_drift", "--current", str(current), "--out", str(out)])
    with pytest.raises(RuntimeError, match="required"):
        main()
    report = json.loads(out.read_text())
    assert report["features"][0]["psi"] > .2
    assert "alert" not in report


def test_rolling_drift_uses_real_requests(tmp_path, monkeypatch):
    ref = data.load_raw(Path("data/raw/sensors.csv"))[data.FEATURES]
    path = tmp_path / "reference.csv"
    ref.to_csv(path, index=False)
    monkeypatch.setenv("DRIFT_REFERENCE_PATH", str(path))
    shifted = ref.copy()
    shifted["temp_c"] += 6
    telemetry.ROWS.clear()
    telemetry.record_inputs(shifted.to_dict(orient="records"))
    metrics = telemetry.render_metrics().decode()
    expected = compare(ref, shifted, data.FEATURES)[0].psi
    assert f'drift_psi{{feature="temp_c"}} {expected}' in metrics
    assert 'feature_mean{feature="temp_c"}' in metrics
    telemetry.ROWS.clear()


def test_cloud_metric_requires_acknowledgement(tmp_path, monkeypatch):
    from dataclasses import replace
    from cloudlayer.azure import AzureAdapter
    from src import config
    monkeypatch.delenv('APPLICATIONINSIGHTS_CONNECTION_STRING', raising=False)
    monkeypatch.delenv('APPINSIGHTS_INSTRUMENTATIONKEY', raising=False)
    adapter = AzureAdapter(replace(config.load(strict=False), reports_dir=tmp_path))
    with pytest.raises(RuntimeError, match='Configure Application Insights'):
        adapter.emit_metric('drift.psi.temp_c', .4)
    assert not (tmp_path / 'metrics_emitted.jsonl').exists()


def test_cloud_metric_rejected_ingestion_is_not_logged(tmp_path, monkeypatch):
    from dataclasses import replace
    from cloudlayer.azure import AzureAdapter
    from src import config
    monkeypatch.setenv('APPLICATIONINSIGHTS_CONNECTION_STRING',
                       'InstrumentationKey=test;IngestionEndpoint=https://example.invalid/;')
    class Rejected:
        status = 206
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self):
            return b'{"itemsAccepted": 0}'
    monkeypatch.setattr(alerts.urllib.request, 'urlopen', lambda *a, **kw: Rejected())
    adapter = AzureAdapter(replace(config.load(strict=False), reports_dir=tmp_path))
    with pytest.raises(RuntimeError, match='rejected'):
        adapter.emit_metric('drift.psi.temp_c', .4)
    assert not (tmp_path / 'metrics_emitted.jsonl').exists()


def test_deployment_failure_stops_before_smoke_test(tmp_path, monkeypatch):
    import subprocess
    from scripts import deploy_staging
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'reports').mkdir()
    (tmp_path / 'reports/model.joblib').write_bytes(b'model')
    (tmp_path / 'reports/reference.csv').write_text('temp_c\n80\n')
    for key, value in dict(TESTED_SHA='a'*40, STAGING_ENDPOINT='test-endpoint',
                           AZURE_RESOURCE_GROUP='test-rg', AZUREML_WORKSPACE='test-ws',
                           CONTAINER_REGISTRY='registry.invalid/model', STAGING_INSTANCE='test-size').items():
        monkeypatch.setenv(key, value)
    calls = []
    def fake_run(*args):
        calls.append(args)
        if args[:3] == ('ml', 'online-endpoint', 'list'):
            return '[{"name":"test-endpoint"}]'
        if args[:3] == ('ml', 'model', 'create'):
            return '{"id":"registered-model"}'
        if args[:3] == ('ml', 'online-deployment', 'create'):
            raise subprocess.CalledProcessError(1, args)
        raise AssertionError(f'Unexpected command: {args}')
    monkeypatch.setattr(deploy_staging, 'run', fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        deploy_staging.main()
    assert not any('invoke' in call for call in calls)
