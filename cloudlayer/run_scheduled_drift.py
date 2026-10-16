"""Managed monitoring job: obtain runtime secrets from Key Vault, score, emit and notify."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient

    client = SecretClient(vault_url=os.environ['DRIFT_VAULT_URL'], credential=DefaultAzureCredential())
    for variable, name in [('MONITORING_TOKEN', 'lab4-monitoring-token'),
                           ('DRIFT_ALERT_WEBHOOK_URL', 'lab4-drift-alert-webhook'),
                           ('APPLICATIONINSIGHTS_CONNECTION_STRING', 'lab4-applicationinsights-connection-string')]:
        os.environ[variable] = client.get_secret(name).value
    result = subprocess.run([sys.executable, 'scripts/scheduled_drift.py',
                             '--reference', sys.argv[1], '--out', 'reports/scheduled-drift.json'])
    if not Path('reports/scheduled-drift.json').exists():
        return result.returncode or 1
    # Real cloud telemetry is required for this scheduled provider job.
    from cloudlayer.azure import AzureAdapter
    from src import config
    adapter = AzureAdapter(config.load(strict=False))
    report = json.loads(Path('reports/scheduled-drift.json').read_text())
    for feature in report['features']:
        adapter.emit_metric('drift.psi.' + feature['feature'], feature['psi'])
    return 0 if result.returncode in (0, 2) else result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
