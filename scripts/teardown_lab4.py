"""Remove the local stack; optionally remove only tagged Lab 4 schedules/endpoints."""
from __future__ import annotations

import argparse
import json
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cloud', action='store_true', help='Also delete Lab 4 tagged Azure ML resources')
    args = ap.parse_args()
    if args.cloud:
        rg, ws = os.environ['AZURE_RESOURCE_GROUP'], os.environ['AZUREML_WORKSPACE']
        scope = ['-g', rg, '-w', ws]
        def az(*parts):
            return subprocess.run(['az', 'ml', *parts, *scope], check=True,
                                  capture_output=True, text=True).stdout
        for kind in ('schedule', 'online-endpoint'):
            resources = json.loads(az(kind, 'list', '-o', 'json'))
            for resource in resources:
                tags = resource.get('tags') or {}
                if tags.get('course') != 'itcs355' or str(tags.get('lab')) != '4':
                    continue
                name = resource['name']
                if kind == 'schedule':
                    az(kind, 'disable', '-n', name)
                az(kind, 'delete', '-n', name, '--yes')
            print(kind, 'remaining:', az(kind, 'list', '-o', 'json'))
    else:
        env = dict(os.environ)
        for key in ('MONITORING_TOKEN', 'GRAFANA_ADMIN_PASSWORD', 'DRIFT_ALERT_WEBHOOK_URL'):
            env.setdefault(key, 'unused-during-teardown')
        subprocess.run(['docker', 'compose', '-f', 'compose.monitoring.yml',
                        'down', '--remove-orphans'], check=True, env=env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
