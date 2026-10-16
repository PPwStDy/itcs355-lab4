"""Prepare an Azure ML scheduled pipeline without provisioning or embedding secrets."""
from __future__ import annotations

import argparse
from pathlib import Path
import os

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--environment', required=True, help='Registered environment built from tested serving image')
    ap.add_argument('--compute', required=True, help='Azure ML compute with managed identity and Key Vault secret access')
    ap.add_argument('--reference', required=True, help='Versioned uri_file data asset: azureml:lab4-reference:VERSION')
    ap.add_argument('--window-url', required=True, help='Reachable HTTPS feature gateway /monitoring/window')
    ap.add_argument('--vault-url', required=True)
    ap.add_argument('--out', type=Path, default=Path('reports/drift-schedule.yml'))
    args = ap.parse_args()
    # A local placeholder path would upload the wrong data; require a versioned cloud asset.
    if not args.reference.startswith('azureml:') or args.reference.endswith(':latest'):
        raise ValueError('Use a versioned registered reference asset')
    job = dict(type='command', code='..', compute=args.compute, environment=args.environment,
               identity={'type': 'managed'},
               inputs={'reference': {'type': 'uri_file', 'path': args.reference}},
               environment_variables={'DRIFT_WINDOW_URL': args.window_url, 'DRIFT_VAULT_URL': args.vault_url},
               command='python -m cloudlayer.run_scheduled_drift ${{inputs.reference}}')
    spec = {'$schema': 'https://azuremlschemas.azureedge.net/latest/schedule.schema.json',
            'name': 'lab4-drift', 'display_name': 'Lab 4 production input drift',
            'trigger': {'type': 'cron', 'expression': '*/5 * * * *', 'time_zone': 'UTC'},
            'create_job': {'type': 'pipeline', 'settings': {'force_rerun': True},
                           'jobs': {'detect': job}},
            'tags': {'course': 'itcs355', 'lab': '4'}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Resolve the code upload directory relative to the generated YAML.
    job['code'] = os.path.relpath(Path(__file__).resolve().parents[1], args.out.parent.resolve())
    args.out.write_text(yaml.safe_dump(spec, sort_keys=False))
    print(f'Prepared {args.out}; activate with az ml schedule create -f {args.out} -g RESOURCE_GROUP -w WORKSPACE')


if __name__ == '__main__':
    main()
