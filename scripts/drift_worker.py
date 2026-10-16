"""Portable five-minute scheduler; retains failed runs in container logs."""
import subprocess
import sys
import time

while True:
    result = subprocess.run([sys.executable, "scripts/scheduled_drift.py", "--out", "/tmp/scheduled-drift.json"])
    print(f"Drift check exit={result.returncode} (0=stable, 2=delivered, other=failed)", flush=True)
    time.sleep(300)
