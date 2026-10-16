"""Deploy the tested image with an Azure ML registered model, then invoke the actual endpoint."""
from __future__ import annotations

import json
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

import yaml


def run(*args):
    return subprocess.run(["az", *args], check=True, capture_output=True, text=True).stdout


def main():
    sha = os.environ["TESTED_SHA"]
    endpoint = os.environ["STAGING_ENDPOINT"]
    rg, ws = os.environ["AZURE_RESOURCE_GROUP"], os.environ["AZUREML_WORKSPACE"]
    if not rg or not ws:
        raise ValueError("Set AZURE_RESOURCE_GROUP and AZUREML_WORKSPACE repository variables")
    scope = ["-g", rg, "-w", ws]
    endpoints = json.loads(run("ml", "online-endpoint", "list", *scope, "-o", "json"))
    tags = {"course": "itcs355", "lab": "4", "commit": sha}
    with tempfile.TemporaryDirectory() as directory:
        ep_path, dep_path = Path(directory)/"endpoint.yml", Path(directory)/"deployment.yml"
        model_dir = Path(directory) / "model"
        model_dir.mkdir()
        shutil.copyfile("reports/model.joblib", model_dir / "model.joblib")
        shutil.copyfile("reports/reference.csv", model_dir / "sensors.csv")
        model = json.loads(run("ml", "model", "create", "--name", "itcs355-lab4",
                              "--version", str(int(sha[:12], 16)), "--path", str(model_dir),
                              "--type", "custom_model", *scope, "-o", "json"))
        if not any(ep["name"] == endpoint for ep in endpoints):
            ep_path.write_text(yaml.safe_dump(dict(name=endpoint, auth_mode="key", tags=tags)))
            run("ml", "online-endpoint", "create", "-f", str(ep_path), *scope)
        routes = {name: {"port": 8080, "path": path} for name, path in
                  [("liveness_route", "/health"), ("readiness_route", "/ready"),
                   ("scoring_route", "/predict")]}
        spec = dict(name="staging", endpoint_name=endpoint, model=model["id"],
                    environment={"image": os.environ["CONTAINER_REGISTRY"]+":"+sha,
                                 "inference_config": routes},
                    environment_variables={"MODEL_VERSION": sha},
                    instance_type=os.environ["STAGING_INSTANCE"], instance_count=1, tags=tags)
        dep_path.write_text(yaml.safe_dump(spec))
        run("ml", "online-deployment", "create", "-f", str(dep_path), "--all-traffic", *scope)
        sample = Path(directory)/"sample.json"
        sample.write_text(json.dumps(dict(temp_c=78.4, vibration_mm_s=3.1, pressure_kpa=315.2,
                                         hours_since_service=4200, load_pct=68, ambient_humidity=55)))
        body = json.loads(run("ml", "online-endpoint", "invoke", "-n", endpoint,
                              "--request-file", str(sample), *scope))
        assert 0 <= body["probability"] <= 1, body
        assert body["model_version"] == sha, body
    print("Staging smoke test passed for " + sha)


if __name__ == "__main__":
    main()
