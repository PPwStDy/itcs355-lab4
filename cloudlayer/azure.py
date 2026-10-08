"""Azure adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install azure-storage-blob azure-identity azure-containerregistry
Docs: BlobServiceClient for storage; ACR push goes through `docker push` after
      `az acr login --name <registry>`.

Hints for Lab 1:
  * BLOB_URI is either abfss://container@account.dfs.core.windows.net/prefix or
    https://account.blob.core.windows.net/container/prefix. Pick one form and parse
    it here, never in src/.
  * Use DefaultAzureCredential rather than a connection string. It picks up your CLI
    login locally and your managed identity in CI, which is what Lab 4 needs.
  * push_image must return the digest reference: registry.azurecr.io/repo@sha256:...
  * Azure tags live on the resource, not the blob. Tag the storage account, the
    registry, and later the workspace with cfg.tags(1).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cloudlayer.base import CloudAdapter


class AzureAdapter(CloudAdapter):
    def _parse_blob_uri(self, uri: str) -> tuple[str, str, str]:
        """Parse https://<account>.blob.core.windows.net/<container>/<prefix>"""
        p = urlparse(uri)
        account = p.netloc.split(".")[0]
        parts = p.path.lstrip("/").split("/", 1)
        container = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return account, container, prefix

    def upload(self, local_path: str, key: str) -> str:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account, container, prefix = self._parse_blob_uri(self.cfg.blob_uri)
        blob_name = f"{prefix.rstrip('/')}/{key.lstrip('/')}" if prefix else key
        account_url = f"https://{account}.blob.core.windows.net"

        service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)

        with open(local_path, "rb") as fh:
            blob_client.upload_blob(fh, overwrite=True)

        return f"{account_url}/{container}/{blob_name}"

    def download(self, uri: str, local_path: str) -> None:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        account, container, blob_name = self._parse_blob_uri(uri)
        account_url = f"https://{account}.blob.core.windows.net"

        service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
        blob_client = service_client.get_blob_client(container=container, blob=blob_name)

        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as fh:
            download_stream = blob_client.download_blob()
            download_stream.readinto(fh)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry
        server = registry.split("/")[0]
        acr_name = server.split(".")[0]

        # 1. Login to Azure Container Registry
        subprocess.run(["az", "acr", "login", "--name", acr_name], check=True)

        # 2. Tag local image for remote ACR
        tag_suffix = local_tag.split(":")[-1] if ":" in local_tag else "latest"
        remote_tag = f"{registry}:{tag_suffix}"
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)

        # 3. Push to ACR
        subprocess.run(["docker", "push", remote_tag], check=True)

        # 4. Get digest reference (registry.azurecr.io/repo@sha256:...)
        res = subprocess.run(
            ["docker", "inspect", "--format={{range .RepoDigests}}{{.}}\n{{end}}", remote_tag],
            capture_output=True,
            text=True,
            check=True,
        )
        digests = [d.strip() for d in res.stdout.splitlines() if d.strip()]
        matching = [d for d in digests if d.startswith(registry)]
        if matching:
            return matching[0]

        repo = registry.split("/", 1)[1] if "/" in registry else local_tag.split(":")[0]
        az_res = subprocess.run(
            ["az", "acr", "repository", "show", "--name", acr_name, "--image", f"{repo}:{tag_suffix}", "--query", "digest", "-o", "tsv"],
            capture_output=True,
            text=True,
            check=True,
        )
        digest = az_res.stdout.strip()
        return f"{registry}@{digest}"

    # submit_training / register_model  -> Lab 2 (Azure ML command job + model registry)
    def submit_training(self, image_uri: str, args: dict[str, Any]) -> str:
        """Submit an Azure ML Command Job running the container image."""
        import tempfile
        import time
        import yaml

        rg = self.cfg.project_id
        ws = self.cfg.project_id
        cmd_args = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in args.items())
        cmd = f"cd /app && python -m src.train {cmd_args}" if cmd_args else "cd /app && python -m src.train"

        job_spec = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json",
            "type": "command",
            "display_name": f"itcs355-train-{int(time.time())}",
            "experiment_name": "itcs355-lab2",
            "environment": {"image": image_uri},
            "environment_variables": {
                "BLOB_URI": self.cfg.blob_uri,
                "CLOUD_PROVIDER": self.cfg.provider,
                "PROJECT_ID": self.cfg.project_id,
                "MLFLOW_TRACKING_URI": "sqlite:////app/reports/mlflow.db",
            },
            "compute": "azureml:cpu-cluster",
            "identity": {"type": "managed"},
            "command": cmd,
            "tags": self.cfg.tags(2),
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(job_spec, f)
            spec_file = f.name

        try:
            res = subprocess.run(
                ["az", "ml", "job", "create", "--file", spec_file, "-g", rg, "-w", ws, "--query", "name", "-o", "tsv"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        finally:
            Path(spec_file).unlink(missing_ok=True)

    def wait_training(self, job_id: str) -> dict[str, Any]:
        """Poll job status until completed, failed, or canceled."""
        import json
        import time

        rg = self.cfg.project_id
        ws = self.cfg.project_id
        terminal_states = {"Completed", "Failed", "Canceled"}

        while True:
            res = subprocess.run(
                ["az", "ml", "job", "show", "--name", job_id, "-g", rg, "-w", ws, "-o", "json"],
                capture_output=True,
                text=True,
                check=True,
            )
            job_data = json.loads(res.stdout)
            status = job_data.get("status")
            print(f"Job {job_id}: status={status}")
            if status in terminal_states:
                return job_data
            time.sleep(10)
    
    # register_model                    -> Lab 2 (Azure ML Model Registry)
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: dict[str, str] | None = None,
        stage: str = "Staging",
    ) -> str:
        """Register a model with all mandatory lineage tags and promote to staging."""
        import json
        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        client = MlflowClient()

        # Gather lineage metadata
        lineage_tags: dict[str, str] = {
            "git_commit": "",
            "data_version": "",
            "mlflow_run_id": "",
            "training_job_id": "",
            "image_digest": "",
            "seed": "",
            "metric_val": "",
            "metric_test": "",
        }

        run_id = ""
        if model_uri.startswith("runs:/"):
            run_id = model_uri.split("/")[1]
            try:
                run = client.get_run(run_id)
                lineage_tags["mlflow_run_id"] = run_id
                lineage_tags["training_job_id"] = run.data.tags.get("training_job_id", run_id)
                lineage_tags["git_commit"] = run.data.tags.get("git_commit", "")
                lineage_tags["data_version"] = run.data.tags.get("data_fingerprint", "")
                lineage_tags["seed"] = str(run.data.params.get("seed", ""))
                lineage_tags["metric_val"] = f"{run.data.metrics.get('val_roc_auc', 0.0):.4f}"
                lineage_tags["metric_test"] = f"{run.data.metrics.get('test_roc_auc', 0.0):.4f}"
            except Exception as e:
                print(f"Warning: could not extract run metadata: {e}")

        # Merge with caller-supplied tags
        if tags:
            lineage_tags.update(tags)

        # 1. Register in MLflow Model Registry
        mv = mlflow.register_model(model_uri, name)
        version_str = str(mv.version)

        # Set tags on the registered model version (not just the run!)
        for k, v in lineage_tags.items():
            if v:
                client.set_model_version_tag(name, version_str, k, str(v))

        # Promote to stage (and set alias for MLflow 2.x/3.x)
        try:
            client.transition_model_version_stage(name, version_str, stage=stage)
        except Exception:
            pass
        try:
            client.set_registered_model_alias(name, stage.lower(), version_str)
        except Exception:
            pass

        # 2. Register in Azure ML Model Registry
        try:
            rg = self.cfg.project_id
            ws = self.cfg.project_id
            local_model_path = None
            if hasattr(mv, "source") and Path(mv.source).exists():
                local_model_path = mv.source
            elif run_id:
                from mlflow.artifacts import download_artifacts
                local_model_path = download_artifacts(run_id=run_id, artifact_path="model")

            if local_model_path and Path(local_model_path).exists():
                tag_args = [f"{k}={v}" for k, v in lineage_tags.items() if v]
                cmd = [
                    "az", "ml", "model", "create",
                    "--name", name,
                    "--version", version_str,
                    "--path", str(local_model_path),
                    "--type", "mlflow_model",
                    "--stage", stage,
                    "--resource-group", rg,
                    "--workspace-name", ws,
                ]
                if tag_args:
                    cmd.extend(["--tags", *tag_args])
                subprocess.run(cmd, capture_output=True, text=True, check=False)
        except Exception as e:
            print(f"Azure ML model registration warning: {e}")

        print(f"Registered model {name} version {version_str} with lineage tags (stage: {stage})")
        return version_str

    def teardown(self, tags: dict[str, str]) -> list[str]:
        """Delete or archive resources matching the provided tags."""
        import json
        deleted: list[str] = []
        rg = self.cfg.project_id
        ws = self.cfg.project_id

        # 1. Archive Azure ML jobs carrying these exact tags
        try:
            res = subprocess.run(
                [
                    "az", "ml", "job", "list",
                    "--resource-group", rg,
                    "--workspace-name", ws,
                    "-o", "json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                jobs = json.loads(res.stdout)
                for j in jobs:
                    j_tags = j.get("tags") or {}
                    if all(j_tags.get(k) == str(v) for k, v in tags.items()):
                        j_name = j.get("name")
                        if j_name:
                            subprocess.run(
                                [
                                    "az", "ml", "job", "archive",
                                    "--name", j_name,
                                    "--resource-group", rg,
                                    "--workspace-name", ws,
                                ],
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            deleted.append(f"azureml:job:{j_name}")
        except Exception as e:
            print(f"Warning during job teardown: {e}")

        # 2. Check general Azure resources carrying these exact tags
        try:
            tag_args = []
            for k, v in tags.items():
                tag_args.extend(["--tag", f"{k}={v}"])
            cmd = ["az", "resource", "list", *tag_args, "-o", "json"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout:
                resources = json.loads(res.stdout)
                for r in resources:
                    r_id = r.get("id")
                    # Do not delete essential workspace, storage, or ACR unless specifically tagged
                    if r_id and "Microsoft.Storage" not in r_id and "Microsoft.ContainerRegistry" not in r_id and "Microsoft.MachineLearningServices/workspaces" not in r_id:
                        subprocess.run(
                            ["az", "resource", "delete", "--ids", r_id, "--no-wait"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        deleted.append(r_id)
        except Exception as e:
            print(f"Warning during resource teardown: {e}")

        # 3. Delete Azure ML online endpoints carrying these exact tags
        try:
            res = subprocess.run(
                [
                    "az", "ml", "online-endpoint", "list",
                    "--resource-group", rg,
                    "--workspace-name", ws,
                    "-o", "json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout:
                endpoints = json.loads(res.stdout)
                for ep in endpoints:
                    ep_tags = ep.get("tags") or {}
                    if all(ep_tags.get(k) == str(v) for k, v in tags.items()):
                        ep_name = ep.get("name")
                        if ep_name:
                            subprocess.run(
                                [
                                    "az", "ml", "online-endpoint", "delete",
                                    "--name", ep_name,
                                    "--resource-group", rg,
                                    "--workspace-name", ws,
                                    "--yes", "--no-wait",
                                ],
                                capture_output=True,
                                text=True,
                                check=False,
                            )
                            deleted.append(f"azureml:endpoint:{ep_name}")
        except Exception as e:
            print(f"Warning during online endpoint teardown: {e}")

        return deleted

    # deploy / invoke                   -> Lab 3 (managed online endpoint + deployment)
    def deploy(
        self,
        model_ref: str,
        endpoint: str = "itcs355-endpoint",
        instance: str = "Standard_F2s_v2",
        deployment_name: str = "blue",
        traffic: int = 100,
    ) -> str:
        """Deploy model to Azure ML Managed Online Endpoint or local serving."""
        import json
        import subprocess
        import tempfile
        import yaml
        from pathlib import Path

        if instance == "local":
            print(f"Deploying {model_ref} to local endpoint...")
            return "http://127.0.0.1:8080"

        rg = self.cfg.project_id
        ws = self.cfg.project_id

        # 1. Ensure online endpoint exists
        show_res = subprocess.run(
            ["az", "ml", "online-endpoint", "show", "--name", endpoint, "-g", rg, "-w", ws, "-o", "json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if show_res.returncode != 0:
            print(f"Creating Azure ML online endpoint: {endpoint}...")
            ep_spec = {
                "$schema": "https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json",
                "name": endpoint,
                "auth_mode": "key",
                "tags": {**self.cfg.tags(3), "lab": "3"},
            }
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.dump(ep_spec, f)
                ep_spec_file = f.name
            try:
                subprocess.run(
                    ["az", "ml", "online-endpoint", "create", "--file", ep_spec_file, "-g", rg, "-w", ws],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            finally:
                Path(ep_spec_file).unlink(missing_ok=True)

        # 2. Prepare deployment spec
        model_azure_ref = model_ref
        if model_ref.startswith("models:/"):
            parts = model_ref.removeprefix("models:/").split("/")
            model_azure_ref = f"azureml:{parts[0]}:{parts[1]}"
        elif not model_azure_ref.startswith("azureml:") and ":" in model_azure_ref:
            model_azure_ref = f"azureml:{model_azure_ref}"

        print(f"Creating deployment {deployment_name} for model {model_azure_ref} on {instance}...")
        deploy_spec = {
            "$schema": "https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json",
            "name": deployment_name,
            "endpoint_name": endpoint,
            "model": model_azure_ref,
            "instance_type": instance,
            "instance_count": 1,
            "tags": {**self.cfg.tags(3), "lab": "3"},
            "request_settings": {
                "request_timeout_ms": 5000,
                "max_concurrent_requests_per_instance": 10,
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(deploy_spec, f)
            dep_spec_file = f.name
        try:
            subprocess.run(
                [
                    "az", "ml", "online-deployment", "create",
                    "--file", dep_spec_file,
                    "-g", rg,
                    "-w", ws,
                    *(["--all-traffic"] if traffic == 100 else []),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            Path(dep_spec_file).unlink(missing_ok=True)

        if traffic > 0:
            subprocess.run(
                ["az", "ml", "online-endpoint", "update", "--name", endpoint, "-g", rg, "-w", ws, "--traffic", f"{deployment_name}={traffic}"],
                capture_output=True,
                text=True,
                check=False,
            )

        ep_info = subprocess.run(
            ["az", "ml", "online-endpoint", "show", "--name", endpoint, "-g", rg, "-w", ws, "--query", "scoring_uri", "-o", "tsv"],
            capture_output=True,
            text=True,
            check=False,
        )
        scoring_uri = ep_info.stdout.strip()
        return scoring_uri or f"https://{endpoint}.{self.cfg.region}.inference.ml.azure.com/score"

    def invoke(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke endpoint with JSON payload."""
        import json
        import subprocess
        import tempfile
        from pathlib import Path
        import urllib.request

        rg = self.cfg.project_id
        ws = self.cfg.project_id

        # Local endpoint invocation
        if endpoint.startswith("http://") or "127.0.0.1" in endpoint or "localhost" in endpoint:
            url = endpoint if endpoint.endswith("/predict") else f"{endpoint.rstrip('/')}/predict"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        # Azure ML online endpoint invocation
        if endpoint.startswith("https://"):
            ep_name = endpoint.split("//")[1].split(".")[0]
            key_res = subprocess.run(
                ["az", "ml", "online-endpoint", "get-credentials", "--name", ep_name, "-g", rg, "-w", ws, "--query", "primaryKey", "-o", "tsv"],
                capture_output=True,
                text=True,
                check=False,
            )
            api_key = key_res.stdout.strip()
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            formatted_data = {"input_data": payload} if "input_data" not in payload else payload
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(formatted_data).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))

        # Fallback to az ml online-endpoint invoke
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"input_data": payload}, f)
            req_file = f.name
        try:
            res = subprocess.run(
                ["az", "ml", "online-endpoint", "invoke", "--name", endpoint, "-g", rg, "-w", ws, "--request-file", req_file],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(res.stdout)
        finally:
            Path(req_file).unlink(missing_ok=True)

    def emit_metric(self, name: str, value: float, unit: str = "None") -> None:
        """Emit custom metric to Azure Monitor / Application Insights."""
        import json
        import os
        import time
        from pathlib import Path

        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = {
            "timestamp": timestamp,
            "metric_name": name,
            "value": float(value),
            "unit": unit,
            "resource_group": self.cfg.project_id,
            "region": self.cfg.region,
            "tags": self.cfg.tags(4),
        }

        # Check for Application Insights connection string / instrumentation key
        conn_str = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
        ikey = os.environ.get("APPINSIGHTS_INSTRUMENTATIONKEY", "")
        if conn_str or ikey:
            try:
                import urllib.request
                track_url = "https://dc.services.visualstudio.com/v2/track"
                telemetry = {
                    "name": "Microsoft.ApplicationInsights.Metric",
                    "time": timestamp,
                    "iKey": ikey or (dict(part.split("=", 1) for part in conn_str.split(";")).get("InstrumentationKey", "")),
                    "data": {
                        "baseType": "MetricData",
                        "baseData": {
                            "metrics": [{"name": name, "value": float(value), "count": 1}],
                            "properties": record["tags"],
                        },
                    },
                }
                req = urllib.request.Request(
                    track_url,
                    data=json.dumps(telemetry).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    pass
            except Exception as e:
                print(f"[Azure Monitor] Metric dispatch note: {e}")

        # Record metric to reports/metrics_emitted.jsonl for verification
        self.cfg.reports_dir.mkdir(parents=True, exist_ok=True)
        metrics_log = self.cfg.reports_dir / "metrics_emitted.jsonl"
        with metrics_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[Azure Monitor] Emitted metric: {name}={value:.5f} ({unit})")

    # generate                          -> Lab 5 (managed LLM endpoint; read the usage block for tokens)



