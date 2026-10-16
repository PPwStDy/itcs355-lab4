# ITCS355 Lab 4 — CI/CD, Observability, and Drift

Predict machine failure from sensor readings. The [handout](course/labs/lab-04-cicd-monitoring-drift.md) defines the acceptance criteria. Code and local verification are available; external evidence is tracked explicitly in [reports/evidence/README.md](reports/evidence/README.md). Do not treat local tests as proof of a cloud deployment or delivered notification.

## Reproduce

Use Python 3.12, matching CI and both images:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make data
ruff check src/ service/ monitoring/ scripts/ tests/
python scripts/portability_audit.py
pytest -q tests
python scripts/calibrate_drift.py
python scripts/export_model.py --out reports/model.joblib
```

Missing training data fails the required tests rather than silently skipping them. Training and serving use matching scikit-learn, pandas, NumPy and joblib pins. Export the model again after changing those dependencies.

## Task 1 — Tests and incidents

| Test | Incident caught |
|---|---|
| `test_schema_columns_present_and_typed` | Gateway removes or renames a sensor column, changes its type, or adds unexpected columns. |
| `test_no_nulls_in_required_columns` | Lost telemetry introduces missing required readings; tolerated null rate is 0%. |
| `test_features_within_plausible_ranges` | Probe reports impossible values such as −999°C; temperature contract is −10 to 140°C. |
| `test_target_is_binary_and_not_degenerate` | Label ETL emits unknown classes or labels every machine identically. |
| `test_identifier_is_unique` | Retried ingestion duplicates readings. |
| `test_no_machine_leaks_across_splits` | Machine readings leak across partitions and inflate validation results. |

Split determinism, coverage and content fingerprint tests protect preprocessing. Model behaviour tests cover a healthy machine, valid probabilities, wear-related risk, inference latency and nonconstant predictions. The separate container integration job checks readiness, a numeric probability in [0,1], response shape, model version in body and header, and exported metrics. Monitoring tests check failed notification delivery, expired windows, HTTP 4xx/5xx metrics and a deliberately missing feature.

## Tasks 2–3 — CI, deployment and blocked commit

CI runs on pull requests and main pushes: lint → portability audit → unit tests → data contracts → behaviour → service/monitoring tests → images → container integration. A main push uploads the tested serving image, model and frozen training reference. CD downloads that artifact from the triggering CI run, checks out its exact `head_sha`, pushes the same image tagged by that SHA, registers the model/reference, deploys to Azure ML, and invokes the actual endpoint. Any deployment or smoke-test failure stops CD. Pull request runs cannot deploy.

Configure repository variables `ACR_NAME`, `CONTAINER_REGISTRY` (registry/repository, without a tag), `AZURE_RESOURCE_GROUP`, `AZUREML_WORKSPACE`; optionally `STAGING_ENDPOINT` and `STAGING_INSTANCE`. Configure `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` as repository secrets with Azure OIDC federation for the staging environment. The identity needs ACR push and Azure ML deployment permissions; the endpoint identity needs image-pull access. No credential file is committed.

**Required external evidence remains:** after committing the working solution, open a temporary PR changing `scripts/make_dataset.py` to omit `ambient_humidity`. Save the PR URL, failed CI URL, named failing test and exact output. Close it without merging. The local negative test proves the assertion can fail, but does not replace the required failed GitHub run. No failed PR or 28-second CI timing is claimed here.

## Task 4 — Live dashboard and SLO

The serving app exports `/metrics` with request counts by status class, latency histogram, model version, rolling feature means and PSI. The input window contains successful prediction inputs only, covers at most five minutes and 10,000 rows, and requires 30 rows before reporting PSI. One worker per container keeps these metrics coherent; scale containers for more throughput. `/monitoring/window` requires a configured bearer token.

`monitoring/dashboard.json` is a provisioned Grafana dashboard with request rate, 4xx/5xx error fractions, p50/p95/p99, rolling temperature mean, rolling PSI and model version. `monitoring/slo.yaml` states targets, windows and error-budget responses. Prometheus rule definitions are in `monitoring/alerts.yml`.

Start the portable live stack after exporting the model. Set `MONITORING_TOKEN`, `GRAFANA_ADMIN_PASSWORD` and `DRIFT_ALERT_WEBHOOK_URL` in the shell or an ignored local environment file, then:

```sh
docker compose -f compose.monitoring.yml up --build -d
```

Open [Grafana](http://localhost:3000/d/lab4) and [Prometheus](http://localhost:9090). Send prediction traffic; empty windows do not imply healthy inputs. The detector checks every 300 seconds and sends breached scores to the configured Slack incoming webhook. Errors remain visible in its logs. The Prometheus alert rule also exposes pending/firing state; notification delivery comes from the scheduled detector.

## Task 5 — Scheduled drift and threshold

PSI and KS compare recent production inputs with the frozen training reference. The Prometheus `drift_psi{feature="temp_c"}` metric matches the dashboard query. The detector saves every score and delivery acknowledgement; missing secrets or unsuccessful HTTP delivery fail visibly. There is no fallback to a local successful smoke test or a fake notification.

The committed [calibration](reports/threshold-calibration.json) uses seed 42 and 500 bootstrap windows of 300 rows: baseline PSI p99≈0.06445, zero baseline alerts at 0.20, and 100% detection of a +6°C shift in that simulation. Threshold 0.20 therefore separates this tested fault from sampled noise. It is a starting point for this synthetic dataset; recheck with real seasonal windows and investigate small windows rather than assuming the simulation guarantees production sensitivity.

For Azure ML scheduling, build `monitoring/Dockerfile.job` using the tested serving image as `SERVING_IMAGE`, register it as an Azure ML environment, and register the deployed training reference as a versioned `uri_file` data asset. Grant the job compute managed identity read access to three Key Vault secrets: `lab4-monitoring-token`, `lab4-drift-alert-webhook`, `lab4-applicationinsights-connection-string`.

```sh
python scripts/prepare_drift_schedule.py \
  --environment azureml:lab4-monitor:1 --compute azureml:YOUR_COMPUTE \
  --reference azureml:lab4-reference:1 \
  --window-url https://YOUR_FEATURE_GATEWAY/monitoring/window \
  --vault-url https://YOUR_VAULT.vault.azure.net
az ml schedule create -f reports/drift-schedule.yml -g YOUR_RESOURCE_GROUP -w YOUR_WORKSPACE
```

The generator prepares a five-minute Azure ML pipeline schedule without creating resources or storing secrets. Use a reachable, authenticated feature gateway serving actual inference inputs; an Azure ML scoring URL alone is not a general metrics/window URL. The scheduled job reads secrets at runtime, saves the detector report, emits cloud metrics and notifies the real channel. Capture the successful scheduled run after configuring it. Job startup and polling add delay and compute cost.

`.github/workflows/drift.yml` provides an additional GitHub scheduler for a reachable monitoring window. It needs `DRIFT_WINDOW_URL`, `DRIFT_REFERENCE_RUN_ID` (the deployed main CI run), `MONITORING_TOKEN`, and `DRIFT_ALERT_WEBHOOK_URL`. Its 90-day reference artifact must be renewed before expiration. GitHub schedule timing is best effort; the Azure schedule is the provider-specific submission path.

## Task 6 — Injected drift

The retained full-dataset batch results reproduce:

| Mode | Reference mean | Current mean | KS | PSI | Result |
|---|---:|---:|---:|---:|---|
| shift +6°C | 79.58 | 85.58 | 0.24567 | 0.38333 | alert |
| scale ×2 | 79.58 | 79.58 | 0.18783 | 0.51663 | alert |
| mix | 79.58 | 80.1723 | 0.03417 | 0.00873 | stable |

For the actual live exercise, start with an empty or expired input window, configure the real Slack webhook and monitoring token, then:

```sh
python scripts/drift_exercise.py --endpoint http://127.0.0.1:8080
```

This sends 300 shifted readings to the service, immediately invokes detection and saves actual injection, detection and acknowledged delivery timestamps in `reports/evidence/drift-exercise.json`. Its elapsed time measures immediate invocation, **not** scheduler detection latency. For scheduled detection latency, record injection time, allow the real schedule to run, then subtract injection time from that run's delivered timestamp. Capture the real notification and dashboard while the shifted window is present. A successful webhook response proves transport acknowledgement; include a channel screenshot to prove the notification was seen.

The [five-part post-mortem](reports/lab4-postmortem.md) keeps the incumbent model and fixes the corrupted producer. Its weekly cost is explicitly an estimate with stated assumptions, not an observed bill. Do not retrain on corrupted telemetry.

## Teardown and evidence

```sh
docker compose -f compose.monitoring.yml down --remove-orphans
az ml schedule disable -n lab4-drift -g YOUR_RESOURCE_GROUP -w YOUR_WORKSPACE
az ml schedule delete -n lab4-drift -g YOUR_RESOURCE_GROUP -w YOUR_WORKSPACE --yes
az ml online-endpoint delete -n itcs355-staging -g YOUR_RESOURCE_GROUP -w YOUR_WORKSPACE --yes
```

Alternatively use `make teardown` for the local stack and `make teardown-cloud` for schedules/endpoints tagged `course=itcs355, lab=4` in the explicitly configured Azure workspace.

Only remove the lab resources you created. If the optional GitHub drift schedule was enabled, disable that workflow too. Check the cloud schedule and endpoint lists after deletion and retain timestamped evidence; no teardown has been claimed on the basis of code alone.

Implementation references: [Azure custom-container deployment](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-custom-container?view=azureml-api-2), [Azure ML schedule schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-schedule?view=azureml-api-2), [Prometheus Python histograms](https://prometheus.github.io/client_python/instrumenting/histogram/).
