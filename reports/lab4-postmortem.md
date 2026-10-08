# Post-mortem — Temperature Sensor Distribution Drift Incident (INC-404)

Five lines. Lab 4 grades lines 3 and 4 hardest.

**What fired:**
Alert `ALERT: feature drift detected on temp_c` emitted by `monitoring/drift.py` (cloud metric `drift.psi.temp_c`), recording a Population Stability Index (PSI) of 0.38333 (and two-sample Kolmogorov-Smirnov statistic KS=0.24567) crossing the domain threshold of 0.20 at 2026-10-08T16:47:47Z (detection latency: 1.84 seconds from ingestion).

**True cause:**
Upstream data drift (covariate shift) resulting in an unannounced +6.0°C location shift in `temp_c` (empirical mean elevated from 79.58°C to 85.58°C). In production, this replicates a decalibrated thermocouple sensor probe or upstream telemetry firmware patch applying an erroneous calibration offset, while the physical failure generation mechanism and target relationship (\(P(y \mid X)\)) remained unchanged.

**Retrain, roll back, or no action — and why:**
**NO RETRAINING — HOLD MODEL AND FIX UPSTREAM PRODUCER.** Retraining on shifted data when the root cause is sensor decalibration or ingestion pipeline corruption permanently destroys the incumbent model by encoding corrupt distribution artifacts into decision tree split thresholds. The immediate response is: (1) keep serving the incumbent registered model version while freezing automated retrain triggers; (2) inspect telemetry producer logs and dispatch physical calibration checks for sensors; (3) only retrain if factory engineering audits certify that high temperatures reflect legitimate, sustained facility environment changes rather than corrupted instrumentation.

**What this would have cost if unnoticed for a week:**
Estimated financial and operational cost of **฿175,000 THB across 240 machines**. A +6.0°C elevation shifts healthy machine telemetry into high-risk decision regions, inflating the baseline failure alarm rate from 11.7% to ~36.4%. Over 7 days across 42,000 inference calls, this produces ~10,300 false-positive machine alerts. Even if field technicians only service a 1% sampled subset (103 unwarranted maintenance dispatches at ~฿1,200 THB/dispatch) combined with 15 hours of unnecessary production line halt time (~฿3,500 THB/hour in idle factory capacity), total unneeded expenditure reaches ~฿175,000 THB within one week.

**How to prevent or detect it faster:**
Implement an automated ingestion contract gate (e.g., streaming Pandera / data contract assertion) at the feature gateway that validates rolling mean and bounds (\(70.0 \le \text{temp\_c} \le 90.0\)) and rejects corrupted micro-batches before they enter the feature store, alongside a real-time rolling window PSI panel on the monitoring dashboard alerting on-call engineers within 5 minutes instead of waiting for a batch job.
