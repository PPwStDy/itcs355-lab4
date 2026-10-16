# Post-mortem — simulated temperature calibration offset

**What fired:** Local batch detection reproduced PSI=0.38333 and KS=0.24567 for a +6°C shift at threshold 0.20. A real channel delivery and its latency must be captured with `scripts/drift_exercise.py`; the older metric log alone does not prove notification.

**True cause:** The exercise added +6°C to `temp_c`, simulating a sensor calibration or upstream transformation error; the physical process and labels were not changed. This is corrupted measurement input, not evidence that the real world changed.

**Retrain, roll back, or no action — and why:** Keep the incumbent model, stop automated retraining, fix or revert the faulty producer, and replay validated readings. Retrain only if engineering confirms a legitimate sustained change and newly labelled validation data supports it.

**What this would have cost if unnoticed for a week:** An illustrative assumption of 103 unnecessary dispatches at ฿1,200 plus 15 hours of downtime at ฿3,500 gives ฿176,100. These are planning assumptions, not observed model outcomes or an actual bill; measured false-positive counts and dispatch costs are needed for a production estimate.

**How to prevent or detect it faster:** Validate schema, nulls and physical bounds at ingestion, monitor the five-minute feature window against a frozen reference, and deliver the threshold alert to the configured real channel. A +6°C offset can remain inside physical bounds, so range checks alone cannot catch it; scheduled detection includes window fill, polling, job startup and notification delay.
