# Lab 4 evidence status

Only real captured artifacts satisfy the external acceptance criteria. Local unit tests and mock webhook responses do not substitute for them.

| Item | Status / evidence to capture |
|---|---|
| Local tests and lint | Run in the isolated Lab 4 environment; see local-verification.txt. |
| Threshold calibration | Reproducible in ../threshold-calibration.json. |
| Three batch drift modes | Reproduced locally; score table in README. |
| Failed bad-contract PR | Pending: closed unmerged PR URL, failed Actions URL, named assertion and exact output. |
| Green main CI/container run | Pending: Actions run URL and integration output. |
| Staging deployment | Pending: successful CD URL and actual endpoint response containing tested SHA. |
| Scheduled provider drift job | Pending: Azure ML schedule configuration and successful scheduled run. |
| Real notification | Pending: channel screenshot and acknowledged delivery record from drift_exercise.py. |
| Live dashboard | Pending: screenshot showing shifted temperature/PSI and other required signals. |
| Scheduled detection latency | Pending: actual injection timestamp and scheduled notification timestamp; immediate manual invocation is a separate measurement. |
| Teardown | Pending: timestamped schedule and endpoint listings after deleting lab resources. |

The older local metrics_emitted.jsonl is a historical file record, not proof that cloud ingestion or channel notification succeeded. The previous 1.84-second delivery claim and unlinked blocked-PR story have been removed because supporting artifacts were unavailable.
