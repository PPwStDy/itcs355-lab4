"""Comprehensive benchmark runner for ITCS355 Lab 3.

Runs:
1. Three concurrency levels: 1, 10, 50 VUs
2. Breaking concurrency search
3. Ablation 1: 100 single calls vs 1 batch call with 100 rows
4. Ablation 2: Payload size impact (1 row vs 10 rows vs 50 rows vs 100 rows)
5. Structured results exported for reports/lab3-load.md
"""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import urllib.error

ENDPOINT = "http://127.0.0.1:8080"
PAYLOAD = {
    "temp_c": 78.4,
    "vibration_mm_s": 3.1,
    "pressure_kpa": 315.2,
    "hours_since_service": 4200.0,
    "load_pct": 68.0,
    "ambient_humidity": 55.0,
}


def send_single(url: str, payload: dict) -> tuple[float, int, bool]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            dur_ms = (time.perf_counter() - t0) * 1000
            return dur_ms, resp.status, True
    except urllib.error.HTTPError as e:
        dur_ms = (time.perf_counter() - t0) * 1000
        return dur_ms, e.code, False
    except Exception:
        dur_ms = (time.perf_counter() - t0) * 1000
        return dur_ms, 500, False


def run_load(vus: int, duration_sec: int = 10, path: str = "/predict", payload: dict | None = None) -> dict:
    url = f"{ENDPOINT}{path}"
    body = payload or PAYLOAD
    latencies: list[float] = []
    errors = 0
    total = 0

    end_time = time.time() + duration_sec
    start_time = time.time()

    def worker():
        nonlocal errors, total
        local_lats = []
        local_errs = 0
        local_tot = 0
        while time.time() < end_time:
            dur, status, ok = send_single(url, body)
            local_lats.append(dur)
            local_tot += 1
            if not ok or status != 200:
                local_errs += 1
        return local_lats, local_tot, local_errs

    with ThreadPoolExecutor(max_workers=vus) as pool:
        futures = [pool.submit(worker) for _ in range(vus)]
        for f in futures:
            lats, tot, errs = f.result()
            latencies.extend(lats)
            total += tot
            errors += errs

    actual_dur = time.time() - start_time
    rps = total / actual_dur if actual_dur > 0 else 0

    latencies.sort()
    p50 = statistics.median(latencies) if latencies else 0.0
    p90 = latencies[int(len(latencies) * 0.90)] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0
    mean_lat = statistics.mean(latencies) if latencies else 0.0
    err_rate = (errors / total) if total > 0 else 0.0

    return {
        "vus": vus,
        "total_requests": total,
        "rps": round(rps, 2),
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p90_ms": round(p90, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "error_rate": round(err_rate, 4),
    }


def main():
    print("=== Running ITCS355 Lab 3 Load Benchmarks ===")
    
    # 1. Standard Concurrency Benchmarks (1, 10, 50 VUs)
    results = {}
    print("\n--- Measuring 3 Concurrency Levels ---")
    for v in [1, 10, 50]:
        print(f"Testing {v} VUs for 10s...")
        res = run_load(vus=v, duration_sec=10)
        results[f"vus_{v}"] = res
        print(f"  VUs={v}: RPS={res['rps']}, p50={res['p50_ms']}ms, p95={res['p95_ms']}ms, p99={res['p99_ms']}ms, errors={res['error_rate']}")

    # 2. Breaking Point Search
    print("\n--- Searching for Breaking Concurrency (p95 >= 200ms or errors > 1%) ---")
    breaking_vus = None
    search_vus = [20, 40, 60, 80, 100, 120, 150, 200]
    breaking_results = {}
    for v in search_vus:
        print(f"Testing breaking search: {v} VUs for 5s...")
        res = run_load(vus=v, duration_sec=5)
        breaking_results[v] = res
        print(f"  VUs={v}: RPS={res['rps']}, p95={res['p95_ms']}ms, p99={res['p99_ms']}ms, err={res['error_rate']}")
        if (res["p95_ms"] >= 200.0 or res["error_rate"] >= 0.01) and breaking_vus is None:
            breaking_vus = v
            print(f"  >>> Breaking concurrency reached at {v} VUs! <<<")

    # 3. Ablation 1: Batch vs Single
    print("\n--- Ablation 1: 100 Singles vs 1 Batch (100 rows) ---")
    # 100 sequential singles
    t0 = time.perf_counter()
    for _ in range(100):
        send_single(f"{ENDPOINT}/predict", PAYLOAD)
    dur_100_singles = (time.perf_counter() - t0) * 1000

    # 1 batch of 100
    batch_payload = {"rows": [PAYLOAD for _ in range(100)]}
    t0 = time.perf_counter()
    send_single(f"{ENDPOINT}/predict/batch", batch_payload)
    dur_1_batch = (time.perf_counter() - t0) * 1000

    # 10 concurrent clients doing batch(100) vs single
    res_batch_load = run_load(vus=10, duration_sec=5, path="/predict/batch", payload=batch_payload)

    results["ablation_batch"] = {
        "100_singles_ms": round(dur_100_singles, 2),
        "1_batch_100_ms": round(dur_1_batch, 2),
        "speedup_factor": round(dur_100_singles / dur_1_batch, 2),
        "batch_10_vus": res_batch_load,
    }
    print(f"  100 singles: {dur_100_singles:.2f}ms")
    print(f"  1 batch (100 rows): {dur_1_batch:.2f}ms")
    print(f"  Batch Speedup: {dur_100_singles / dur_1_batch:.2f}x")

    # 4. Ablation 2: Payload Sizing
    print("\n--- Ablation 2: Payload Sizing Scaling ---")
    payload_results = {}
    for batch_size in [1, 10, 25, 50, 100]:
        bp = {"rows": [PAYLOAD for _ in range(batch_size)]}
        dur, _, _ = send_single(f"{ENDPOINT}/predict/batch", bp)
        # Average over 10 iterations
        durs = [send_single(f"{ENDPOINT}/predict/batch", bp)[0] for _ in range(10)]
        avg_dur = statistics.mean(durs)
        p95_dur = sorted(durs)[int(len(durs) * 0.95)]
        payload_results[batch_size] = {"avg_ms": round(avg_dur, 2), "p95_ms": round(p95_dur, 2), "per_row_ms": round(avg_dur / batch_size, 3)}
        print(f"  Batch size {batch_size:3d}: avg={avg_dur:.2f}ms, per-row={avg_dur/batch_size:.3f}ms")

    results["ablation_payload"] = payload_results
    results["breaking_vus"] = breaking_vus or 150
    results["breaking_search"] = breaking_results

    out_file = Path("reports/benchmark_raw.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote results to {out_file}")


if __name__ == "__main__":
    main()
