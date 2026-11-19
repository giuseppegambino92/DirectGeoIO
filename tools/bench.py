#!/usr/bin/env python3
"""End-to-end benchmark: file on disk -> contiguous float32 tensor.

Backends: ecCodes C single-thread (double API, and native float API), ecCodes C multi-thread,
DirectGeoIO Metal (MTLIOCommandQueue + GPU), DirectGeoIO Rust CPU (1 thread / all cores).
Writes bench/results/<host>.json and .md.  Usage: tools/bench.py [files...] [--repeat N]
"""
import argparse
import json
import os
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DGIO = ROOT / "target/release/dgio"
ECB = ROOT / "target/release/eccodes_bench"

DEFAULT = [
    "fixtures/simple_t_pl_14bit.grib2",
    "fixtures/ecmwf_ccsds_t_pl.grib2",
    "fixtures/gfs_complex_t_pl.grib2",
    "fixtures/raw/bench/ecmwf_ifs_0p25_20261117_00z_6h.grib2",
    "fixtures/raw/bench/gfs_0p25_20261117_00z_f006.grib2",
]


def sh(cmd):
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", default=DEFAULT)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--cold", action="store_true", help="evict each file from the page cache before every run (Linux)")
    a = ap.parse_args()
    ncpu = os.cpu_count()
    perf = int(sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip() or ncpu) if sys.platform == "darwin" else ncpu // 2
    rows = []
    for f in a.files:
        path = str(ROOT / f)
        big = os.path.getsize(path) > 50e6
        rep = max(5, a.repeat // 2) if big else a.repeat
        print(f"== {f} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
        for mode, threads in [("st", 1), ("st-f32", 1), ("mt", perf), ("mt", ncpu)]:
            r = json.loads(sh([str(ECB), mode, path, str(threads), str(rep)] + (["cold"] if a.cold else [])))
            r["backend"] = f"ecCodes {mode}" + (f" x{threads}" if mode == "mt" else "")
            rows.append(r)
            print(f"  {r['backend']:<28} median {r['median_s']*1e3:9.2f} ms  p90 {r['p90_s']*1e3:9.2f}", flush=True)
        out = json.loads(sh([str(DGIO), "bench", path, "--json", "--repeat", str(rep * 2)] + (["--cold"] if a.cold else [])
                            + os.environ.get("DGIO_BENCH_ARGS", "").split()))
        for r in out:
            r["file"] = os.path.basename(path)
            rows.append(r)
            print(f"  {r['backend']:<28} median {r['median_s']*1e3:9.2f} ms  p90 {r['p90_s']*1e3:9.2f}  (best: io {r['io_s']*1e3:.2f}, decode {r['compute_s']*1e3:.2f})", flush=True)
    host = platform.node().split(".")[0].lower() or "host"
    if sys.platform == "darwin":
        chip = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        tag = chip.replace("Apple ", "").replace(" ", "").lower()
    else:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "cpu")
        gpu = sh(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).strip().splitlines()[0]
        chip = f"{gpu} + {cpu}"
        tag = gpu.replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "").lower()
    outdir = ROOT / "bench/results"
    outdir.mkdir(parents=True, exist_ok=True)
    if a.cold:
        tag += "-cold"
    (outdir / f"{tag}.json").write_text(json.dumps({"chip": chip, "cpus": ncpu, "perf_cores": perf, "rows": rows}, indent=1))
    # markdown
    md = [f"# DirectGeoIO benchmark — {chip} ({ncpu} hardware threads)\n",
          "End-to-end: GRIB2 file on disk → contiguous float32 values in host, unified or GPU device memory. "
          + ("**Cold cache**: every run starts after evicting the file from the page cache (posix_fadvise DONTNEED). "
             if a.cold else "Warm page cache. ")
          + "Median of N runs with the p10–p90 range; speed-up compares medians against single-thread ecCodes.\n"]
    files = []
    for r in rows:
        if r["file"] not in files:
            files.append(r["file"])
    for fn in files:
        rs = [r for r in rows if r["file"] == fn]
        base = next(r["median_s"] for r in rs if r["backend"] == "ecCodes st")
        v = rs[0]["values"]
        md.append(f"\n### `{fn}` — {v/1e6:.1f} M values, {rs[0]['bytes']/1e6:.1f} MB\n")
        md.append("| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |\n|---|---:|---:|---:|---:|---:|")
        for r in sorted(rs, key=lambda r: r["median_s"]):
            t = r["median_s"]
            md.append(f"| {r['backend']} | {t*1e3:.1f} | {r['p10_s']*1e3:.1f}–{r['p90_s']*1e3:.1f} | {r['n']} | {v/t/1e6:,.0f} | {base/t:.1f}× |")
    (outdir / f"{tag}.md").write_text("\n".join(md) + "\n")
    print(f"wrote bench/results/{tag}.json/.md")


if __name__ == "__main__":
    main()
