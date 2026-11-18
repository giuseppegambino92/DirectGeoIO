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
    ap.add_argument("--repeat", type=int, default=5)
    a = ap.parse_args()
    ncpu = os.cpu_count()
    perf = int(sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip() or ncpu) if sys.platform == "darwin" else ncpu
    rows = []
    for f in a.files:
        path = str(ROOT / f)
        big = os.path.getsize(path) > 50e6
        rep = max(2, a.repeat // 2) if big else a.repeat
        print(f"== {f} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
        for mode, threads in [("st", 1), ("st-f32", 1), ("mt", perf), ("mt", ncpu)]:
            r = json.loads(sh([str(ECB), mode, path, str(threads), str(rep)]))
            r["backend"] = f"ecCodes {mode}" + (f" x{threads}" if mode == "mt" else "")
            rows.append(r)
            print(f"  {r['backend']:<28} {r['total_s']*1e3:9.2f} ms", flush=True)
        out = json.loads(sh([str(DGIO), "bench", path, "--json", "--repeat", str(rep * 2)]))
        for r in out:
            r["file"] = os.path.basename(path)
            rows.append(r)
            print(f"  {r['backend']:<28} {r['total_s']*1e3:9.2f} ms  (io {r['io_s']*1e3:.2f}, decode {r['compute_s']*1e3:.2f})", flush=True)
    host = platform.node().split(".")[0].lower() or "host"
    chip = sh(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() if sys.platform == "darwin" else platform.processor()
    outdir = ROOT / "bench/results"
    outdir.mkdir(parents=True, exist_ok=True)
    tag = chip.replace("Apple ", "").replace(" ", "").lower()
    (outdir / f"{tag}.json").write_text(json.dumps({"chip": chip, "cpus": ncpu, "perf_cores": perf, "rows": rows}, indent=1))
    # markdown
    md = [f"# DirectGeoIO benchmark — {chip} ({ncpu} cores, {perf} performance)\n",
          "End-to-end: GRIB2 file on disk → contiguous float32 values in (unified) memory. Best of N runs; "
          "files are in the OS page cache (warm). Speed-up is relative to single-thread ecCodes.\n"]
    files = []
    for r in rows:
        if r["file"] not in files:
            files.append(r["file"])
    for fn in files:
        rs = [r for r in rows if r["file"] == fn]
        base = next(r["total_s"] for r in rs if r["backend"] == "ecCodes st")
        v = rs[0]["values"]
        md.append(f"\n### `{fn}` — {v/1e6:.1f} M values, {rs[0]['bytes']/1e6:.1f} MB\n")
        md.append("| backend | total ms | M values/s | output GB/s | speed-up |\n|---|---:|---:|---:|---:|")
        for r in sorted(rs, key=lambda r: r["total_s"]):
            t = r["total_s"]
            md.append(f"| {r['backend']} | {t*1e3:.1f} | {v/t/1e6:,.0f} | {v*4/t/1e9:.2f} | {base/t:.1f}× |")
    (outdir / f"{tag}.md").write_text("\n".join(md) + "\n")
    print(f"wrote bench/results/{tag}.json/.md")


if __name__ == "__main__":
    main()
