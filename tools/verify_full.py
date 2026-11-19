#!/usr/bin/env python3
"""Bit-exact verification of DirectGeoIO (GPU) against ecCodes for every field of arbitrary
GRIB2 files (no stored ground truth needed). Usage: tools/verify_full.py FILE... [--backend metal]"""
import argparse
import pathlib
import sys
import time

import numpy as np
import eccodes as ec

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))
import directgeoio as dg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--backend", default="auto")
a = ap.parse_args()
ok = True
for path in a.files:
    f = dg.open(path)
    t0 = time.perf_counter()
    t = f.decode(backend=a.backend)
    t_dg = time.perf_counter() - t0
    if t.native_device == "cuda":
        sys.exit("verify_full.py compares through host memory; use --backend cpu-mt here and "
                 "`cargo test -p directgeoio-cuda` (DGIO_EXTRA_FILES) for the CUDA backend")
    out = t.cpu().numpy()
    flat = out.reshape(-1)
    off, bad_fields, total, masked = 0, 0, 0, 0
    t_ec = 0.0
    with open(path, "rb") as fh:
        i = 0
        while True:
            t1 = time.perf_counter()
            h = ec.codes_grib_new_from_file(fh)
            if h is None:
                break
            ec.codes_set(h, "missingValue", 1e35)
            want = ec.codes_get_double_array(h, "values")
            t_ec += time.perf_counter() - t1
            ec.codes_release(h)
            n = want.size
            got = flat[off:off + n]
            miss = want == 1e35
            w32 = want.astype(np.float32)
            same = np.isnan(got[miss]).all() and np.array_equal(got[~miss].view(np.uint32), w32[~miss].view(np.uint32))
            if not same:
                bad_fields += 1
                d = np.flatnonzero(got[~miss].view(np.uint32) != w32[~miss].view(np.uint32))
                print(f"  MISMATCH field {i} ({f.fields[i].short_name}): {d.size} values, first at {d[:3]}")
            masked += int(miss.sum())
            total += n
            off += n
            i += 1
    status = "BIT-EXACT" if bad_fields == 0 else f"{bad_fields} FIELDS DIFFER"
    ok &= bad_fields == 0
    print(f"{pathlib.Path(path).name}: {i} fields, {total:,} values ({masked:,} masked) -> {status} "
          f"[directgeoio {t_dg*1e3:.1f} ms incl. first-call setup, ecCodes python {t_ec*1e3:.0f} ms]")
sys.exit(0 if ok else 1)
