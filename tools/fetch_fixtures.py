#!/usr/bin/env python3
"""Fetch real WMO GRIB2 messages from NOAA GFS (AWS Open Data) and ECMWF Open Data.

Only selected messages are downloaded, using HTTP Range requests driven by the
providers' own index files (.idx for GFS, .index JSON-lines for ECMWF).

Usage: tools/fetch_fixtures.py [--date YYYYMMDD] [--outdir fixtures/raw]
"""
import argparse
import datetime as dt
import json
import os
import sys
import urllib.request

GFS_BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{d}/00/atmos/gfs.t00z.pgrb2.0p25.f006"
ECMWF_BASE = "https://data.ecmwf.int/forecasts/{d}/00z/ifs/0p25/oper/{d}000000-6h-oper-fc"

# (variable, level) substrings matched against GFS .idx lines
GFS_FIELDS = [
    ("PRMSL", "mean sea level"),
    ("TMP", "2 m above ground"),
    ("UGRD", "10 m above ground"),
    ("VGRD", "10 m above ground"),
    ("HGT", "500 mb"),
    ("TMP", "850 mb"),
    ("SOILW", "0-0.1 m below ground"),  # land-only field -> Section 6 bitmap
    ("TSOIL", "0-0.1 m below ground"),
    ("WEASD", "surface"),
    ("ICEC", "surface"),                # ocean-only -> bitmap
]
# complex packing with missing-value management (levels below terrain are missing)
GFS_MISSING = [("TMP", "2743 m above mean sea level"), ("UGRD", "1829 m above mean sea level"), ("VGRD", "3658 m above mean sea level")]
# multi-level temperature for a 4D (var, level, lat, lon) tensor
GFS_LEVELS = ["1000 mb", "925 mb", "850 mb", "700 mb", "500 mb", "300 mb", "250 mb", "200 mb"]

ECMWF_FIELDS = [
    {"param": "2t", "levtype": "sfc"},
    {"param": "10u", "levtype": "sfc"},
    {"param": "10v", "levtype": "sfc"},
    {"param": "msl", "levtype": "sfc"},
    {"param": "sp", "levtype": "sfc"},
    {"param": "tp", "levtype": "sfc"},
    {"param": "z", "levtype": "pl", "levelist": "500"},
]
ECMWF_PL_T = ["1000", "925", "850", "700", "500", "300", "250", "200"]


def http_get(url, rng=None):
    req = urllib.request.Request(url, headers={"User-Agent": "directgeoio-fixtures/0.1"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def gfs_ranges(idx_text):
    lines = [l.split(":") for l in idx_text.strip().splitlines()]
    out = []
    for i, f in enumerate(lines):
        start = int(f[1])
        end = int(lines[i + 1][1]) - 1 if i + 1 < len(lines) else None
        out.append((f[3], f[4], start, end))
    return out


def fetch_gfs(date, outdir):
    base = GFS_BASE.format(d=date)
    entries = gfs_ranges(http_get(base + ".idx").decode())
    wanted = []
    for var, lev in GFS_FIELDS:
        for e in entries:
            if e[0] == var and e[1] == lev:
                wanted.append(e)
                break
    sfc = b"".join(http_get(base, (s, e)) for (_, _, s, e) in wanted)
    p = os.path.join(outdir, "gfs_0p25_sfc_mix.grib2")
    open(p, "wb").write(sfc)
    print(f"{p}: {len(wanted)} msgs, {len(sfc)/1e6:.1f} MB")

    mm = [e for var, lev in GFS_MISSING for e in entries if e[0] == var and e[1] == lev]
    blob = b"".join(http_get(base, (s, e)) for (_, _, s, e) in mm)
    p = os.path.join(outdir, "gfs_0p25_missing_mgmt.grib2")
    open(p, "wb").write(blob)
    print(f"{p}: {len(mm)} msgs, {len(blob)/1e6:.1f} MB")

    pl = [e for lev in GFS_LEVELS for e in entries if e[0] == "TMP" and e[1] == lev][: len(GFS_LEVELS)]
    blob = b"".join(http_get(base, (s, e)) for (_, _, s, e) in pl)
    p = os.path.join(outdir, "gfs_0p25_t_pl.grib2")
    open(p, "wb").write(blob)
    print(f"{p}: {len(pl)} msgs, {len(blob)/1e6:.1f} MB")


def fetch_ecmwf(date, outdir):
    base = ECMWF_BASE.format(d=date)
    idx = [json.loads(l) for l in http_get(base + ".index").decode().splitlines() if l.strip()]

    def pick(spec):
        for e in idx:
            if all(e.get(k) == v for k, v in spec.items()):
                return e
        raise SystemExit(f"ECMWF: no message matching {spec}")

    sel = [pick(s) for s in ECMWF_FIELDS]
    blob = b"".join(http_get(base + ".grib2", (e["_offset"], e["_offset"] + e["_length"] - 1)) for e in sel)
    p = os.path.join(outdir, "ecmwf_0p25_sfc_mix.grib2")
    open(p, "wb").write(blob)
    print(f"{p}: {len(sel)} msgs, {len(blob)/1e6:.1f} MB")

    sel = [pick({"param": "t", "levtype": "pl", "levelist": l}) for l in ECMWF_PL_T]
    blob = b"".join(http_get(base + ".grib2", (e["_offset"], e["_offset"] + e["_length"] - 1)) for e in sel)
    p = os.path.join(outdir, "ecmwf_0p25_t_pl.grib2")
    open(p, "wb").write(blob)
    print(f"{p}: {len(sel)} msgs, {len(blob)/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(dt.date.today() - dt.timedelta(days=1)).strftime("%Y%m%d"))
    ap.add_argument("--outdir", default="fixtures/raw")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    fetch_gfs(a.date, a.outdir)
    fetch_ecmwf(a.date, a.outdir)
    with open(os.path.join(a.outdir, "SOURCE.txt"), "w") as f:
        f.write(f"date={a.date} run=00z step=6h\n{GFS_BASE.format(d=a.date)}\n{ECMWF_BASE.format(d=a.date)}.grib2\n")


if __name__ == "__main__":
    sys.exit(main())
