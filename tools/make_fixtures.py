#!/usr/bin/env python3
"""Build the DirectGeoIO test corpus + ground truth from real GRIB2 downloads.

Inputs  (fixtures/raw, from tools/fetch_fixtures.py): real NOAA GFS + ECMWF Open Data.
Outputs (fixtures/):
  *.grib2            real messages as published, plus messages re-encoded by ecCodes
                     itself (packingType/bitsPerValue changes) to cover Template 5.0
                     at arbitrary non-byte-aligned widths and Template 5.42 with bitmaps.
  ref/<file>.<i>.npy float64 values decoded by ecCodes (NaN where Section 6 masks a point)
  manifest.json      per-message metadata used by the test-suite and benchmarks.
"""
import json
import os
import sys

import numpy as np
import eccodes as ec

RAW = "fixtures/raw"
OUT = "fixtures"
MISSING = 1.0e35


def messages(path):
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                return
            yield h


def values_of(h):
    ec.codes_set(h, "missingValue", MISSING)
    v = ec.codes_get_double_array(h, "values")
    v = np.where(v == MISSING, np.nan, v)
    return v


def meta_of(h):
    keys = ["shortName", "typeOfLevel", "level", "dataRepresentationTemplateNumber", "packingType",
            "bitsPerValue", "bitmapPresent", "numberOfValues", "numberOfDataPoints", "Ni", "Nj",
            "referenceValue", "binaryScaleFactor", "decimalScaleFactor", "totalLength"]
    m = {}
    for k in keys:
        try:
            m[k] = ec.codes_get(h, k)
        except ec.KeyValueNotFoundError:
            pass
    if m.get("packingType") == "grid_ccsds":
        for k in ["ccsdsFlags", "ccsdsBlockSize", "ccsdsRsi"]:
            m[k] = ec.codes_get(h, k)
    return m


def repack(h, packing, bpv):
    """Re-encode the real field with ecCodes itself (genuine ecCodes bitstream)."""
    vals = ec.codes_get_double_array(h, "values")  # missing already = MISSING sentinel
    c = ec.codes_clone(h)
    ec.codes_set(c, "missingValue", MISSING)
    ec.codes_set(c, "packingType", packing)
    ec.codes_set(c, "bitsPerValue", bpv)
    ec.codes_set_double_array(c, "values", vals)
    return c


# (output name, source file, source message indices, packing, bitsPerValue or None=keep)
PLAN = [
    # --- real, as published -------------------------------------------------------------
    ("ecmwf_ccsds_sfc.grib2", "ecmwf_0p25_sfc_mix.grib2", None, None, None),
    ("ecmwf_ccsds_t_pl.grib2", "ecmwf_0p25_t_pl.grib2", None, None, None),
    ("gfs_complex_sfc.grib2", "gfs_0p25_sfc_mix.grib2", None, None, None),
    ("gfs_complex_t_pl.grib2", "gfs_0p25_t_pl.grib2", None, None, None),
    ("gfs_complex_missing.grib2", "gfs_0p25_missing_mgmt.grib2", None, None, None),
    # --- Template 5.0 simple packing, arbitrary bit widths (ecCodes-encoded) ---------------
    ("simple_12bit.grib2", "ecmwf_0p25_sfc_mix.grib2", [0, 1, 2, 3], "grid_simple", 12),
    ("simple_19bit.grib2", "ecmwf_0p25_sfc_mix.grib2", [3, 4, 6], "grid_simple", 19),
    ("simple_widths.grib2", "ecmwf_0p25_sfc_mix.grib2", [0, 0, 0, 0, 0, 0], "grid_simple", [1, 5, 11, 23, 27, 32]),
    ("simple_t_pl_14bit.grib2", "ecmwf_0p25_t_pl.grib2", None, "grid_simple", 14),
    ("simple_bitmap.grib2", "gfs_0p25_sfc_mix.grib2", [6, 7, 8], "grid_simple", [11, 13, 17]),
    # --- Template 5.42 CCSDS at other widths and with bitmaps (ecCodes/libaec-encoded) ------
    ("ccsds_widths.grib2", "ecmwf_0p25_sfc_mix.grib2", [3, 3, 3, 3, 3], "grid_ccsds", [3, 8, 17, 21, 24]),
    ("ccsds_bitmap.grib2", "gfs_0p25_sfc_mix.grib2", [6, 7, 8], "grid_ccsds", [11, 9, 17]),
]


def main():
    os.makedirs(os.path.join(OUT, "ref"), exist_ok=True)
    manifest = {}
    for name, src, idxs, packing, bpv in PLAN:
        srcmsgs = list(messages(os.path.join(RAW, src)))
        sel = range(len(srcmsgs)) if idxs is None else idxs
        out_path = os.path.join(OUT, name)
        entries = []
        offset = 0
        with open(out_path, "wb") as fo:
            for j, i in enumerate(sel):
                h = srcmsgs[i]
                if packing is None:
                    m = h
                else:
                    ec.codes_set(h, "missingValue", MISSING)
                    b = bpv[j] if isinstance(bpv, list) else bpv
                    m = repack(h, packing, b)
                blob = ec.codes_get_message(m)
                fo.write(blob)
                md = meta_of(m)
                md.update(offset=offset, length=len(blob))
                vals = values_of(m)
                ref = os.path.join(OUT, "ref", f"{name}.{j}.npy")
                np.save(ref, vals)
                md["ref"] = os.path.relpath(ref, OUT)
                md["n_missing"] = int(np.isnan(vals).sum())
                entries.append(md)
                offset += len(blob)
                if m is not h:
                    ec.codes_release(m)
        for h in srcmsgs:
            ec.codes_release(h)
        manifest[name] = entries
        kinds = sorted({(e["packingType"], e["bitsPerValue"], e["bitmapPresent"]) for e in entries})
        print(f"{name:28s} {len(entries):2d} msgs {offset/1e6:6.2f} MB  {kinds}")
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)


if __name__ == "__main__":
    sys.exit(main())
