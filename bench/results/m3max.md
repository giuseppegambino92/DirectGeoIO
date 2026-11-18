# DirectGeoIO benchmark — Apple M3 Max (14 cores, 10 performance)

End-to-end: GRIB2 file on disk → contiguous float32 values in (unified) memory. Best of N runs; files are in the OS page cache (warm). Speed-up is relative to single-thread ecCodes.

> Measured 2026-11-19 with ecCodes 2.50.1 / libaec 1.1.8 (Homebrew), macOS 27.2, while unrelated jobs kept the
> machine at a load average of ~11 (CPU and GPU contention). Re-run `make bench` on an idle machine for clean numbers.
> Cold-cache (NVMe-bound) numbers require `sudo purge` between runs and are not included.


### `simple_t_pl_14bit.grib2` — 8.3 M values, 14.5 MB

| backend | total ms | M values/s | output GB/s | speed-up |
|---|---:|---:|---:|---:|
| metal (Apple M3 Max) | 2.3 | 3,613 | 14.45 | 9.8× |
| cpu-rust x14 | 5.4 | 1,537 | 6.15 | 4.2× |
| ecCodes mt x14 | 8.9 | 937 | 3.75 | 2.6× |
| ecCodes mt x10 | 9.0 | 926 | 3.70 | 2.5× |
| cpu-rust x1 | 13.1 | 635 | 2.54 | 1.7× |
| ecCodes st-f32 | 21.4 | 388 | 1.55 | 1.1× |
| ecCodes st | 22.6 | 367 | 1.47 | 1.0× |

### `ecmwf_ccsds_t_pl.grib2` — 8.3 M values, 2.3 MB

| backend | total ms | M values/s | output GB/s | speed-up |
|---|---:|---:|---:|---:|
| metal (Apple M3 Max) | 6.6 | 1,262 | 5.05 | 6.8× |
| cpu-rust x14 | 10.9 | 761 | 3.04 | 4.1× |
| ecCodes mt x14 | 11.2 | 743 | 2.97 | 4.0× |
| ecCodes mt x10 | 11.2 | 742 | 2.97 | 4.0× |
| ecCodes st-f32 | 43.3 | 192 | 0.77 | 1.0× |
| ecCodes st | 44.6 | 186 | 0.74 | 1.0× |
| cpu-rust x1 | 56.2 | 148 | 0.59 | 0.8× |

### `gfs_complex_t_pl.grib2` — 8.3 M values, 6.2 MB

| backend | total ms | M values/s | output GB/s | speed-up |
|---|---:|---:|---:|---:|
| metal (Apple M3 Max) | 3.1 | 2,669 | 10.68 | 14.5× |
| cpu-rust x14 | 6.7 | 1,234 | 4.94 | 6.7× |
| ecCodes mt x10 | 12.7 | 653 | 2.61 | 3.5× |
| ecCodes mt x14 | 13.1 | 635 | 2.54 | 3.4× |
| cpu-rust x1 | 29.9 | 278 | 1.11 | 1.5× |
| ecCodes st-f32 | 44.7 | 186 | 0.74 | 1.0× |
| ecCodes st | 45.1 | 184 | 0.74 | 1.0× |

### `ecmwf_ifs_0p25_20261117_00z_6h.grib2` — 198.3 M values, 100.0 MB

| backend | total ms | M values/s | output GB/s | speed-up |
|---|---:|---:|---:|---:|
| metal (Apple M3 Max) | 48.9 | 4,058 | 16.23 | 26.3× |
| cpu-rust x14 | 189.3 | 1,047 | 4.19 | 6.8× |
| ecCodes mt x10 | 219.2 | 905 | 3.62 | 5.9× |
| ecCodes mt x14 | 241.5 | 821 | 3.28 | 5.3× |
| ecCodes st-f32 | 1229.4 | 161 | 0.65 | 1.0× |
| ecCodes st | 1285.9 | 154 | 0.62 | 1.0× |
| cpu-rust x1 | 1335.0 | 149 | 0.59 | 1.0× |

### `gfs_0p25_20261117_00z_f006.grib2` — 771.4 M values, 532.5 MB

| backend | total ms | M values/s | output GB/s | speed-up |
|---|---:|---:|---:|---:|
| metal (Apple M3 Max) | 147.5 | 5,232 | 20.93 | 28.2× |
| cpu-rust x14 | 447.9 | 1,722 | 6.89 | 9.3× |
| ecCodes mt x14 | 894.6 | 862 | 3.45 | 4.6× |
| ecCodes mt x10 | 985.0 | 783 | 3.13 | 4.2× |
| cpu-rust x1 | 2578.6 | 299 | 1.20 | 1.6× |
| ecCodes st-f32 | 4028.7 | 191 | 0.77 | 1.0× |
| ecCodes st | 4151.9 | 186 | 0.74 | 1.0× |
