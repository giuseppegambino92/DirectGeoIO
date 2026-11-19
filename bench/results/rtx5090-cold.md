# DirectGeoIO benchmark — NVIDIA GeForce RTX 5090 + AMD Ryzen 9 9950X3D 16-Core Processor (32 hardware threads)

End-to-end: GRIB2 file on disk → contiguous float32 values in host, unified or GPU device memory. **Cold cache**: every run starts after evicting the file from the page cache (posix_fadvise DONTNEED). Median of N runs with the p10–p90 range; speed-up compares medians against single-thread ecCodes.


### `simple_t_pl_14bit.grib2` — 8.3 M values, 14.5 MB

| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |
|---|---:|---:|---:|---:|---:|
| cuda (NVIDIA GeForce RTX 5090 (sm_120)) | 6.6 | 6.4–6.7 | 14 | 1,265 | 3.3× |
| ecCodes mt x16 | 8.2 | 7.2–22.9 | 7 | 1,015 | 2.7× |
| ecCodes mt x32 | 8.9 | 8.0–23.4 | 7 | 932 | 2.4× |
| cpu-rust x32 | 11.2 | 11.0–13.9 | 5 | 744 | 2.0× |
| ecCodes st-f32 | 20.5 | 20.3–33.9 | 7 | 405 | 1.1× |
| cpu-rust x1 | 21.5 | 20.0–36.4 | 5 | 386 | 1.0× |
| ecCodes st | 21.8 | 21.6–31.7 | 7 | 381 | 1.0× |

### `ecmwf_ccsds_t_pl.grib2` — 8.3 M values, 2.3 MB

| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |
|---|---:|---:|---:|---:|---:|
| cuda (NVIDIA GeForce RTX 5090 (sm_120)) | 3.3 | 3.2–3.9 | 14 | 2,549 | 10.1× |
| ecCodes mt x32 | 9.9 | 8.8–22.6 | 7 | 836 | 3.3× |
| ecCodes mt x16 | 10.2 | 8.6–21.8 | 7 | 812 | 3.2× |
| cpu-rust x32 | 13.9 | 13.5–15.3 | 5 | 597 | 2.4× |
| ecCodes st-f32 | 30.2 | 30.0–44.9 | 7 | 275 | 1.1× |
| ecCodes st | 33.0 | 32.9–47.8 | 7 | 252 | 1.0× |
| cpu-rust x1 | 43.1 | 42.3–61.8 | 5 | 193 | 0.8× |

### `gfs_complex_t_pl.grib2` — 8.3 M values, 6.2 MB

| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |
|---|---:|---:|---:|---:|---:|
| cuda (NVIDIA GeForce RTX 5090 (sm_120)) | 5.4 | 5.4–5.8 | 14 | 1,530 | 8.2× |
| cpu-rust x32 | 10.2 | 9.9–12.8 | 5 | 813 | 4.3× |
| ecCodes mt x16 | 13.9 | 12.7–25.7 | 7 | 597 | 3.2× |
| ecCodes mt x32 | 14.1 | 12.3–24.6 | 7 | 589 | 3.1× |
| cpu-rust x1 | 21.2 | 20.8–37.4 | 5 | 392 | 2.1× |
| ecCodes st | 44.3 | 43.5–52.7 | 7 | 187 | 1.0× |
| ecCodes st-f32 | 64.3 | 64.0–72.2 | 7 | 129 | 0.7× |

### `ecmwf_ifs_0p25_20261117_00z_6h.grib2` — 198.3 M values, 100.0 MB

| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |
|---|---:|---:|---:|---:|---:|
| cuda (NVIDIA GeForce RTX 5090 (sm_120)) | 87.6 | 86.8–90.1 | 10 | 2,265 | 9.8× |
| ecCodes mt x16 | 181.3 | 178.1–196.7 | 5 | 1,094 | 4.7× |
| ecCodes mt x32 | 202.8 | 201.7–217.6 | 5 | 978 | 4.2× |
| cpu-rust x32 | 297.4 | 295.6–298.1 | 5 | 667 | 2.9× |
| ecCodes st | 856.2 | 854.7–865.9 | 5 | 232 | 1.0× |
| ecCodes st-f32 | 859.6 | 855.4–867.9 | 5 | 231 | 1.0× |
| cpu-rust x1 | 1458.9 | 1454.0–1483.5 | 5 | 136 | 0.6× |

### `gfs_0p25_20261117_00z_f006.grib2` — 771.4 M values, 532.5 MB

| backend | median ms | p10–p90 ms | runs | M values/s | speed-up |
|---|---:|---:|---:|---:|---:|
| cuda (NVIDIA GeForce RTX 5090 (sm_120)) | 242.2 | 240.3–249.0 | 10 | 3,185 | 14.5× |
| ecCodes mt x16 | 900.4 | 890.5–907.4 | 5 | 857 | 3.9× |
| ecCodes mt x32 | 1206.2 | 1205.6–1218.1 | 5 | 640 | 2.9× |
| cpu-rust x32 | 1373.8 | 1365.3–1383.0 | 5 | 562 | 2.6× |
| cpu-rust x1 | 3281.1 | 3245.7–3295.8 | 5 | 235 | 1.1× |
| ecCodes st | 3505.2 | 3482.2–3557.3 | 5 | 220 | 1.0× |
| ecCodes st-f32 | 5425.5 | 5419.0–5440.6 | 5 | 142 | 0.6× |
