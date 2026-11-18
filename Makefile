# DirectGeoIO — top-level tasks.  `make test-all` runs every verification layer.
SHELL := /bin/bash
PY ?= .venv/bin/python
ECCODES ?= $(shell brew --prefix eccodes 2>/dev/null || echo /usr)
DATE ?= $(shell date -v-1d +%Y%m%d 2>/dev/null || date -d yesterday +%Y%m%d)

.PHONY: all build venv fixtures fixtures-full test test-rust test-python verify-full bench \
        cuda-check example-c clean

all: build

build:
	cargo build --release
	@mkdir -p target/release
	cc -O3 -I$(ECCODES)/include bench/eccodes_bench.c -L$(ECCODES)/lib -leccodes \
	   -Wl,-rpath,$(ECCODES)/lib -o target/release/eccodes_bench

venv:
	uv venv --python 3.12 .venv && uv pip install --python $(PY) eccodes numpy torch mlx pytest

fixtures:            ## real GFS + ECMWF Open Data messages, ecCodes re-encodings and ground truth
	python3 tools/fetch_fixtures.py --date $(DATE)
	$(PY) tools/make_fixtures.py

fixtures-full:       ## complete operational files used by verify-full / bench
	mkdir -p fixtures/raw/bench
	curl -fsS -o fixtures/raw/bench/ecmwf_ifs_0p25_$(DATE)_00z_6h.grib2 \
	  https://data.ecmwf.int/forecasts/$(DATE)/00z/ifs/0p25/oper/$(DATE)000000-6h-oper-fc.grib2
	curl -fsS -o fixtures/raw/bench/gfs_0p25_$(DATE)_00z_f006.grib2 \
	  https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.$(DATE)/00/atmos/gfs.t00z.pgrb2.0p25.f006

test-rust: build
	cargo test --release --workspace --exclude directgeoio-cuda -- --test-threads=1

test-python: build
	$(PY) -m pytest python/tests -q

verify-full: build
	$(PY) tools/verify_full.py fixtures/raw/bench/*.grib2

example-c: build
	cc -O2 -Wall -Iinclude examples/c/decode.c -Ltarget/release -ldirectgeoio \
	   -Wl,-rpath,$(CURDIR)/target/release -o target/release/decode_c
	./target/release/decode_c fixtures/gfs_complex_sfc.grib2

test: test-rust test-python example-c

test-all: test verify-full bench

bench: build
	$(PY) tools/bench.py

cuda-check:          ## compile CUDA kernels + cuFile backend inside an NVIDIA CUDA container
	tools/cuda_check.sh

clean:
	cargo clean
