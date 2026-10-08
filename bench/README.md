# Embedded HNSW comparison

This is the project's only benchmark. It compares sqlite-hnsw, Chroma
PersistentClient, and Milvus Lite HNSW on the complete SIFT1M dataset and keeps
one report representing the current code and measurement process.

The current result is
[SIFT1M embedded HNSW comparison](results/sift1m-embedded-hnsw-comparison.md).

## Requirements

- Linux with `taskset` and `/proc`;
- Python 3.12;
- a C17 compiler and CMake 3.20 or newer;
- SQLite CLI, headers, and shared library version 3.51.2;
- enough disk and memory for SIFT1M and all three persistent indexes.

## Dataset

Install the conversion dependencies and download the ANN-Benchmarks SIFT1M
copy:

```sh
python3 -m pip install h5py numpy
./bench/embedded/download_sift1m.sh /tmp/sift1m-fvecs
```

Set `PYTHON` when the conversion dependencies are installed in a virtual
environment rather than the default `python3`.

The downloader verifies the source SHA-256 and produces:

```text
sift_base.fvecs
sift_query.fvecs
sift_groundtruth.ivecs
```

An existing TEXMEX-format directory with these files can be used directly.

## Run

From the repository root:

```sh
./bench/embedded/run_comparison.sh \
  /tmp/sift1m-fvecs /tmp/sqlite-hnsw-embedded 0-15
```

The optional third argument is the CPU set. `SQLITE_PREFIX` may be set when
SQLite is not installed below the prefix inferred from `sqlite3`. The runner
checks that the CLI, Python stdlib SQLite, and APSW all use SQLite 3.51.2.

The script creates a locked virtual environment, builds and tests the
extension, runs every engine in separate monitored processes, generates the
report, and then removes the virtual environment, build directory, databases,
and raw JSON records. It overwrites
`bench/results/sift1m-embedded-hnsw-comparison.md`; no result history is kept.

## Method

- Dataset: one million 128-dimensional SIFT vectors using L2 and `k=10`.
- HNSW configuration: `M=16` and `ef_construction=128`, verified for every
  engine.
- Query sweep: `ef_search` values 10, 20, 30, 40, 50, 60, 80, 100, 150, and
  200.
- Repeats: three deterministic query permutations with rotated parameter order.
- Queries: 100 measured queries and 10 disjoint warm-up queries per parameter.
- Recall: official SIFT1M ground truth.
- Memory: peak RSS of the complete process tree sampled from `/proc`.

Persistent-ready build time starts immediately before database or collection
creation and ends after vector insertion, HNSW construction, persistence or
checkpoint, and close. Dataset validation, the explicit sqlite-hnsw integrity
check, and storage accounting are outside the timer.

Connection-cold latency starts after opening the persistent database. It does
not clear the operating-system page cache. Steady-state query measurements are
single-client, read-only, and unfiltered.

The comparison uses one built index per engine, so it does not measure build
variance. Results do not represent cosine or inner-product distance, other
dimensions, concurrent clients, filtered search, or mixed read/write workloads.

