# sqlite3-vector HNSW

## Introduction

`vechnsw` is a C17 SQLite loadable extension that provides an HNSW-backed
virtual table for approximate nearest-neighbor vector search.

This repository is in a fast-iteration phase. SQL shape, shadow tables, and
on-disk layout are intentionally not compatibility promises yet.

The performance target is x86-64 with AVX2. The scalar distance path exists for
debugging and tests, but it is not the v1 performance target. ARM/NEON and SSE
paths are intentionally not implemented.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

The build produces:

- `build/vechnsw.so`: SQLite loadable extension.
- `build/test_vechnsw`: unit test binary.
- `build/bench_vechnsw`: synthetic benchmark.
- `build/bench_sift`: SIFT1M benchmark.

## Usage

This example can be pasted directly into the `sqlite3` CLI after building
`build/vechnsw.so`. It shows loading the extension, creating an ordinary
metadata table, creating a vector table, inserting document rows and vector
BLOBs, rebuilding the HNSW graph, querying nearest neighbors, and joining ANN
results back to metadata.

```sql
.load ./build/vechnsw sqlite3_vechnsw_init

CREATE TABLE docs(
  id INTEGER PRIMARY KEY,
  title TEXT NOT NULL
);

CREATE VIRTUAL TABLE doc_vec USING vec_hnsw(
  dim=2,
  metric=l2,
  build_threads=2,
  build_batch_size=4
);

INSERT INTO docs(id, title)
VALUES
  (1, 'origin'),
  (2, 'unit-x'),
  (3, 'unit-y');

INSERT INTO doc_vec(rowid, embedding)
VALUES
  (1, x'0000000000000000'),
  (2, x'0000803f00000000'),
  (3, x'000000000000803f');

SELECT vec_hnsw_rebuild('main', 'doc_vec');

SELECT rowid, distance
FROM doc_vec
WHERE embedding MATCH x'0000000000000000'
  AND k = 2
ORDER BY distance;

SELECT docs.id, docs.title, ann.distance
FROM (
  SELECT rowid, distance
  FROM doc_vec
  WHERE embedding MATCH x'0000000000000000'
    AND k = 2
  ORDER BY distance
) AS ann
JOIN docs ON docs.id = ann.rowid
ORDER BY ann.distance;
```

The BLOB literals above are little-endian float32 vectors:

- `x'0000000000000000'`: `(0.0, 0.0)`
- `x'0000803f00000000'`: `(1.0, 0.0)`
- `x'000000000000803f'`: `(0.0, 1.0)`

## API Reference

### Virtual Table

```sql
CREATE VIRTUAL TABLE table_name USING vec_hnsw(
  dim=N,
  metric=l2|cosine|ip,
  m=16,
  ef_construction=128,
  ef_search=64,
  build_threads=8,
  build_batch_size=512,
  seed=...
);
```

`dim` is required. Other options have defaults.

| Option | Default | Limit |
| --- | ---: | --- |
| `dim` | required | `1..4096` |
| `metric` | `cosine` | `l2`, `cosine`, `ip` |
| `m` | `16` | `2..128` |
| `ef_construction` | `128` | `m..8192` |
| `ef_search` | `64` | `1..8192` |
| `build_threads` | `8` | `2..256` |
| `build_batch_size` | `512` | `2..65536` |
| `seed` | deterministic default | positive signed integer |

### Vector Format

Vectors are fixed-dimension little-endian float32 `BLOB`s with finite values.
The BLOB length must be exactly `dim * 4` bytes, up to 16384 bytes. NaN and
infinity are rejected.

Supported metrics:

- `l2`: Euclidean distance.
- `cosine`: `1 - cosine_similarity`.
- `ip`: negative inner product, so smaller values rank first.

### Writes

```sql
INSERT INTO doc_vec(rowid, embedding) VALUES (<rowid>, <float32_blob>);
DELETE FROM doc_vec WHERE rowid = <rowid>;
UPDATE doc_vec SET embedding = <float32_blob> WHERE rowid = <rowid>;
```

`rowid` is the logical id returned by ANN queries and is normally the primary
key of an ordinary metadata table. `DELETE` tombstones the active vector.
`UPDATE` tombstones the old vector and inserts the new vector as a new node.

### ANN Query

```sql
SELECT rowid, distance
FROM doc_vec
WHERE embedding MATCH <query_blob>
  AND k = <positive_integer>
ORDER BY distance;
```

`ef_search` can be overridden per query:

```sql
SELECT rowid, distance
FROM doc_vec
WHERE embedding MATCH <query_blob>
  AND k = <positive_integer>
  AND ef_search = <1_to_8192>
ORDER BY distance;
```

ANN query mode materializes `rowid` and `distance`; `embedding` is `NULL` in ANN
results. Use a subquery and join on `rowid` when metadata or the original vector
BLOB is needed.

### Scalar Functions

```sql
SELECT vec_dim(<float32_blob>);
SELECT vec_distance(<float32_blob_a>, <float32_blob_b>, 'l2');
SELECT vec_distance(<float32_blob_a>, <float32_blob_b>, 'cosine');
SELECT vec_distance(<float32_blob_a>, <float32_blob_b>, 'ip');
```

`vec_dim()` and `vec_distance()` enforce the same maximum dimension and
finite-value rule as the virtual table.

### Optimize

```sql
SELECT vec_hnsw_rebuild('main', 'doc_vec');
```

Run `vec_hnsw_rebuild(db_name, table_name)` after bulk loading vectors. It
loads active vectors, compacts them into dense internal node ids, rebuilds the
HNSW graph, and persists the graph in one savepoint.

Use `vec_hnsw_rebuild()` after large batches of `UPDATE` or `DELETE` as well:
incremental writes leave tombstoned nodes behind until rebuild compacts the
table.

Tuning guidelines:

- Increase `ef_search` to improve recall at higher query cost.
- Increase `m` and `ef_construction` to improve graph quality at higher build
  time and storage cost.
- Use larger `k` values to overfetch before joining to selective metadata
  filters.

## Benchmark

### Synthetic Benchmark

```sh
./build/bench_vechnsw            # 100000 vectors, 128 dimensions, 20 queries
./build/bench_vechnsw 10000 128 5 128 64 8 512
```

The v1 acceptance target is:

- `100K x 128`, `cosine`, `M=16`, `ef_construction=128`, `k=10`.
- Bulk build under 15 seconds on an 8-core AVX2 x86-64 machine with SSD/NVMe.
- `recall@10 >= 0.90` at `ef_search=64`.

If build time misses the target, optimize in this order: adjacency BLOB
serialization, AVX2 distance hot path, allocation reuse, visited-set clearing,
candidate queue layout, then HNSW parameters.

### SIFT1M Method

```sh
./build/bench_sift /tmp/sift1m-fvecs 10000 100 64
./build/bench_sift /tmp/sift1m-fvecs 100000 1000 64 8 512
./build/bench_sift /tmp/sift1m-fvecs 1000000 10000 64 8 512
```

`bench_sift` reads the raw SIFT files directly:

- `sift_base.fvecs`
- `sift_query.fvecs`
- `sift_groundtruth.ivecs`

It uses `metric=l2`, `M=16`, `ef_construction=128`, `k=10`, and configurable
`build_threads`/`build_batch_size`. It reports build time, P50/P95/P99 query
latency, QPS, recall, and database size. Rowids are stored as `base_id + 1`,
then converted back to SIFT's 0-based ids for groundtruth comparison.

To compare sqlite3-vector with Chroma and Milvus Lite through the same Python
client path:

```sh
python3 -m venv /tmp/vechnsw-bench-venv
/tmp/vechnsw-bench-venv/bin/python -m pip install -r bench/requirements-sift-compare.txt
/tmp/vechnsw-bench-venv/bin/python bench/bench_sift_compare.py --engine all
```

The comparison benchmark uses SIFT1M with `metric=L2`, `M=16`,
`ef_construction=128`, `ef_search=64`, and `k=10`. Build total time covers
creating the local store, loading vectors, and making the index queryable.
Dependency installation, CMake builds, and Python venv creation are not
included. Chroma is measured as an embedded persistent client with online HNSW
maintenance during inserts; Milvus is measured as Milvus Lite, not Milvus
Standalone or Distributed.

### SIFT1M Comparison Results

Measured on 2026-10-14 with an Intel Xeon Gold 5218 machine, 64 logical CPUs,
and 376 GiB RAM. The Python environment used Python 3.12.11, NumPy 2.5.4,
Chroma 1.5.9, pymilvus 3.0.2, and pysqlite3 SQLite 3.51.1. This interim run
uses all 1M base vectors and a 1K-query sample; full 10K-query SIFT1M results
should be measured separately for final recall reporting.

Command:

```sh
cmake --build build -j2
/tmp/vechnsw-bench-venv/bin/python -m py_compile bench/bench_sift_compare.py
/tmp/vechnsw-bench-venv/bin/python bench/bench_sift_compare.py \
  --engine all \
  --base-limit 1000000 \
  --query-limit 1000 \
  --work-dir /tmp/vechnsw_sift_compare_1m_q1000 \
  2>&1 | tee /tmp/vechnsw_sift_compare_1m_q1000.log
```

The run used insert batch size 5000. sqlite3-vector used `build_threads=8` and
`build_batch_size=512`. The comparison script prints query progress every 100
completed queries by default, including completed, remaining, elapsed seconds,
and current QPS. Use `--query-progress-interval 0` to disable it, or pass
another positive interval.

| Engine | Build total (s) | Queries | Query total (s) | P50 latency (ms) | P95 latency (ms) | P99 latency (ms) | QPS | Recall@10 | Storage bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite3-vector | 232.368 | 1000 | 2.406 | 0.644 | 0.715 | 0.768 | 415.599 | 0.9629 | 768466944 |
| Chroma | 239.189 | 1000 | 1.279 | 1.207 | 1.566 | 1.634 | 781.864 | 0.9549 | 834339868 |
| Milvus Lite | 125.357 | 1000 | 686.004 | 647.678 | 1208.085 | 1215.407 | 1.458 | 0.9914 | 790130993 |

Milvus Lite built the local index quickly in this embedded Lite setup, but
single-query search through the Python API was much slower than the other two
engines. Its 1K-query run took 686.004 seconds, so a full 10K-query Milvus
Lite run is expected to be very long with this exact client loop.

## Internal Design

`vec_hnsw` is a virtual table, not a native SQLite index access method. It uses
ordinary SQLite shadow tables for metadata, vectors, and graph edges.

### Shadow Tables

For a virtual table named `doc_vec`, the extension creates:

- `doc_vec_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)`
- `doc_vec_vectors(node_id INTEGER PRIMARY KEY, rowid_value INTEGER NOT NULL,
  level INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, embedding BLOB
  NOT NULL)`
- `doc_vec_vectors_rowid_idx` on `(rowid_value, deleted)`
- `doc_vec_graph(node_id INTEGER NOT NULL, layer INTEGER NOT NULL, neighbors
  BLOB NOT NULL, PRIMARY KEY(node_id, layer))`

`neighbors` is a compact BLOB containing a little-endian array of 32-bit
neighbor node ids. Metadata integer values are parsed strictly; corrupted shadow
metadata is rejected instead of being silently coerced.

### Insert

`INSERT` validates the vector length and finite values, tombstones any active
row with the same logical `rowid`, appends a new internal `node_id`, stores the
embedding, and marks the graph stale.

### Select

`MATCH + k` uses ANN search. The table cache is reused only when both the graph
generation and `PRAGMA data_version` match the cached values. A valid persisted
graph is loaded from shadow tables; a stale graph is built in memory for the
query path.

A full scan, without `MATCH`, returns active rows ordered by logical rowid and
materializes the `embedding` BLOB. ANN mode returns `rowid` and `distance`; the
`embedding` column is `NULL`.

### Update

`UPDATE` tombstones the old logical rowid, validates and inserts the new vector
as a new internal node, and marks the graph stale. Rebuild later removes the
tombstoned node from the compacted representation.

### Delete

`DELETE` tombstones the active row and marks the graph stale. Tombstoned vectors
remain in shadow storage until rebuild.

### Rebuild

`vec_hnsw_rebuild(db_name, table_name)` loads active vectors, compacts them into
dense internal node ids, assigns HNSW levels, builds the graph with
batch-parallel search/selection and batched pruning, then persists compacted
vectors, graph edges, entry node, max level, generation, and `graph_valid=1` in
one savepoint.
