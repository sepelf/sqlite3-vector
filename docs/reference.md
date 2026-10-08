# sqlite-hnsw reference

This document is the authoritative reference for the public SQL interface and
operational behavior. See [architecture.md](architecture.md) for implementation
details.

## Vectors and scalar functions

Production vectors are non-empty little-endian IEEE-754 `float32` BLOBs. Every
value must be finite, the width must match the virtual table declaration, and
cosine vectors must have non-zero norm.

The extension registers these scalar functions:

| Function | Result |
| --- | --- |
| `hnsw_f32('[1,2,3]')` | Encode a JSON-style array for shells and tests |
| `hnsw_dims(vector)` | Number of float32 dimensions |
| `hnsw_distance_l2(a,b)` | Euclidean distance |
| `hnsw_distance_cosine(a,b)` | Cosine distance |
| `hnsw_distance_ip(a,b)` | Negative inner product |
| `hnsw_version()` | Extension version |
| `hnsw_info(schema,table)` | Index state and cache estimate as JSON |
| `hnsw_check(schema,table)` | Deep logical integrity result as JSON |

## Creating an index

```sql
CREATE VIRTUAL TABLE item_hnsw USING hnsw(
  embedding FLOAT32(768),
  metric=cosine,
  m=16,
  ef_construction=128,
  cache_size_mb=256,
  build_memory_mb=1024,
  build_threads=0
);
```

Exactly one vector column is supported.

| Option | Default | Accepted values | Meaning |
| --- | ---: | --- | --- |
| `metric` | `l2` | `l2`, `cosine`, `ip` | Distance function |
| `m` | `16` | `2..64` | HNSW neighbor parameter |
| `ef_construction` | `128` | `4..1000`, at least `2*m` | Construction search width |
| `cache_size_mb` | `256` | `0..65536` | Per-connection decoded cache or snapshot budget |
| `build_memory_mb` | `1024` | `128..16384` | Empty-index arena memory limit |
| `build_threads` | `0` | `0..64` | Arena workers; `0` selects up to 16 online CPUs |

The declared virtual table exposes the vector column plus these hidden columns:
`distance`, `k`, `ef_search`, `visited_count`, and `command`.

## Writing vectors

The virtual-table `rowid` is the application key. The application owns the
relationship between its content table and the HNSW table and should change
both in one transaction.

```sql
INSERT INTO item_hnsw(rowid, embedding) VALUES (?, ?);
UPDATE item_hnsw SET embedding = ? WHERE rowid = ?;
DELETE FROM item_hnsw WHERE rowid = ?;
```

Inserts are staged until SQLite calls the virtual table's sync callback before
commit. A writer can query its own transaction: linked nodes are searched
normally and pending nodes are scanned exactly. Rollback, savepoints, WAL
visibility, and crash recovery follow SQLite transaction semantics.

Updates and deletes leave traversable tombstones until the index is rebuilt.
SQLite permits one writer at a time; WAL mode is recommended when readers must
continue during writes.

## Nearest-neighbour queries

An ANN plan requires both `MATCH` and an equality constraint on `k`:

```sql
SELECT rowid, distance, visited_count
FROM item_hnsw
WHERE embedding MATCH :query
  AND k = 10
  AND ef_search = 100
ORDER BY distance;
```

- `k` must be between 1 and 10,000.
- `ef_search` defaults to `max(k,64)` and must be between `k` and 1,000,000.
- `visited_count` is the number of distance evaluations.

Search uses HNSW greedy descent and layer-0 candidate expansion. L2 results are
returned as Euclidean distance; inner product uses negative inner product so
smaller remains better.

## Caching and construction

Decoded nodes and edges use a connection-local LRU within `cache_size_mb`. If a
complete graph fits, the first ANN query builds a compact connection-local
snapshot and subsequent traversal avoids SQLite point lookups. A committed
write invalidates the snapshot, and the next query rebuilds it synchronously.
Use a small number of long-lived query connections for large snapshots.

When an empty index receives at least 4,096 rows in one transaction, commit
uses the bounded multi-threaded build arena. If the estimated arena exceeds
`build_memory_mb`, commit fails instead of falling back to an unbounded build.
Smaller transactions and additions to an existing graph use incremental
single-threaded construction.

## Maintenance and integrity

Inspect an index with:

```sql
SELECT hnsw_info('main', 'item_hnsw');
SELECT hnsw_check('main', 'item_hnsw');
```

`hnsw_info` reports live and tombstone counts, pending nodes, change
sequence, estimated snapshot bytes, snapshot eligibility, and whether
`optimize` is recommended. The recommendation accounts for tombstones and
unused internal node-id space.

`hnsw_check` validates metadata, row mappings, vectors and norms, node state,
all adjacency lists, degree and layer limits, neighbor references, and rebuild
residue. It reports checked counts, directed layer-0 reachability, and the first
logical error. Stable error codes are:

- metadata/state: `META_MISSING`, `META_INVALID`, `STATE_COUNT_MISMATCH`,
  `STATE_ENTRY_INVALID`;
- row/node: `ROW_NODE_MISSING`, `ROW_NODE_DELETED`, `LIVE_NODE_UNMAPPED`,
  `NODE_ID_INVALID`, `NODE_LEVEL_INVALID`, `NODE_VECTOR_INVALID`,
  `NODE_NORM_INVALID`, `PENDING_NODE`;
- graph: `EDGE_OWNER_MISSING`, `EDGE_ROW_MISSING`, `EDGE_LAYER_INVALID`,
  `EDGE_BLOB_INVALID`, `EDGE_DEGREE_EXCEEDED`, `EDGE_SELF_REFERENCE`,
  `EDGE_DUPLICATE_NEIGHBOR`, `EDGE_NEIGHBOR_MISSING`,
  `EDGE_NEIGHBOR_LAYER_INVALID`;
- rebuild state: `REBUILD_NOT_EMPTY`.

Reachability is a graph-quality diagnostic and does not by itself make the
check fail. Pair the logical check with `PRAGMA integrity_check` for SQLite page
and B-tree integrity. Do not continue using an index after either check fails;
rebuild it from the source vectors.

Remove tombstones and compact internal node ids with:

```sql
INSERT INTO item_hnsw(command) VALUES ('optimize');
```

The rebuild is part of the surrounding transaction and can temporarily require
database or WAL space for another copy of all live vectors.

## Filtering and limits

The extension has no metadata, partition, or filter columns. Predicates on a
joined application table run after ANN candidate selection. Over-fetch, then
apply the predicate and final `LIMIT`; post-filtering is not guaranteed to
produce `k` rows.

The current format and API are pre-1.0. There is no automatic migration or
repair path, no background maintenance, and no shared in-process index between
connections. Cache budgets exclude SQLite's page cache and transient
construction allocations.

