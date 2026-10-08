# sqlite-hnsw architecture

This document describes the internal storage and execution model. Public SQL
behavior is defined in [reference.md](reference.md).

## Scope

The extension is an index-like virtual table with one fixed-width `float32`
vector per row. All durable state lives in SQLite shadow tables; there are no
sidecar index files, background workers, or process-global mutable indexes.

## Durable layout

For a virtual table named `items_hnsw`, format version 1 creates:

- `items_hnsw_meta`: configuration, entry point, counters, next node id, and
  change sequence;
- `items_hnsw_rows`: external `rowid` to internal node-id mapping;
- `items_hnsw_nodes`: vector BLOB, norm, maximum level, and deletion marker;
- `items_hnsw_edges`: one encoded neighbor list per node and layer;
- `items_hnsw_rebuild`: staging vectors used by `optimize`.

External rowids and internal node ids are independent. Updates map the external
rowid to a new node and leave the old node traversable but deleted. `optimize`
rebuilds the graph with dense internal ids starting at 1.

Edge BLOBs contain a little-endian 32-bit count followed by 64-bit internal
node ids. The layout is private and is validated before use.

## Mutation and graph construction

A new node is inserted with its level encoded as `-level-2`. SQLite's `xSync`
callback links every pending node immediately before commit and replaces the
encoded level with its normal non-negative value. A successful commit cannot
leave pending nodes.

Incremental construction performs normal HNSW descent, candidate search,
neighbor selection, and backlink pruning while buffering dirty edge lists in a
connection-local cache.

When an empty graph has at least 4,096 pending nodes, `xSync` uses a contiguous
arena. The first 1,024 nodes seed the graph sequentially. Later nodes are
searched in deterministic 64-node batches by up to 16 workers. Workers operate
only on immutable arena snapshots and never call SQLite. Backlinks are grouped
by owner and applied before exposing the next snapshot; the connection thread
persists the completed graph in node-id order.

The arena stores float32 vectors, uint32 indices, fixed-capacity adjacency, and
one visited-epoch array per worker. Estimation or allocation failure aborts the
commit.

## Search and derived state

HNSW search performs greedy descent above layer 0, then an `ef_search`
candidate expansion. Deleted nodes remain valid navigation points but cannot
enter the result heap.

The page-backed path uses shared-budget node and edge LRUs. Cache entries are
pinned while borrowed, and dirty edge entries are flushed before a graph state
is committed.

If the complete graph fits the connection budget, search builds a compact
snapshot with contiguous vectors, direct norm and rowid arrays, uint32 neighbor
ids, and precomputed node/layer offsets. A dense visited-epoch array replaces
the hash set. The snapshot is derived state and is never persisted.

`change_seq` invalidates cached state after another connection commits. A
writer clears its snapshot, uses the page-backed graph, and exact-scans pending
nodes to provide read-your-writes.

## Transactions and failure handling

Every mutation is an ordinary write to shadow tables on the same SQLite
connection. `xSync` completes graph construction; `xCommit` discards derived
write state; rollback and rollback-to-savepoint clear statements, caches, and
snapshots so SQLite can restore the durable rows.

SQLite remains the concurrency and recovery layer: one writer, WAL visibility
for readers, statement atomicity, rollback journals or WAL recovery, and VFS
durability rules all apply unchanged.

Malformed metadata, vectors, nodes, mappings, or edges encountered by a query
produce `SQLITE_CORRUPT_VTAB`. The explicit integrity checker scans all durable
records, constructs a node-id hash, and retains layer-0 adjacency for its
reachability diagnostic. It detects logical format damage but does not inspect
SQLite pages or repair records.

## Internal constraints

- Node levels are capped at 32 and vector dimensions at 4,096.
- Compact snapshots require internal node ids to fit in uint32.
- Connection caches are private; multiple connections duplicate derived state.
- SQLite page-cache and transient construction memory are outside the decoded
  cache budget.
- Directed layer-0 reachability is diagnostic because backlink pruning does
  not guarantee that every stored node remains reachable.

