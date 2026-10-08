# SIFT1M embedded HNSW comparison

This report compares three persistent embedded HNSW databases in separate
native processes with identical SIFT vectors, query order, CPU affinity, and
Python measurement code.

The benchmark method is documented in [the benchmark guide](../README.md).

## Environment

- Measured: `2026-10-09`
- Platform: `Linux-5.15.120.bsk.6-sign-amd64-x86_64-with-glibc2.28`
- Python: `3.12.11`
- CPU affinity: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- sqlite-hnsw: `0.4.0`
- SQLite CLI/APSW: `3.51.2` / `3.51.2`
- SQLite CLI: `/home/yzy/.local/bin/sqlite3`
- SQLite prefix: `/home/yzy/.local`
- Packages: `{"chromadb": "1.5.9", "milvus-lite": "3.2.1", "numpy": "2.3.3", "pymilvus": "3.0.2"}`

| Dataset file | Bytes | SHA-256 |
| --- | --- | --- |
| sift_base.fvecs | 516000000 | 21f66e2975057b5728ba56de1c825bac4f4d89d596609ae985741c6242631816 |
| sift_groundtruth.ivecs | 4040000 | 75571702a3e995a67479418940df669952e89d9ad99bc8539c2ba0c6365480af |
| sift_query.fvecs | 5160000 | f7fc9be140accdfd64116c2fa2365ecdb69b8f084970c6b0532db5ff79ac8fdc |

## Verified HNSW configuration

| Engine | Configuration |
| --- | --- |
| sqlite-hnsw | {"ef_construction": 128, "m": 16, "metric": "l2"} |
| chroma | {"ef_construction": 128, "ef_search": 40, "max_neighbors": 16, "resize_factor": 1.2, "space": "l2", "sync_threshold": 1000} |
| milvus-lite | {"field_name": "vector", "index_name": "vector", "index_type": "HNSW", "indexed_rows": 1000000, "metric_type": "L2", "params": {"M": 16, "efConstruction": 128}, "pending_index_rows": 0, "state": "Finished", "total_rows": 1000000} |

## Build

| Engine | Persistent-ready build seconds | Peak RSS MiB |
| --- | --- | --- |
| sqlite-hnsw | 125.601 | 1254.9 |
| chroma | 174.587 | 1634.4 |
| milvus-lite | 120.075 | 2530.1 |

## Physical storage

| Engine | Total MiB | Components MiB |
| --- | --- | --- |
| sqlite-hnsw | 880.9 | hnsw_graph=296.9, other=0.0, row_mapping=24.5, vectors_and_nodes=559.4 |
| chroma | 795.7 | hnsw_graph=664.0, metadata_sqlite=131.7, other=0.0 |
| milvus-lite | 755.4 | hnsw_graph=625.9, metadata_and_wal=0.0, vector_data=129.4 |

## Recall and latency sweep

| Engine | ef | Recall@10 median (range) | p50 ms | p95 ms median (range) | p99 ms | QPS |
| --- | --- | --- | --- | --- | --- | --- |
| sqlite-hnsw | 10 | 0.7850 (0.7850-0.7850) | 0.333 | 0.376 (0.376-0.407) | 0.393 | 2828.04 |
| sqlite-hnsw | 20 | 0.9070 (0.9070-0.9070) | 0.405 | 0.449 (0.449-0.487) | 0.469 | 2381.23 |
| sqlite-hnsw | 30 | 0.9480 (0.9480-0.9480) | 0.471 | 0.530 (0.527-0.584) | 0.558 | 2060.00 |
| sqlite-hnsw | 40 | 0.9720 (0.9720-0.9720) | 0.535 | 0.599 (0.593-0.638) | 0.608 | 1826.99 |
| sqlite-hnsw | 50 | 0.9800 (0.9800-0.9800) | 0.603 | 0.664 (0.661-0.724) | 0.681 | 1643.13 |
| sqlite-hnsw | 60 | 0.9860 (0.9860-0.9860) | 0.662 | 0.733 (0.730-0.768) | 0.746 | 1502.07 |
| sqlite-hnsw | 80 | 0.9910 (0.9910-0.9910) | 0.783 | 0.880 (0.876-0.903) | 0.894 | 1278.36 |
| sqlite-hnsw | 100 | 0.9940 (0.9940-0.9940) | 0.901 | 1.020 (1.015-1.035) | 1.031 | 1121.98 |
| sqlite-hnsw | 150 | 0.9950 (0.9950-0.9950) | 1.191 | 1.359 (1.339-1.372) | 1.385 | 865.44 |
| sqlite-hnsw | 200 | 0.9970 (0.9970-0.9970) | 1.446 | 1.694 (1.678-1.702) | 1.720 | 712.07 |
| chroma | 10 | 0.7360 (0.7360-0.7360) | 0.637 | 0.730 (0.729-0.743) | 0.751 | 1469.47 |
| chroma | 20 | 0.8610 (0.8610-0.8610) | 0.684 | 0.783 (0.756-0.785) | 0.797 | 1381.35 |
| chroma | 30 | 0.9080 (0.9080-0.9080) | 0.723 | 0.814 (0.804-0.822) | 0.858 | 1315.23 |
| chroma | 40 | 0.9390 (0.9390-0.9390) | 0.753 | 0.860 (0.829-0.860) | 0.875 | 1268.10 |
| chroma | 50 | 0.9570 (0.9570-0.9570) | 0.785 | 0.876 (0.864-0.894) | 0.917 | 1224.23 |
| chroma | 60 | 0.9670 (0.9670-0.9670) | 0.835 | 0.932 (0.921-0.933) | 0.977 | 1153.84 |
| chroma | 80 | 0.9730 (0.9730-0.9730) | 0.894 | 0.966 (0.953-0.993) | 1.021 | 1092.35 |
| chroma | 100 | 0.9900 (0.9900-0.9900) | 0.944 | 1.039 (1.020-1.081) | 1.082 | 1036.86 |
| chroma | 150 | 0.9940 (0.9940-0.9940) | 1.086 | 1.196 (1.186-1.266) | 1.242 | 903.47 |
| chroma | 200 | 0.9970 (0.9970-0.9970) | 1.228 | 1.383 (1.363-1.396) | 1.451 | 812.02 |
| milvus-lite | 10 | 0.9230 (0.9230-0.9230) | 692.383 | 703.580 (698.576-708.601) | 709.621 | 1.45 |
| milvus-lite | 20 | 0.9750 (0.9750-0.9750) | 687.968 | 698.474 (694.767-708.081) | 703.029 | 1.45 |
| milvus-lite | 30 | 0.9870 (0.9870-0.9870) | 691.473 | 707.357 (705.716-723.007) | 715.069 | 1.44 |
| milvus-lite | 40 | 0.9900 (0.9900-0.9900) | 694.176 | 708.870 (703.803-709.550) | 710.937 | 1.44 |
| milvus-lite | 50 | 0.9980 (0.9980-0.9980) | 694.871 | 711.465 (707.126-729.108) | 715.057 | 1.44 |
| milvus-lite | 60 | 0.9990 (0.9990-0.9990) | 691.592 | 702.988 (701.472-721.128) | 706.959 | 1.45 |
| milvus-lite | 80 | 1.0000 (1.0000-1.0000) | 691.066 | 709.202 (707.568-712.147) | 712.120 | 1.44 |
| milvus-lite | 100 | 1.0000 (1.0000-1.0000) | 691.373 | 704.565 (702.644-717.934) | 712.257 | 1.44 |
| milvus-lite | 150 | 1.0000 (1.0000-1.0000) | 699.544 | 713.244 (713.104-741.290) | 722.997 | 1.43 |
| milvus-lite | 200 | 1.0000 (1.0000-1.0000) | 697.524 | 715.671 (709.173-1022.713) | 718.197 | 1.43 |

## Recall targets

| Target recall | Engine | ef | Recall@10 | p95 ms | QPS |
| --- | --- | --- | --- | --- | --- |
| 0.95 | sqlite-hnsw | 40 | 0.9720 | 0.599 | 1826.99 |
| 0.95 | chroma | 50 | 0.9570 | 0.876 | 1224.23 |
| 0.95 | milvus-lite | 20 | 0.9750 | 698.474 | 1.45 |
| 0.98 | sqlite-hnsw | 50 | 0.9800 | 0.664 | 1643.13 |
| 0.98 | chroma | 100 | 0.9900 | 1.039 | 1036.86 |
| 0.98 | milvus-lite | 60 | 0.9990 | 702.988 | 1.45 |
| 0.99 | sqlite-hnsw | 80 | 0.9910 | 0.880 | 1278.36 |
| 0.99 | chroma | 100 | 0.9900 | 1.039 | 1036.86 |
| 0.99 | milvus-lite | 60 | 0.9990 | 702.988 | 1.45 |

## Pareto frontier

| Engine | ef | Recall@10 | p95 ms |
| --- | --- | --- | --- |
| sqlite-hnsw | 10 | 0.7850 | 0.376 |
| sqlite-hnsw | 20 | 0.9070 | 0.449 |
| sqlite-hnsw | 30 | 0.9480 | 0.530 |
| sqlite-hnsw | 40 | 0.9720 | 0.599 |
| sqlite-hnsw | 50 | 0.9800 | 0.664 |
| sqlite-hnsw | 60 | 0.9860 | 0.733 |
| sqlite-hnsw | 80 | 0.9910 | 0.880 |
| sqlite-hnsw | 100 | 0.9940 | 1.020 |
| sqlite-hnsw | 150 | 0.9950 | 1.359 |
| sqlite-hnsw | 200 | 0.9970 | 1.694 |
| chroma | 10 | 0.7360 | 0.730 |
| chroma | 20 | 0.8610 | 0.783 |
| chroma | 30 | 0.9080 | 0.814 |
| chroma | 40 | 0.9390 | 0.860 |
| chroma | 50 | 0.9570 | 0.876 |
| chroma | 60 | 0.9670 | 0.932 |
| chroma | 80 | 0.9730 | 0.966 |
| chroma | 100 | 0.9900 | 1.039 |
| chroma | 150 | 0.9940 | 1.196 |
| chroma | 200 | 0.9970 | 1.383 |
| milvus-lite | 20 | 0.9750 | 698.474 |
| milvus-lite | 60 | 0.9990 | 702.988 |
| milvus-lite | 100 | 1.0000 | 704.565 |

## Connection-cold query

| Engine | Median ms | Min ms | Max ms | Peak RSS MiB |
| --- | --- | --- | --- | --- |
| sqlite-hnsw | 2751.743 | 2734.437 | 2758.783 | 683.1 |
| chroma | 1761.573 | 1731.537 | 1763.331 | 1030.2 |
| milvus-lite | 719.549 | 712.729 | 741.794 | 1815.9 |

## Steady-state query memory

| Engine | Peak process-tree RSS MiB |
| --- | --- |
| sqlite-hnsw | 697.9 |
| chroma | 1311.0 |
| milvus-lite | 1848.3 |

## Capability notes

| Engine | Persistence | Transactions/recovery | Filtering | Multi-instance memory |
| --- | --- | --- | --- | --- |
| sqlite-hnsw | Single SQLite file | SQLite transactions and WAL | Post-filter through joined tables | Snapshot is connection-local |
| Chroma | Directory with SQLite metadata and HNSW files | Chroma local persistence semantics | Metadata filtering | Product-managed local index |
| Milvus Lite | Local database file and sidecars | Milvus Lite flush and local persistence | Scalar filtering API | Product-managed local index |
