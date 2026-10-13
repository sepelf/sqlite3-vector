#!/usr/bin/env python3

import argparse
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np


SIFT_DIM = 128
DEFAULT_K = 10


class BenchError(RuntimeError):
    pass


@dataclass
class Dataset:
    dataset_dir: str
    base_path: str
    query_path: str
    groundtruth_path: str
    base_vectors: np.ndarray
    query_vectors: np.ndarray
    groundtruth: np.ndarray
    base_file_vectors: int
    query_file_vectors: int
    groundtruth_file_vectors: int
    groundtruth_width: int
    base_limit: int
    query_limit: int


@dataclass
class QueryStats:
    latencies_ms: list[float]
    query_seconds_total: float
    recall_at_1: float
    recall_at_10: float


def now_sec() -> float:
    return time.perf_counter()


def require_pysqlite3():
    try:
        import pysqlite3
    except ImportError as exc:
        raise BenchError(
            "pysqlite3-binary is required for sqlite3-vector and Chroma "
            "benchmarks; install bench/requirements-sift-compare.txt"
        ) from exc
    sys.modules["sqlite3"] = pysqlite3
    return pysqlite3


def require_module(import_name: str, package_name: str | None = None):
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise BenchError(
            f"{package_name or import_name} is required; install "
            "bench/requirements-sift-compare.txt"
        ) from exc


def remove_path(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def path_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def checked_mmap_i32(path: str) -> np.memmap:
    if not os.path.exists(path):
        raise BenchError(f"missing dataset file: {path}")
    if os.path.getsize(path) % 4 != 0:
        raise BenchError(f"file size is not 4-byte aligned: {path}")
    return np.memmap(path, dtype="<i4", mode="r")


def fvecs_info(path: str) -> tuple[int, int]:
    raw = checked_mmap_i32(path)
    if raw.size == 0:
        raise BenchError(f"empty fvecs file: {path}")
    dim = int(raw[0])
    if dim <= 0:
        raise BenchError(f"invalid fvecs dimension {dim}: {path}")
    record_width = dim + 1
    if raw.size % record_width != 0:
        raise BenchError(f"fvecs file is not aligned to dim {dim}: {path}")
    count = raw.size // record_width
    dims = raw.reshape(count, record_width)[:, 0]
    if not np.all(dims == dim):
        raise BenchError(f"inconsistent fvecs dimensions: {path}")
    return int(count), dim


def ivecs_info(path: str) -> tuple[int, int]:
    raw = checked_mmap_i32(path)
    if raw.size == 0:
        raise BenchError(f"empty ivecs file: {path}")
    width = int(raw[0])
    if width <= 0:
        raise BenchError(f"invalid ivecs width {width}: {path}")
    record_width = width + 1
    if raw.size % record_width != 0:
        raise BenchError(f"ivecs file is not aligned to width {width}: {path}")
    count = raw.size // record_width
    widths = raw.reshape(count, record_width)[:, 0]
    if not np.all(widths == width):
        raise BenchError(f"inconsistent ivecs widths: {path}")
    return int(count), width


def open_fvecs(path: str) -> tuple[np.ndarray, int, int]:
    count, dim = fvecs_info(path)
    raw = np.memmap(path, dtype="<f4", mode="r")
    vectors = raw.reshape(count, dim + 1)[:, 1:]
    return vectors, count, dim


def load_groundtruth(path: str, query_limit: int, k: int) -> tuple[np.ndarray, int, int]:
    count, width = ivecs_info(path)
    if width < k:
        raise BenchError(f"groundtruth width {width} is smaller than k={k}")
    raw = np.memmap(path, dtype="<i4", mode="r")
    rows = raw.reshape(count, width + 1)
    topk = np.ascontiguousarray(rows[:query_limit, 1 : 1 + k], dtype=np.int64)
    return topk, count, width


def load_dataset(dataset_dir: str, base_limit: int, query_limit: int, k: int) -> Dataset:
    base_path = os.path.join(dataset_dir, "sift_base.fvecs")
    query_path = os.path.join(dataset_dir, "sift_query.fvecs")
    groundtruth_path = os.path.join(dataset_dir, "sift_groundtruth.ivecs")

    base_vectors, base_count, base_dim = open_fvecs(base_path)
    query_vectors, query_count, query_dim = open_fvecs(query_path)
    if base_dim != SIFT_DIM or query_dim != SIFT_DIM:
        raise BenchError(
            f"expected SIFT dim {SIFT_DIM}, got base={base_dim} query={query_dim}"
        )

    gt_count, gt_width = ivecs_info(groundtruth_path)
    base_limit = min(base_limit, base_count)
    query_limit = min(query_limit, query_count, gt_count)
    groundtruth, _, _ = load_groundtruth(groundtruth_path, query_limit, k)
    query_vectors = np.ascontiguousarray(query_vectors[:query_limit], dtype=np.float32)
    groundtruth = groundtruth[:query_limit]

    return Dataset(
        dataset_dir=dataset_dir,
        base_path=base_path,
        query_path=query_path,
        groundtruth_path=groundtruth_path,
        base_vectors=base_vectors,
        query_vectors=query_vectors,
        groundtruth=groundtruth,
        base_file_vectors=base_count,
        query_file_vectors=query_count,
        groundtruth_file_vectors=gt_count,
        groundtruth_width=gt_width,
        base_limit=base_limit,
        query_limit=query_limit,
    )


def iter_batches(vectors: np.ndarray, limit: int, batch_size: int) -> Iterable[tuple[int, np.ndarray]]:
    for start in range(0, limit, batch_size):
        end = min(start + batch_size, limit)
        yield start, np.ascontiguousarray(vectors[start:end], dtype=np.float32)


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(math.ceil(len(sorted_values) * p)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return sorted_values[idx]


def compute_query_stats(
    queries: np.ndarray,
    groundtruth: np.ndarray,
    base_limit: int,
    k: int,
    search_one: Callable[[np.ndarray], list[int]],
    progress_label: str,
    progress_interval: int,
) -> QueryStats:
    latencies_ms: list[float] = []
    recall_at_1 = 0.0
    recall_at_10 = 0.0
    query_t0 = now_sec()
    total = len(queries)

    for i, query in enumerate(queries):
        t0 = now_sec()
        ids = search_one(query)
        latencies_ms.append((now_sec() - t0) * 1000.0)
        if ids and ids[0] == int(groundtruth[i, 0]):
            recall_at_1 += 1.0
        truth = set(int(x) for x in groundtruth[i, :k])
        for id_value in ids[:k]:
            if 0 <= id_value < base_limit and id_value in truth:
                recall_at_10 += 1.0
        done = i + 1
        if progress_interval > 0 and (
            done == 1 or done % progress_interval == 0 or done == total
        ):
            elapsed = now_sec() - query_t0
            qps = done / elapsed if elapsed > 0.0 else 0.0
            print(
                "# query progress "
                f"engine={progress_label} "
                f"completed={done} "
                f"remaining={total - done} "
                f"total={total} "
                f"elapsed_seconds={elapsed:.3f} "
                f"qps={qps:.3f}",
                file=sys.stderr,
                flush=True,
            )

    query_seconds_total = now_sec() - query_t0
    latencies_ms.sort()
    count = total
    return QueryStats(
        latencies_ms=latencies_ms,
        query_seconds_total=query_seconds_total,
        recall_at_1=recall_at_1 / count if count else 0.0,
        recall_at_10=recall_at_10 / (count * k) if count else 0.0,
    )


def base_result(args: argparse.Namespace, data: Dataset, engine: str, path: str) -> dict:
    return {
        "engine": engine,
        "dataset": "sift1m",
        "dataset_dir": data.dataset_dir,
        "database": path,
        "base_file_vectors": data.base_file_vectors,
        "query_file_vectors": data.query_file_vectors,
        "groundtruth_file_vectors": data.groundtruth_file_vectors,
        "groundtruth_width": data.groundtruth_width,
        "base_vectors": data.base_limit,
        "query_vectors": data.query_limit,
        "dim": SIFT_DIM,
        "metric": "l2",
        "m": args.m,
        "ef_construction": args.ef_construction,
        "ef_search": args.ef_search,
        "k": args.k,
        "insert_batch_size": args.insert_batch_size,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def add_query_fields(result: dict, stats: QueryStats) -> None:
    result["query_seconds_total"] = stats.query_seconds_total
    result["query_ms_p50"] = percentile(stats.latencies_ms, 0.50)
    result["query_ms_p95"] = percentile(stats.latencies_ms, 0.95)
    result["query_ms_p99"] = percentile(stats.latencies_ms, 0.99)
    result["qps"] = len(stats.latencies_ms) / stats.query_seconds_total
    result["recall_at_1"] = stats.recall_at_1
    result["recall_at_10"] = stats.recall_at_10
    result["recall_scope"] = "full_groundtruth_top10"


def run_sqlite(args: argparse.Namespace, data: Dataset) -> dict:
    sqlite3 = require_pysqlite3()
    db_path = os.path.join(args.work_dir, "sqlite_sift1m.db")
    remove_path(db_path)
    extension_path = os.path.abspath(args.sqlite_extension)
    if not os.path.exists(extension_path):
        raise BenchError(f"sqlite extension not found: {extension_path}")

    ready_t0 = now_sec()
    setup_t0 = now_sec()
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(extension_path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        "CREATE VIRTUAL TABLE sift_vec USING vec_hnsw("
        f"dim={SIFT_DIM}, metric=l2, m={args.m}, "
        f"ef_construction={args.ef_construction}, ef_search={args.ef_search}, "
        f"build_threads={args.sqlite_build_threads}, "
        f"build_batch_size={args.sqlite_build_batch_size}, seed=123)"
    )
    setup_seconds = now_sec() - setup_t0

    load_t0 = now_sec()
    conn.execute("BEGIN")
    insert_sql = "INSERT INTO sift_vec(rowid, embedding) VALUES (?, ?)"
    for start, batch in iter_batches(data.base_vectors, data.base_limit, args.insert_batch_size):
        rows = ((start + i + 1, row.tobytes()) for i, row in enumerate(batch))
        conn.executemany(insert_sql, rows)
    conn.execute("COMMIT")
    load_seconds = now_sec() - load_t0

    index_t0 = now_sec()
    conn.execute("SELECT vec_hnsw_rebuild('main', 'sift_vec')").fetchall()
    index_seconds = now_sec() - index_t0
    ready_seconds = now_sec() - ready_t0

    query_sql = (
        "SELECT rowid FROM sift_vec "
        f"WHERE embedding MATCH ?1 AND k = {args.k} AND ef_search = ?2 "
        "ORDER BY distance"
    )

    def search_one(query: np.ndarray) -> list[int]:
        rows = conn.execute(query_sql, (query.tobytes(), args.ef_search)).fetchall()
        return [int(row[0]) - 1 for row in rows]

    stats = compute_query_stats(
        data.query_vectors,
        data.groundtruth,
        data.base_limit,
        args.k,
        search_one,
        "sqlite3-vector",
        args.query_progress_interval,
    )
    conn.close()

    result = base_result(args, data, "sqlite3-vector", db_path)
    result.update(
        {
            "sqlite_extension": extension_path,
            "sqlite_build_threads": args.sqlite_build_threads,
            "sqlite_build_batch_size": args.sqlite_build_batch_size,
            "index_mode": "explicit_rebuild",
            "setup_seconds": setup_seconds,
            "load_seconds": load_seconds,
            "index_seconds": index_seconds,
            "ready_seconds": ready_seconds,
            "total_seconds": ready_seconds,
            "database_bytes": path_size(db_path),
        }
    )
    add_query_fields(result, stats)
    return result


def run_chroma(args: argparse.Namespace, data: Dataset) -> dict:
    require_pysqlite3()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    import chromadb
    from chromadb.config import Settings

    db_path = os.path.join(args.work_dir, "chroma_sift1m")
    remove_path(db_path)

    ready_t0 = now_sec()
    setup_t0 = now_sec()
    client = chromadb.PersistentClient(
        path=db_path, settings=Settings(anonymized_telemetry=False)
    )
    collection = client.create_collection(
        name="sift_vec",
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "l2",
                "max_neighbors": args.m,
                "ef_construction": args.ef_construction,
                "ef_search": args.ef_search,
            }
        },
    )
    setup_seconds = now_sec() - setup_t0

    load_t0 = now_sec()
    for start, batch in iter_batches(data.base_vectors, data.base_limit, args.insert_batch_size):
        end = start + len(batch)
        collection.add(
            ids=[str(i) for i in range(start, end)],
            embeddings=batch.tolist(),
        )
    load_seconds = now_sec() - load_t0
    ready_seconds = now_sec() - ready_t0

    def search_one(query: np.ndarray) -> list[int]:
        result = collection.query(query_embeddings=[query.tolist()], n_results=args.k)
        return [int(id_value) for id_value in result["ids"][0]]

    stats = compute_query_stats(
        data.query_vectors,
        data.groundtruth,
        data.base_limit,
        args.k,
        search_one,
        "chroma",
        args.query_progress_interval,
    )

    result = base_result(args, data, "chroma", db_path)
    result.update(
        {
            "chroma_version": chromadb.__version__,
            "index_mode": "online_during_load",
            "setup_seconds": setup_seconds,
            "load_seconds": load_seconds,
            "index_seconds": 0.0,
            "ready_seconds": ready_seconds,
            "total_seconds": ready_seconds,
            "database_bytes": path_size(db_path),
        }
    )
    add_query_fields(result, stats)
    return result


def run_milvus_lite(args: argparse.Namespace, data: Dataset) -> dict:
    pymilvus = require_module("pymilvus")
    from pymilvus import DataType, MilvusClient

    db_path = os.path.join(args.work_dir, "milvus_lite_sift1m.db")
    remove_path(db_path)

    ready_t0 = now_sec()
    setup_t0 = now_sec()
    client = MilvusClient(uri=db_path)
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=SIFT_DIM)
    client.create_collection(collection_name="sift_vec", schema=schema)
    setup_seconds = now_sec() - setup_t0

    try:
        load_t0 = now_sec()
        for start, batch in iter_batches(
            data.base_vectors, data.base_limit, args.insert_batch_size
        ):
            payload = [
                {"id": start + i, "vector": vector}
                for i, vector in enumerate(batch.tolist())
            ]
            client.insert(collection_name="sift_vec", data=payload)
        client.flush(collection_name="sift_vec")
        load_seconds = now_sec() - load_t0

        index_t0 = now_sec()
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="HNSW",
            metric_type="L2",
            params={"M": args.m, "efConstruction": args.ef_construction},
        )
        client.create_index(collection_name="sift_vec", index_params=index_params)
        client.load_collection("sift_vec")
        index_seconds = now_sec() - index_t0
        ready_seconds = now_sec() - ready_t0

        def search_one(query: np.ndarray) -> list[int]:
            result = client.search(
                collection_name="sift_vec",
                data=[query.tolist()],
                anns_field="vector",
                limit=args.k,
                search_params={"metric_type": "L2", "params": {"ef": args.ef_search}},
                output_fields=["id"],
            )
            return [int(hit["id"]) for hit in result[0]]

        stats = compute_query_stats(
            data.query_vectors,
            data.groundtruth,
            data.base_limit,
            args.k,
            search_one,
            "milvus-lite",
            args.query_progress_interval,
        )
    finally:
        try:
            client.close()
        except Exception:
            pass

    result = base_result(args, data, "milvus-lite", db_path)
    result.update(
        {
            "pymilvus_version": pymilvus.__version__,
            "index_mode": "explicit_create_index",
            "setup_seconds": setup_seconds,
            "load_seconds": load_seconds,
            "index_seconds": index_seconds,
            "ready_seconds": ready_seconds,
            "total_seconds": ready_seconds,
            "database_bytes": path_size(db_path),
        }
    )
    add_query_fields(result, stats)
    return result


def print_result(result: dict) -> None:
    order = [
        "engine",
        "dataset",
        "dataset_dir",
        "database",
        "base_file_vectors",
        "query_file_vectors",
        "groundtruth_file_vectors",
        "groundtruth_width",
        "base_vectors",
        "query_vectors",
        "dim",
        "metric",
        "m",
        "ef_construction",
        "ef_search",
        "k",
        "insert_batch_size",
        "index_mode",
        "setup_seconds",
        "load_seconds",
        "index_seconds",
        "ready_seconds",
        "total_seconds",
        "query_seconds_total",
        "query_ms_p50",
        "query_ms_p95",
        "query_ms_p99",
        "qps",
        "recall_at_1",
        "recall_at_10",
        "recall_scope",
        "database_bytes",
    ]
    for key in order:
        if key in result:
            value = result[key]
            if isinstance(value, float):
                if key.startswith("recall"):
                    print(f"{key}={value:.4f}")
                else:
                    print(f"{key}={value:.3f}")
            else:
                print(f"{key}={value}")
    for key in sorted(k for k in result if k not in set(order)):
        print(f"{key}={result[key]}")
    print()


def print_summary(results: list[dict]) -> None:
    if not results:
        return
    columns = [
        "engine",
        "total_seconds",
        "load_seconds",
        "index_seconds",
        "query_ms_p50",
        "query_ms_p95",
        "qps",
        "recall_at_1",
        "recall_at_10",
        "database_bytes",
    ]
    print("summary_csv=" + ",".join(columns))
    for result in results:
        values = []
        for column in columns:
            value = result[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        print("summary_row=" + ",".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sqlite3-vector, Chroma, and Milvus Lite on SIFT1M."
    )
    parser.add_argument(
        "--engine",
        choices=["all", "sqlite", "chroma", "milvus-lite"],
        default="all",
        help="benchmark engine to run",
    )
    parser.add_argument("--dataset-dir", default="/tmp/sift1m-fvecs")
    parser.add_argument("--base-limit", type=int, default=1_000_000)
    parser.add_argument("--query-limit", type=int, default=10_000)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--insert-batch-size", type=int, default=5_000)
    parser.add_argument(
        "--query-progress-interval",
        type=int,
        default=100,
        help="print query progress every N completed queries; 0 disables progress",
    )
    parser.add_argument("--work-dir", default="/tmp/vechnsw_sift_compare")
    parser.add_argument("--sqlite-extension", default="build/vechnsw.so")
    parser.add_argument("--sqlite-build-threads", type=int, default=8)
    parser.add_argument("--sqlite-build-batch-size", type=int, default=512)
    args = parser.parse_args()

    positive_fields = [
        "base_limit",
        "query_limit",
        "k",
        "m",
        "ef_construction",
        "ef_search",
        "insert_batch_size",
        "sqlite_build_threads",
        "sqlite_build_batch_size",
    ]
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.sqlite_build_threads < 2:
        parser.error("--sqlite-build-threads must be >= 2")
    if args.sqlite_build_batch_size < 2:
        parser.error("--sqlite-build-batch-size must be >= 2")
    if args.query_progress_interval < 0:
        parser.error("--query-progress-interval must be >= 0")
    return args


def main() -> int:
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    engines = (
        ["sqlite", "chroma", "milvus-lite"]
        if args.engine == "all"
        else [args.engine]
    )
    runners = {
        "sqlite": run_sqlite,
        "chroma": run_chroma,
        "milvus-lite": run_milvus_lite,
    }

    try:
        data = load_dataset(args.dataset_dir, args.base_limit, args.query_limit, args.k)
        results = []
        for engine in engines:
            print(f"# running {engine}", flush=True)
            result = runners[engine](args, data)
            print_result(result)
            results.append(result)
        print_summary(results)
        return 0
    except BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
