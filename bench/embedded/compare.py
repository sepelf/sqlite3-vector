#!/usr/bin/env python3
"""SIFT1M comparison for embedded persistent HNSW databases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np


DIMS = 128
ROWS = 1_000_000
QUERIES = 100
WARMUP = 10
K = 10
SEED = 20261008
EFS = (10, 20, 30, 40, 50, 60, 80, 100, 150, 200)
ENGINES = ("sqlite-hnsw", "chroma", "milvus-lite")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def append_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_fvecs(path: Path, count: int) -> np.ndarray:
    dtype = np.dtype([("dims", "<u4"), ("values", "<f4", (DIMS,))])
    available = path.stat().st_size // dtype.itemsize
    if available < count or path.stat().st_size % dtype.itemsize != 0:
        raise ValueError(f"{path} does not contain {count} complete vectors")
    records = np.memmap(path, dtype=dtype, mode="r", shape=(available,))
    if not np.all(records["dims"][:count] == DIMS):
        raise ValueError(f"{path} contains a vector with unexpected dimensions")
    return records["values"][:count]


def load_truth(path: Path, count: int) -> np.ndarray:
    width = 100
    dtype = np.dtype([("width", "<u4"), ("values", "<i4", (width,))])
    available = path.stat().st_size // dtype.itemsize
    if available < count or path.stat().st_size % dtype.itemsize != 0:
        raise ValueError(f"{path} does not contain {count} complete records")
    records = np.memmap(path, dtype=dtype, mode="r", shape=(available,))
    if not np.all(records["width"][:count] == width):
        raise ValueError(f"{path} contains ground truth with unexpected width")
    return records["values"][:count, :K]


def dataset_paths(dataset: Path) -> tuple[Path, Path, Path]:
    return (
        dataset / "sift_base.fvecs",
        dataset / "sift_query.fvecs",
        dataset / "sift_groundtruth.ivecs",
    )


def validate_dataset(dataset: Path) -> dict[str, Any]:
    base, queries, truth = dataset_paths(dataset)
    load_fvecs(base, ROWS)
    load_fvecs(queries, QUERIES + WARMUP)
    load_truth(truth, QUERIES)
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in (base, queries, truth)
    }


def storage_files(paths: Iterable[Path]) -> dict[str, int]:
    result: dict[str, int] = {}
    for path in paths:
        if path.is_file():
            result[str(path)] = path.stat().st_size
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file():
                    result[str(child)] = child.stat().st_size
    return result


def sqlite_storage(database: Path) -> dict[str, Any]:
    import apsw

    connection = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        pages = dict(
            connection.execute(
                "SELECT name,sum(pgsize) FROM dbstat GROUP BY name"
            )
        )
        page_size = int(next(connection.execute("PRAGMA page_size"))[0])
        freelist = int(next(connection.execute("PRAGMA freelist_count"))[0])
    finally:
        connection.close()
    nodes = int(pages.get("sift_hnsw_nodes", 0))
    edges = int(pages.get("sift_hnsw_edges", 0))
    mapping = int(pages.get("sift_hnsw_rows", 0)) + int(
        pages.get("sqlite_autoindex_sift_hnsw_rows_1", 0)
    )
    total = database.stat().st_size
    return {
        "total_bytes": total,
        "components": {
            "vectors_and_nodes": nodes,
            "hnsw_graph": edges,
            "row_mapping": mapping,
            "other": total - nodes - edges - mapping,
        },
        "freelist_bytes": freelist * page_size,
    }


def chroma_storage(directory: Path) -> dict[str, Any]:
    files = storage_files((directory,))
    metadata = sum(size for name, size in files.items() if name.endswith("chroma.sqlite3"))
    hnsw = sum(
        size
        for name, size in files.items()
        if Path(name).name in {"data_level0.bin", "header.bin", "length.bin", "link_lists.bin"}
        or Path(name).suffix in {".pickle", ".bin"}
    )
    total = sum(files.values())
    return {
        "total_bytes": total,
        "components": {
            "metadata_sqlite": metadata,
            "hnsw_graph": hnsw,
            "other": total - metadata - hnsw,
        },
        "file_count": len(files),
    }


def milvus_storage(database: Path) -> dict[str, Any]:
    files = storage_files((database,))
    data = sum(size for name, size in files.items() if name.endswith(".parquet"))
    hnsw = sum(size for name, size in files.items() if name.endswith(".hnsw.idx"))
    total = sum(files.values())
    return {
        "total_bytes": total,
        "components": {
            "vector_data": data,
            "hnsw_graph": hnsw,
            "metadata_and_wal": total - data - hnsw,
        },
        "file_count": len(files),
    }


def chroma_client(path: Path) -> Any:
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )


def stop_chroma(client: Any) -> None:
    client.close()


def create_milvus_collection(client: Any, name: str) -> None:
    from pymilvus import DataType

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(
        field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMS
    )
    client.create_collection(collection_name=name, schema=schema)


def create_milvus_hnsw(client: Any, name: str) -> dict[str, Any]:
    params = client.prepare_index_params()
    params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type="L2",
        params={"params": {"M": 16, "efConstruction": 128}},
    )
    client.create_index(collection_name=name, index_params=params)
    indexes = client.list_indexes(name)
    if len(indexes) != 1:
        raise RuntimeError(f"Milvus Lite returned unexpected indexes: {indexes}")
    details = client.describe_index(name, indexes[0])
    if details.get("index_type") != "HNSW":
        raise RuntimeError(f"Milvus Lite did not create HNSW: {details}")
    if details.get("params") != {"M": 16, "efConstruction": 128}:
        raise RuntimeError(f"Milvus Lite HNSW parameters mismatch: {details}")
    return details


def validate(args: argparse.Namespace) -> None:
    import apsw
    import sqlite3

    expected = args.sqlite_version
    cli = subprocess.run(
        [args.sqlite_cli, "--version"], check=True, text=True, capture_output=True
    ).stdout.split()[0]
    if cli != expected or sqlite3.sqlite_version != expected or apsw.sqlitelibversion() != expected:
        raise RuntimeError(
            f"SQLite version mismatch: cli={cli}, stdlib={sqlite3.sqlite_version}, "
            f"APSW={apsw.sqlitelibversion()}, expected={expected}"
        )
    metadata = {
        "dataset": validate_dataset(args.dataset),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "sqlite": {
            "expected": expected,
            "cli": cli,
            "stdlib": sqlite3.sqlite_version,
            "apsw": apsw.sqlitelibversion(),
            "apsw_version": apsw.apswversion(),
            "cli_path": args.sqlite_cli,
            "prefix": args.sqlite_prefix,
        },
        "packages": {
            name: distribution_version(name)
            for name in ("chromadb", "pymilvus", "milvus-lite", "numpy")
        },
    }

    probe = args.work / "capability"
    if probe.exists():
        shutil.rmtree(probe)
    probe.mkdir(parents=True)
    chroma = chroma_client(probe / "chroma")
    try:
        collection = chroma.create_collection(
            "probe",
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "l2",
                    "ef_construction": 128,
                    "max_neighbors": 16,
                    "ef_search": 40,
                }
            },
        )
        hnsw = collection.configuration["hnsw"]
        if (hnsw["space"], hnsw["ef_construction"], hnsw["max_neighbors"]) != (
            "l2",
            128,
            16,
        ):
            raise RuntimeError(f"Chroma HNSW configuration mismatch: {hnsw}")
        collection.modify(configuration={"hnsw": {"ef_search": 80}})
        if collection.configuration["hnsw"]["ef_search"] != 80:
            raise RuntimeError("Chroma ef_search is not mutable")
    finally:
        stop_chroma(chroma)

    from pymilvus import MilvusClient

    milvus = MilvusClient(str(probe / "milvus.db"))
    try:
        create_milvus_collection(milvus, "probe")
        milvus.insert(
            "probe",
            [{"id": 0, "vector": [0.0] * DIMS}, {"id": 1, "vector": [1.0] * DIMS}],
        )
        milvus.flush("probe")
        metadata["milvus_hnsw_probe"] = create_milvus_hnsw(milvus, "probe")
    finally:
        milvus.close()
    shutil.rmtree(probe)
    write_json(args.output, metadata)


def prepare_sqlite(args: argparse.Namespace) -> dict[str, Any]:
    import apsw

    database = args.work / "sqlite-hnsw.db"
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(database) + suffix)
        if candidate.exists():
            candidate.unlink()
    vectors = load_fvecs(dataset_paths(args.dataset)[0], ROWS)
    started = time.perf_counter()
    connection = apsw.Connection(str(database))
    extension_version = "unknown"
    try:
        connection.enableloadextension(True)
        connection.loadextension(args.extension)
        connection.enableloadextension(False)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        extension_version = str(
            next(connection.execute("SELECT hnsw_version()"))[0]
        )
        connection.execute(
            "CREATE VIRTUAL TABLE sift_hnsw USING hnsw("
            "embedding FLOAT32(128),metric=l2,m=16,ef_construction=128,"
            "cache_size_mb=1024)"
        )
        connection.execute("BEGIN IMMEDIATE")
        connection.cursor().executemany(
            "INSERT INTO sift_hnsw(rowid,embedding) VALUES(?,?)",
            (
                (
                    index + 1,
                    np.asarray(vector, dtype="<f4").tobytes(),
                )
                for index, vector in enumerate(vectors)
            ),
        )
        connection.execute("COMMIT")
        count = int(next(connection.execute("SELECT count(*) FROM sift_hnsw"))[0])
        if count != ROWS:
            raise RuntimeError(f"sqlite-hnsw stored {count} vectors, expected {ROWS}")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except BaseException:
        if not connection.getautocommit():
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    total = time.perf_counter() - started
    return {
        "engine": "sqlite-hnsw",
        "phase": "prepare",
        "rows": ROWS,
        "total_build_seconds": total,
        "storage": sqlite_storage(database),
        "hnsw": {"metric": "l2", "m": 16, "ef_construction": 128},
        "extension_version": extension_version,
    }


def check_sqlite(args: argparse.Namespace) -> None:
    import apsw

    database = args.work / "sqlite-hnsw.db"
    connection = apsw.Connection(str(database), flags=apsw.SQLITE_OPEN_READONLY)
    try:
        connection.enableloadextension(True)
        connection.loadextension(args.extension)
        connection.enableloadextension(False)
        count = int(next(connection.execute("SELECT count(*) FROM sift_hnsw"))[0])
        if count != ROWS:
            raise RuntimeError(f"sqlite-hnsw stored {count} vectors, expected {ROWS}")
        check = json.loads(
            str(next(connection.execute("SELECT hnsw_check('main','sift_hnsw')"))[0])
        )
        if not check.get("ok"):
            raise RuntimeError(f"sqlite-hnsw integrity check failed: {check}")
        print(json.dumps(check, sort_keys=True), flush=True)
    finally:
        connection.close()


def prepare_chroma(args: argparse.Namespace) -> dict[str, Any]:
    path = args.work / "chroma"
    if path.exists():
        shutil.rmtree(path)
    vectors = load_fvecs(dataset_paths(args.dataset)[0], ROWS)
    started = time.perf_counter()
    client = chroma_client(path)
    collection = client.create_collection(
        "sift",
        embedding_function=None,
        configuration={
            "hnsw": {
                "space": "l2",
                "ef_construction": 128,
                "max_neighbors": 16,
                "ef_search": 40,
            }
        },
    )
    maximum = int(client.get_max_batch_size())
    batch = min(maximum, 5_000)
    for start in range(0, ROWS, batch):
        end = min(start + batch, ROWS)
        collection.add(
            ids=[str(value) for value in range(start, end)],
            embeddings=np.asarray(vectors[start:end]).tolist(),
        )
    count = collection.count()
    configuration = collection.configuration
    stop_chroma(client)
    total = time.perf_counter() - started
    if count != ROWS:
        raise RuntimeError(f"Chroma stored {count} vectors, expected {ROWS}")
    return {
        "engine": "chroma",
        "phase": "prepare",
        "rows": count,
        "total_build_seconds": total,
        "storage": chroma_storage(path),
        "hnsw": configuration["hnsw"],
        "version": distribution_version("chromadb"),
    }


def prepare_milvus(args: argparse.Namespace) -> dict[str, Any]:
    from pymilvus import MilvusClient

    database = args.work / "milvus-lite.db"
    for path in database.parent.glob(database.name + "*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    vectors = load_fvecs(dataset_paths(args.dataset)[0], ROWS)
    started = time.perf_counter()
    client = MilvusClient(str(database))
    create_milvus_collection(client, "sift")
    batch = 2_000
    for start in range(0, ROWS, batch):
        end = min(start + batch, ROWS)
        values = np.asarray(vectors[start:end]).tolist()
        client.insert(
            "sift",
            [
                {"id": start + offset, "vector": vector}
                for offset, vector in enumerate(values)
            ],
        )
    client.flush("sift")
    details = create_milvus_hnsw(client, "sift")
    client.close()
    total = time.perf_counter() - started
    return {
        "engine": "milvus-lite",
        "phase": "prepare",
        "rows": ROWS,
        "total_build_seconds": total,
        "storage": milvus_storage(database),
        "hnsw": details,
        "version": distribution_version("milvus-lite"),
        "pymilvus_version": distribution_version("pymilvus"),
    }


def prepare(args: argparse.Namespace) -> None:
    result = {
        "sqlite-hnsw": prepare_sqlite,
        "chroma": prepare_chroma,
        "milvus-lite": prepare_milvus,
    }[args.engine](args)
    write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


class QueryAdapter:
    def __init__(self, engine: str, args: argparse.Namespace):
        self.engine = engine
        self.args = args
        self.resource: Any = None
        self.query: Callable[[np.ndarray], list[int]]
        self.set_ef: Callable[[int], None]
        if engine == "sqlite-hnsw":
            self._open_sqlite()
        elif engine == "chroma":
            self._open_chroma()
        else:
            self._open_milvus()

    def _open_sqlite(self) -> None:
        import apsw

        connection = apsw.Connection(
            str(self.args.work / "sqlite-hnsw.db"), flags=apsw.SQLITE_OPEN_READONLY
        )
        connection.enableloadextension(True)
        connection.loadextension(self.args.extension)
        connection.enableloadextension(False)
        cursor = connection.cursor()
        plan = list(
            cursor.execute(
                "EXPLAIN QUERY PLAN SELECT rowid FROM sift_hnsw "
                "WHERE embedding MATCH ? AND k=10 AND ef_search=? "
                "ORDER BY distance",
                (np.zeros(DIMS, dtype="<f4").tobytes(), 40),
            )
        )
        if not any("VIRTUAL TABLE INDEX 19:" in str(row[3]) for row in plan):
            raise RuntimeError(f"SQLite did not choose HNSW ANN plan: {plan}")
        current = 40

        def set_ef(value: int) -> None:
            nonlocal current
            current = value

        def query(vector: np.ndarray) -> list[int]:
            return [
                int(row[0]) - 1
                for row in cursor.execute(
                    "SELECT rowid FROM sift_hnsw WHERE embedding MATCH ? "
                    "AND k=10 AND ef_search=? ORDER BY distance",
                    (np.asarray(vector, dtype="<f4").tobytes(), current),
                )
            ]

        self.resource = (connection, cursor)
        self.set_ef = set_ef
        self.query = query

    def _open_chroma(self) -> None:
        client = chroma_client(self.args.work / "chroma")
        collection = client.get_collection("sift", embedding_function=None)

        def set_ef(value: int) -> None:
            nonlocal client, collection
            collection.modify(configuration={"hnsw": {"ef_search": value}})
            if collection.configuration["hnsw"]["ef_search"] != value:
                raise RuntimeError("Chroma did not apply ef_search")
            stop_chroma(client)
            client = chroma_client(self.args.work / "chroma")
            collection = client.get_collection("sift", embedding_function=None)
            if collection.configuration["hnsw"]["ef_search"] != value:
                raise RuntimeError("Chroma did not persist ef_search")
            self.resource = client

        def query(vector: np.ndarray) -> list[int]:
            result = collection.query(
                query_embeddings=[np.asarray(vector).tolist()],
                n_results=K,
                include=[],
            )
            return [int(value) for value in result["ids"][0]]

        self.resource = client
        self.set_ef = set_ef
        self.query = query

    def _open_milvus(self) -> None:
        from pymilvus import MilvusClient

        client = MilvusClient(str(self.args.work / "milvus-lite.db"))
        indexes = client.list_indexes("sift")
        details = client.describe_index("sift", indexes[0])
        if details.get("index_type") != "HNSW":
            raise RuntimeError(f"Milvus Lite index is not HNSW: {details}")
        client.load_collection("sift")
        current = 40

        def set_ef(value: int) -> None:
            nonlocal current
            current = value

        def query(vector: np.ndarray) -> list[int]:
            result = client.search(
                "sift",
                data=[np.asarray(vector).tolist()],
                anns_field="vector",
                limit=K,
                search_params={"metric_type": "L2", "params": {"ef": current}},
                output_fields=["id"],
            )
            return [int(item["id"]) for item in result[0]]

        self.resource = client
        self.set_ef = set_ef
        self.query = query

    def close(self) -> None:
        if self.engine == "sqlite-hnsw":
            connection, cursor = self.resource
            cursor.close()
            connection.close()
        elif self.engine == "chroma":
            stop_chroma(self.resource)
        else:
            self.resource.close()


def recall_at_10(actual: Sequence[int], expected: Sequence[int]) -> float:
    return len(set(actual).intersection(int(value) for value in expected)) / K


def percentile(values: Sequence[float], percent: int) -> float:
    ordered = sorted(values)
    index = max(0, int(np.ceil(len(ordered) * percent / 100.0)) - 1)
    return ordered[index]


def ef_orders() -> tuple[tuple[int, ...], ...]:
    midpoint = len(EFS) // 2
    return EFS, tuple(reversed(EFS)), EFS[midpoint:] + EFS[:midpoint]


def run_queries(args: argparse.Namespace) -> None:
    queries = load_fvecs(dataset_paths(args.dataset)[1], QUERIES + WARMUP)
    truth = load_truth(dataset_paths(args.dataset)[2], QUERIES)
    adapter = QueryAdapter(args.engine, args)
    try:
        for repeat, order in enumerate(ef_orders(), start=1):
            measured_order = np.random.default_rng(SEED + repeat).permutation(QUERIES)
            warm_order = QUERIES + np.random.default_rng(SEED + 100 + repeat).permutation(WARMUP)
            for ef in order:
                adapter.set_ef(ef)
                for index in warm_order:
                    adapter.query(queries[int(index)])
                latencies: list[float] = []
                recalls: list[float] = []
                started_all = time.perf_counter()
                for index in measured_order:
                    started = time.perf_counter_ns()
                    ids = adapter.query(queries[int(index)])
                    latencies.append((time.perf_counter_ns() - started) / 1_000_000)
                    recalls.append(recall_at_10(ids, truth[int(index)]))
                elapsed = time.perf_counter() - started_all
                result = {
                    "engine": args.engine,
                    "phase": "query",
                    "rows": ROWS,
                    "queries": QUERIES,
                    "warmup_queries": WARMUP,
                    "repeat": repeat,
                    "ef_search": ef,
                    "recall_at_10": statistics.fmean(recalls),
                    "ann_p50_ms": percentile(latencies, 50),
                    "ann_p95_ms": percentile(latencies, 95),
                    "ann_p99_ms": percentile(latencies, 99),
                    "qps": QUERIES / elapsed,
                }
                append_json(args.output, result)
                print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        adapter.close()


def run_cold(args: argparse.Namespace) -> None:
    queries = load_fvecs(dataset_paths(args.dataset)[1], 1)
    adapter = QueryAdapter(args.engine, args)
    try:
        adapter.set_ef(args.ef)
        started = time.perf_counter_ns()
        ids = adapter.query(queries[0])
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if len(ids) != K:
            raise RuntimeError(f"{args.engine} returned {len(ids)} cold results")
        result = {
            "engine": args.engine,
            "phase": "cold",
            "repeat": args.repeat,
            "ef_search": args.ef,
            "cold_query_ms": elapsed,
        }
        append_json(args.output, result)
        print(json.dumps(result, sort_keys=True), flush=True)
    finally:
        adapter.close()


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def mib(value: int | float) -> str:
    return f"{float(value) / (1024 * 1024):.1f}"


def report(args: argparse.Namespace) -> None:
    metadata = read_json(args.metadata)
    builds = {engine: read_json(args.results / f"{engine}-prepare.json") for engine in ENGINES}
    queries = [
        json.loads(line)
        for line in (args.results / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cold = [
        json.loads(line)
        for line in (args.results / "cold.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    memory = {
        path.stem.removesuffix("-memory"): read_json(path)["peak_memory_bytes"]
        for path in args.results.glob("*-memory.json")
    }

    build_rows = []
    storage_rows = []
    for engine in ENGINES:
        value = builds[engine]
        build_rows.append(
            (engine, f"{float(value['total_build_seconds']):.3f}", mib(memory.get(f"{engine}-prepare", 0)))
        )
        components = value["storage"]["components"]
        storage_rows.append(
            (engine, mib(value["storage"]["total_bytes"]), ", ".join(f"{name}={mib(size)}" for name, size in components.items()))
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for value in queries:
        grouped.setdefault((value["engine"], int(value["ef_search"])), []).append(value)
    query_rows = []
    medians: dict[str, list[tuple[int, float, float, float]]] = {engine: [] for engine in ENGINES}
    for engine in ENGINES:
        for ef in EFS:
            values = grouped[(engine, ef)]
            recalls = [float(value["recall_at_10"]) for value in values]
            p95s = [float(value["ann_p95_ms"]) for value in values]
            p50 = statistics.median(float(value["ann_p50_ms"]) for value in values)
            p99 = statistics.median(float(value["ann_p99_ms"]) for value in values)
            qps = statistics.median(float(value["qps"]) for value in values)
            recall = statistics.median(recalls)
            p95 = statistics.median(p95s)
            medians[engine].append((ef, recall, p95, qps))
            query_rows.append(
                (engine, ef, f"{recall:.4f} ({min(recalls):.4f}-{max(recalls):.4f})", f"{p50:.3f}", f"{p95:.3f} ({min(p95s):.3f}-{max(p95s):.3f})", f"{p99:.3f}", f"{qps:.2f}")
            )

    target_rows = []
    for target in (0.95, 0.98, 0.99):
        for engine in ENGINES:
            eligible = [value for value in medians[engine] if value[1] >= target]
            if eligible:
                ef, recall, p95, qps = min(eligible, key=lambda value: value[2])
                target_rows.append((f"{target:.2f}", engine, ef, f"{recall:.4f}", f"{p95:.3f}", f"{qps:.2f}"))
            else:
                target_rows.append((f"{target:.2f}", engine, "N/A", "N/A", "N/A", "N/A"))

    pareto_rows = []
    for engine in ENGINES:
        best_recall = -1.0
        for ef, recall, p95, _qps in sorted(medians[engine], key=lambda value: value[2]):
            if recall > best_recall:
                pareto_rows.append((engine, ef, f"{recall:.4f}", f"{p95:.3f}"))
                best_recall = recall

    cold_rows = []
    for engine in ENGINES:
        values = [float(value["cold_query_ms"]) for value in cold if value["engine"] == engine]
        cold_rows.append((engine, f"{statistics.median(values):.3f}", f"{min(values):.3f}", f"{max(values):.3f}", mib(max(memory.get(f"{engine}-cold-{repeat}", 0) for repeat in range(1, 4)))))

    query_memory_rows = [
        (engine, mib(memory.get(f"{engine}-query", 0))) for engine in ENGINES
    ]
    hnsw_rows = [
        (engine, json.dumps(builds[engine]["hnsw"], sort_keys=True)) for engine in ENGINES
    ]
    dataset_rows = [
        (name, value["bytes"], value["sha256"])
        for name, value in metadata["dataset"].items()
    ]
    report_text = f"""# SIFT1M embedded HNSW comparison

This report compares three persistent embedded HNSW databases in separate
native processes with identical SIFT vectors, query order, CPU affinity, and
Python measurement code.

The benchmark method is documented in [the benchmark guide](../README.md).

## Environment

- Measured: `{args.date}`
- Platform: `{metadata['platform']}`
- Python: `{metadata['python']}`
- CPU affinity: `{metadata['cpu_affinity']}`
- sqlite-hnsw: `{builds['sqlite-hnsw']['extension_version']}`
- SQLite CLI/APSW: `{metadata['sqlite']['cli']}` / `{metadata['sqlite']['apsw']}`
- SQLite CLI: `{metadata['sqlite']['cli_path']}`
- SQLite prefix: `{metadata['sqlite']['prefix']}`
- Packages: `{json.dumps(metadata['packages'], sort_keys=True)}`

{markdown_table(('Dataset file', 'Bytes', 'SHA-256'), dataset_rows)}

## Verified HNSW configuration

{markdown_table(('Engine', 'Configuration'), hnsw_rows)}

## Build

{markdown_table(('Engine', 'Persistent-ready build seconds', 'Peak RSS MiB'), build_rows)}

## Physical storage

{markdown_table(('Engine', 'Total MiB', 'Components MiB'), storage_rows)}

## Recall and latency sweep

{markdown_table(('Engine', 'ef', 'Recall@10 median (range)', 'p50 ms', 'p95 ms median (range)', 'p99 ms', 'QPS'), query_rows)}

## Recall targets

{markdown_table(('Target recall', 'Engine', 'ef', 'Recall@10', 'p95 ms', 'QPS'), target_rows)}

## Pareto frontier

{markdown_table(('Engine', 'ef', 'Recall@10', 'p95 ms'), pareto_rows)}

## Connection-cold query

{markdown_table(('Engine', 'Median ms', 'Min ms', 'Max ms', 'Peak RSS MiB'), cold_rows)}

## Steady-state query memory

{markdown_table(('Engine', 'Peak process-tree RSS MiB'), query_memory_rows)}

## Capability notes

| Engine | Persistence | Transactions/recovery | Filtering | Multi-instance memory |
| --- | --- | --- | --- | --- |
| sqlite-hnsw | Single SQLite file | SQLite transactions and WAL | Post-filter through joined tables | Snapshot is connection-local |
| Chroma | Directory with SQLite metadata and HNSW files | Chroma local persistence semantics | Metadata filtering | Product-managed local index |
| Milvus Lite | Local database file and sidecars | Milvus Lite flush and local persistence | Scalar filtering API | Product-managed local index |
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_text, encoding="utf-8")


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--extension", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--dataset", type=Path, required=True)
    validate_parser.add_argument("--work", type=Path, required=True)
    validate_parser.add_argument("--sqlite-cli", required=True)
    validate_parser.add_argument("--sqlite-prefix", required=True)
    validate_parser.add_argument("--sqlite-version", default="3.51.2")
    validate_parser.add_argument("--output", type=Path, required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("engine", choices=ENGINES)
    add_common(prepare_parser)
    prepare_parser.add_argument("--output", type=Path, required=True)

    check_parser = subparsers.add_parser("check-sqlite")
    check_parser.add_argument("--work", type=Path, required=True)
    check_parser.add_argument("--extension", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("engine", choices=ENGINES)
    add_common(run_parser)
    run_parser.add_argument("--output", type=Path, required=True)

    cold_parser = subparsers.add_parser("cold")
    cold_parser.add_argument("engine", choices=ENGINES)
    add_common(cold_parser)
    cold_parser.add_argument("--ef", type=int, default=100)
    cold_parser.add_argument("--repeat", type=int, required=True)
    cold_parser.add_argument("--output", type=Path, required=True)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--metadata", type=Path, required=True)
    report_parser.add_argument("--results", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    report_parser.add_argument("--date", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.work.mkdir(parents=True, exist_ok=True) if hasattr(args, "work") else None
    if args.command == "validate":
        validate(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "check-sqlite":
        check_sqlite(args)
    elif args.command == "run":
        run_queries(args)
    elif args.command == "cold":
        run_cold(args)
    else:
        report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
