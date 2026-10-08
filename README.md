# sqlite-hnsw

`sqlite-hnsw` is a SQLite loadable extension that stores a persistent HNSW
approximate-nearest-neighbour index in the same SQLite database file as the
application data. It indexes one fixed-width `float32` vector per virtual-table
row and supports L2, cosine, and inner-product distance.

The project is an early implementation. Its SQL API and on-disk format are not
yet stable.

## Build

Requirements are a C17 compiler, CMake 3.20 or newer, and SQLite 3.41 or newer.

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

The loadable extension is produced as `build/sqlite_hnsw.so`.

## Quick start

```sql
.load ./build/sqlite_hnsw

CREATE TABLE documents(id INTEGER PRIMARY KEY, content TEXT NOT NULL);
CREATE VIRTUAL TABLE document_hnsw USING hnsw(
  embedding FLOAT32(3),
  metric=cosine
);

BEGIN;
INSERT INTO documents VALUES (1, 'example');
INSERT INTO document_hnsw(rowid, embedding)
VALUES (1, hnsw_f32('[0.1, 0.2, 0.3]'));
COMMIT;

SELECT d.id, d.content, v.distance
FROM document_hnsw AS v
JOIN documents AS d ON d.id = v.rowid
WHERE v.embedding MATCH :query_vector
  AND v.k = 10
  AND v.ef_search = 100
ORDER BY v.distance;
```

Production applications should bind vectors as little-endian IEEE-754
`float32` BLOBs and update the application table and vector table in the same
SQLite transaction.

## Documentation

- [SQL and operational reference](docs/reference.md)
- [Internal architecture](docs/architecture.md)
- [Embedded database benchmark](bench/README.md)
- [Current SIFT1M comparison](bench/results/sift1m-embedded-hnsw-comparison.md)

