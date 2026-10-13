#include <math.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef TEST_EXTENSION_PATH
#define TEST_EXTENSION_PATH "./vechnsw.so"
#endif

static void fail(sqlite3 *db, const char *msg) {
  fprintf(stderr, "FAIL: %s", msg);
  if (db) {
    fprintf(stderr, ": %s", sqlite3_errmsg(db));
  }
  fputc('\n', stderr);
  exit(1);
}

static void check(int ok, sqlite3 *db, const char *msg) {
  if (!ok) {
    fail(db, msg);
  }
}

static void exec_sql(sqlite3 *db, const char *sql) {
  char *err = 0;
  int rc = sqlite3_exec(db, sql, 0, 0, &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL error: %s\nSQL: %s\n", err ? err : sqlite3_errmsg(db),
            sql);
    sqlite3_free(err);
    exit(1);
  }
}

static sqlite3_int64 query_i64(sqlite3 *db, const char *sql) {
  sqlite3_stmt *stmt = 0;
  sqlite3_int64 value = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare integer query");
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "integer query row");
  value = sqlite3_column_int64(stmt, 0);
  sqlite3_finalize(stmt);
  return value;
}

static void make_test_path(char *path, size_t path_size, const char *suite) {
  snprintf(path, path_size, "/tmp/vechnsw_test_%ld_%s.db", (long)getpid(),
           suite);
}

static void expect_step_fails(sqlite3 *db, const char *sql, const void *blob,
                              int blob_bytes, const char *msg) {
  sqlite3_stmt *stmt = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare expected-failing statement");
  if (blob) {
    sqlite3_bind_blob(stmt, 1, blob, blob_bytes, SQLITE_STATIC);
  }
  rc = sqlite3_step(stmt);
  check(rc != SQLITE_ROW && rc != SQLITE_DONE, db, msg);
  sqlite3_finalize(stmt);
}

static int count_step_rows(sqlite3 *db, const char *sql, const void *blob,
                           int blob_bytes) {
  sqlite3_stmt *stmt = 0;
  int count = 0;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare row-count statement");
  if (blob) {
    sqlite3_bind_blob(stmt, 1, blob, blob_bytes, SQLITE_STATIC);
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    count++;
  }
  check(rc == SQLITE_DONE, db, "row-count statement completed");
  sqlite3_finalize(stmt);
  return count;
}

static void load_extension(sqlite3 *db) {
  char *err = 0;
  int rc = sqlite3_enable_load_extension(db, 1);
  check(rc == SQLITE_OK, db, "enable load extension");
  rc = sqlite3_load_extension(db, TEST_EXTENSION_PATH, "sqlite3_vechnsw_init",
                              &err);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "load extension failed: %s\npath: %s\n",
            err ? err : sqlite3_errmsg(db), TEST_EXTENSION_PATH);
    sqlite3_free(err);
    exit(1);
  }
}

static void make_vec(float *v, int dim, int id) {
  int i;
  for (i = 0; i < dim; i++) {
    float x = (float)((id + 1) * (i + 3));
    v[i] = sinf(x * 0.013f) + cosf(x * 0.007f) + (float)(id % 7) * 0.01f;
  }
}

static void insert_vec(sqlite3 *db, sqlite3_stmt *stmt, sqlite3_int64 rowid,
                       const float *v, int dim) {
  int rc;
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, v, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_DONE, db, "insert vector");
}

static sqlite3_int64 query_first_sql(sqlite3 *db, const char *sql,
                                     const float *v, int dim,
                                     double *distance) {
  sqlite3_stmt *stmt = 0;
  sqlite3_int64 rowid = -1;
  int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare ann query");
  sqlite3_bind_blob(stmt, 1, v, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "ann query returned a row");
  rowid = sqlite3_column_int64(stmt, 0);
  *distance = sqlite3_column_double(stmt, 1);
  sqlite3_finalize(stmt);
  return rowid;
}

static sqlite3_int64 query_first(sqlite3 *db, const float *v, int dim,
                                 double *distance) {
  return query_first_sql(db,
                         "SELECT rowid, distance FROM doc_vec "
                         "WHERE embedding MATCH ?1 AND k = 5 "
                         "ORDER BY distance",
                         v, dim, distance);
}

static void test_scalar_functions(sqlite3 *db) {
  float a[4] = {1.0f, 0.0f, 0.0f, 0.0f};
  float b[4] = {0.0f, 1.0f, 0.0f, 0.0f};
  sqlite3_stmt *stmt = 0;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT vec_dim(?1), vec_distance(?1, ?2, 'l2'), "
      "vec_distance(?1, ?2, 'ip'), vec_distance(?1, ?1, 'cosine')",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare scalar function test");
  sqlite3_bind_blob(stmt, 1, a, sizeof(a), SQLITE_STATIC);
  sqlite3_bind_blob(stmt, 2, b, sizeof(b), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "scalar function row");
  check(sqlite3_column_int(stmt, 0) == 4, db, "vec_dim");
  check(fabs(sqlite3_column_double(stmt, 1) - sqrt(2.0)) < 1e-5, db,
        "l2 distance");
  check(fabs(sqlite3_column_double(stmt, 2)) < 1e-5, db, "ip distance");
  check(fabs(sqlite3_column_double(stmt, 3)) < 1e-5, db, "cosine self");
  sqlite3_finalize(stmt);
}

static void test_scalar_dimension_limits(sqlite3 *db) {
  const int max_dim = 4096;
  float *max_vec = (float *)calloc((size_t)max_dim, sizeof(float));
  float *too_big = (float *)calloc((size_t)max_dim + 1, sizeof(float));
  float bad[2] = {NAN, 0.0f};
  sqlite3_stmt *stmt = 0;
  int rc;
  check(max_vec != 0 && too_big != 0, db, "allocate dimension limit vectors");
  max_vec[0] = 1.0f;
  too_big[0] = 1.0f;

  rc = sqlite3_prepare_v2(
      db, "SELECT vec_dim(?1), vec_distance(?1, ?1, 'l2')", -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare max scalar dimension test");
  sqlite3_bind_blob(stmt, 1, max_vec, max_dim * (int)sizeof(float),
                    SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "4096-dim scalar functions succeed");
  check(sqlite3_column_int(stmt, 0) == max_dim, db, "4096-dim vec_dim");
  check(fabs(sqlite3_column_double(stmt, 1)) < 1e-5, db,
        "4096-dim distance");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(db, "SELECT vec_dim(?1)", -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare too-wide vec_dim test");
  sqlite3_bind_blob(stmt, 1, too_big, (max_dim + 1) * (int)sizeof(float),
                    SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ERROR, db, "4097-dim vec_dim is rejected");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(db, "SELECT vec_distance(?1, ?1, 'l2')", -1, &stmt,
                          0);
  check(rc == SQLITE_OK, db, "prepare too-wide vec_distance test");
  sqlite3_bind_blob(stmt, 1, too_big, (max_dim + 1) * (int)sizeof(float),
                    SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ERROR, db, "4097-dim vec_distance is rejected");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(db, "SELECT vec_dim(?1)", -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare non-finite vec_dim test");
  sqlite3_bind_blob(stmt, 1, bad, sizeof(bad), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ERROR, db, "non-finite vec_dim is rejected");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(db, "SELECT vec_distance(?1, ?1, 'l2')", -1, &stmt,
                          0);
  check(rc == SQLITE_OK, db, "prepare non-finite vec_distance test");
  sqlite3_bind_blob(stmt, 1, bad, sizeof(bad), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ERROR, db, "non-finite vec_distance is rejected");
  sqlite3_finalize(stmt);

  free(max_vec);
  free(too_big);
}

static void test_table_dimension_limits(sqlite3 *db) {
  char *err = 0;
  int rc = sqlite3_exec(
      db,
      "CREATE VIRTUAL TABLE max_dim_vec USING vec_hnsw("
      "dim=4096, metric=l2, build_threads=2, build_batch_size=4)",
      0, 0, &err);
  check(rc == SQLITE_OK, db, "dim=4096 table is accepted");
  sqlite3_free(err);
  exec_sql(db, "DROP TABLE max_dim_vec");

  err = 0;
  rc = sqlite3_exec(
      db,
      "CREATE VIRTUAL TABLE too_wide_vec USING vec_hnsw("
      "dim=4097, metric=l2, build_threads=2, build_batch_size=4)",
      0, 0, &err);
  check(rc != SQLITE_OK, db, "dim=4097 table is rejected");
  check(err && strstr(err, "4096") != 0, db,
        "dim=4097 error mentions max dimension");
  sqlite3_free(err);
}

static void expect_create_fails(sqlite3 *db, const char *sql,
                                const char *msg) {
  char *err = 0;
  int rc = sqlite3_exec(db, sql, 0, 0, &err);
  check(rc != SQLITE_OK, db, msg);
  sqlite3_free(err);
}

static void test_parameter_limits(sqlite3 *db) {
  exec_sql(db,
           "CREATE VIRTUAL TABLE max_param_vec USING vec_hnsw("
           "dim=16, metric=l2, m=128, ef_construction=8192, "
           "ef_search=8192, build_threads=256, build_batch_size=65536)");
  exec_sql(db, "DROP TABLE max_param_vec");

  expect_create_fails(db,
                      "CREATE VIRTUAL TABLE bad_m_vec USING vec_hnsw("
                      "dim=16, metric=l2, m=129, build_threads=2, "
                      "build_batch_size=4)",
                      "m above max is rejected");
  expect_create_fails(db,
                      "CREATE VIRTUAL TABLE bad_efc_vec USING vec_hnsw("
                      "dim=16, metric=l2, m=16, ef_construction=8193, "
                      "build_threads=2, build_batch_size=4)",
                      "ef_construction above max is rejected");
  expect_create_fails(db,
                      "CREATE VIRTUAL TABLE bad_efs_vec USING vec_hnsw("
                      "dim=16, metric=l2, ef_search=8193, "
                      "build_threads=2, build_batch_size=4)",
                      "ef_search above max is rejected");
  expect_create_fails(db,
                      "CREATE VIRTUAL TABLE bad_threads_vec USING vec_hnsw("
                      "dim=16, metric=l2, build_threads=257, "
                      "build_batch_size=4)",
                      "build_threads above max is rejected");
  expect_create_fails(db,
                      "CREATE VIRTUAL TABLE bad_batch_vec USING vec_hnsw("
                      "dim=16, metric=l2, build_threads=2, "
                      "build_batch_size=65537)",
                      "build_batch_size above max is rejected");
}

static void test_invalid_build_threads(sqlite3 *db) {
  char *err = 0;
  int rc = sqlite3_exec(
      db,
      "CREATE VIRTUAL TABLE bad_vec USING vec_hnsw("
      "dim=16, metric=cosine, m=8, ef_construction=32, ef_search=32, "
      "build_threads=1)",
      0, 0, &err);
  check(rc != SQLITE_OK, db, "build_threads=1 is rejected");
  sqlite3_free(err);
}

static int read_max_node_id(sqlite3 *db) {
  sqlite3_stmt *stmt = 0;
  int max_node = -1;
  int rc = sqlite3_prepare_v2(
      db, "SELECT COALESCE(MAX(node_id), -1) FROM doc_vec_vectors", -1, &stmt,
      0);
  check(rc == SQLITE_OK, db, "prepare max node id");
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "read max node id");
  max_node = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return max_node;
}

static unsigned char *read_active_nodes(sqlite3 *db, int max_node) {
  sqlite3_stmt *stmt = 0;
  unsigned char *active =
      (unsigned char *)calloc((size_t)max_node + 1, sizeof(unsigned char));
  int rc;
  check(active != 0, db, "allocate active node map");
  rc = sqlite3_prepare_v2(
      db, "SELECT node_id FROM doc_vec_vectors WHERE deleted = 0", -1, &stmt,
      0);
  check(rc == SQLITE_OK, db, "prepare active node scan");
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int node = sqlite3_column_int(stmt, 0);
    check(node >= 0 && node <= max_node, db, "active node id is in range");
    active[node] = 1;
  }
  check(rc == SQLITE_DONE, db, "scan active nodes");
  sqlite3_finalize(stmt);
  return active;
}

static void check_graph_invariants(sqlite3 *db, int m) {
  sqlite3_stmt *stmt = 0;
  int max_node = read_max_node_id(db);
  unsigned char *active = read_active_nodes(db, max_node);
  int rc = sqlite3_prepare_v2(
      db, "SELECT node_id, layer, neighbors FROM doc_vec_graph", -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare graph invariant scan");
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    int node = sqlite3_column_int(stmt, 0);
    int layer = sqlite3_column_int(stmt, 1);
    const int32_t *ids = (const int32_t *)sqlite3_column_blob(stmt, 2);
    int bytes = sqlite3_column_bytes(stmt, 2);
    int count = bytes / (int)sizeof(int32_t);
    int cap = layer == 0 ? m * 2 : m;
    int i, j;
    check(node >= 0 && node <= max_node && active[node], db,
          "graph node id is active");
    check(layer >= 0, db, "graph layer is non-negative");
    check(bytes >= 0 && bytes % (int)sizeof(int32_t) == 0, db,
          "neighbor blob is int32 aligned");
    check(count <= cap, db, "neighbor count respects HNSW cap");
    for (i = 0; i < count; i++) {
      check(ids[i] >= 0 && ids[i] <= max_node && active[ids[i]], db,
            "neighbor id is active");
      check(ids[i] != node, db, "graph has no self edge");
      for (j = i + 1; j < count; j++) {
        check(ids[i] != ids[j], db, "graph has no duplicate neighbor");
      }
    }
  }
  check(rc == SQLITE_DONE, db, "scan graph invariants");
  sqlite3_finalize(stmt);
  free(active);
}

static void check_ann_embedding_contract(sqlite3 *db, const float *v, int dim) {
  sqlite3_stmt *stmt = 0;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT embedding IS NULL, distance IS NOT NULL FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare ANN column contract query");
  sqlite3_bind_blob(stmt, 1, v, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "ANN column contract row");
  check(sqlite3_column_int(stmt, 0) == 1, db, "ANN embedding is NULL");
  check(sqlite3_column_int(stmt, 1) == 1, db, "ANN distance is not NULL");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(
      db,
      "SELECT embedding IS NULL, length(embedding) FROM doc_vec "
      "ORDER BY rowid LIMIT 1",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare scan column contract query");
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "scan column contract row");
  check(sqlite3_column_int(stmt, 0) == 0, db, "scan embedding is BLOB");
  check(sqlite3_column_int(stmt, 1) == dim * (int)sizeof(float), db,
        "scan embedding length");
  sqlite3_finalize(stmt);
}

static void check_nonfinite_vectors_rejected(sqlite3 *db, int dim) {
  sqlite3_stmt *stmt = 0;
  float *bad = (float *)calloc((size_t)dim, sizeof(float));
  int rc;
  check(bad != 0, db, "allocate non-finite vector");
  bad[0] = INFINITY;

  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare non-finite insert");
  sqlite3_bind_int64(stmt, 1, 9001);
  sqlite3_bind_blob(stmt, 2, bad, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc != SQLITE_DONE, db, "non-finite insert is rejected");
  sqlite3_finalize(stmt);

  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare non-finite query");
  sqlite3_bind_blob(stmt, 1, bad, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ERROR, db, "non-finite ANN query is rejected");
  sqlite3_finalize(stmt);

  free(bad);
}

static void check_vectors_compacted(sqlite3 *db, int expected_count) {
  sqlite3_int64 count =
      query_i64(db, "SELECT COUNT(*) FROM doc_vec_vectors");
  sqlite3_int64 deleted =
      query_i64(db, "SELECT COUNT(*) FROM doc_vec_vectors WHERE deleted<>0");
  sqlite3_int64 min_node =
      query_i64(db, "SELECT COALESCE(MIN(node_id), -1) FROM doc_vec_vectors");
  sqlite3_int64 max_node =
      query_i64(db, "SELECT COALESCE(MAX(node_id), -1) FROM doc_vec_vectors");
  check(count == expected_count, db, "compacted vector row count");
  check(deleted == 0, db, "compacted vectors have no tombstones");
  if (expected_count > 0) {
    check(min_node == 0, db, "compacted node ids start at zero");
    check(max_node == expected_count - 1, db, "compacted node ids are dense");
  } else {
    check(min_node == -1 && max_node == -1, db, "empty compaction node ids");
  }
}

static void test_virtual_table(const char *path) {
  const int dim = 16;
  const int n = 300;
  sqlite3 *db = 0;
  sqlite3_stmt *insert = 0;
  float *v = (float *)malloc((size_t)dim * sizeof(float));
  double distance = 0.0;
  sqlite3_int64 rowid;
  int i;
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open database");
  load_extension(db);

  exec_sql(db, "PRAGMA journal_mode=OFF");
  exec_sql(db, "CREATE VIRTUAL TABLE doc_vec USING vec_hnsw("
               "dim=16, metric=cosine, m=8, ef_construction=32, "
               "ef_search=32, build_threads=2, build_batch_size=16, "
               "seed=123)");
  exec_sql(db, "BEGIN");
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db, "prepare insert");
  for (i = 0; i < n; i++) {
    make_vec(v, dim, i);
    insert_vec(db, insert, i + 1, v, dim);
  }
  sqlite3_finalize(insert);
  exec_sql(db, "COMMIT");
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");
  check_graph_invariants(db, 8);
  check_vectors_compacted(db, n);

  make_vec(v, dim, 42);
  rowid = query_first(db, v, dim, &distance);
  check(rowid == 43, db, "query finds exact inserted vector");
  check(distance < 1e-4, db, "exact vector distance is near zero");
  check_ann_embedding_contract(db, v, dim);
  check_nonfinite_vectors_rejected(db, dim);

  exec_sql(db, "DELETE FROM doc_vec WHERE rowid = 43");
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");
  check_graph_invariants(db, 8);
  check_vectors_compacted(db, n - 1);
  rowid = query_first(db, v, dim, &distance);
  check(rowid != 43, db, "deleted row is not returned");

  rc = sqlite3_exec(db, "UPDATE doc_vec SET embedding = zeroblob(3) WHERE rowid=1",
                    0, 0, 0);
  check(rc != SQLITE_OK, db, "invalid update is rejected");

  rc = sqlite3_prepare_v2(db, "UPDATE doc_vec SET embedding=?1 WHERE rowid=10",
                          -1, &insert, 0);
  check(rc == SQLITE_OK, db, "prepare update");
  sqlite3_bind_blob(insert, 1, v, dim * (int)sizeof(float), SQLITE_STATIC);
  rc = sqlite3_step(insert);
  check(rc == SQLITE_DONE, db, "update vector");
  sqlite3_finalize(insert);
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");
  check_graph_invariants(db, 8);
  check_vectors_compacted(db, n - 1);
  rowid = query_first(db, v, dim, &distance);
  check(rowid == 10, db, "updated row is returned");

  free(v);
  sqlite3_close(db);

  db = 0;
  rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "reopen database");
  load_extension(db);
  v = (float *)malloc((size_t)dim * sizeof(float));
  make_vec(v, dim, 42);
  rowid = query_first(db, v, dim, &distance);
  check(rowid == 10, db, "persisted graph returns updated row");
  free(v);
  sqlite3_close(db);
}

static void test_cross_connection_cache(const char *path) {
  const int dim = 2;
  sqlite3 *db1 = 0;
  sqlite3 *db2 = 0;
  sqlite3_stmt *insert = 0;
  float v[2];
  double distance = 0.0;
  sqlite3_int64 rowid;
  int rc = sqlite3_open(path, &db1);
  check(rc == SQLITE_OK, db1, "open cache test db1");
  load_extension(db1);
  exec_sql(db1, "CREATE VIRTUAL TABLE doc_vec USING vec_hnsw("
                "dim=2, metric=l2, m=4, ef_construction=16, ef_search=16, "
                "build_threads=2, build_batch_size=4, seed=123)");
  rc = sqlite3_prepare_v2(
      db1, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db1, "prepare cache test insert");
  for (int i = 0; i < 6; i++) {
    v[0] = (float)(i * 10);
    v[1] = 0.0f;
    insert_vec(db1, insert, i + 1, v, dim);
  }
  sqlite3_finalize(insert);
  exec_sql(db1, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");

  rc = sqlite3_open(path, &db2);
  check(rc == SQLITE_OK, db2, "open cache test db2");
  load_extension(db2);

  v[0] = 0.0f;
  v[1] = 0.0f;
  rowid = query_first(db2, v, dim, &distance);
  check(rowid == 1, db2, "db2 initial query caches rowid 1");

  exec_sql(db1, "DELETE FROM doc_vec WHERE rowid = 1");
  rowid = query_first(db2, v, dim, &distance);
  check(rowid != 1, db2, "db2 cache is invalidated after db1 delete");

  rc = sqlite3_prepare_v2(
      db1, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db1, "prepare cache test second insert");
  v[0] = 0.0f;
  v[1] = 0.0f;
  insert_vec(db1, insert, 99, v, dim);
  sqlite3_finalize(insert);
  rowid = query_first(db2, v, dim, &distance);
  check(rowid == 99, db2, "db2 cache sees db1 insert");

  sqlite3_close(db2);
  sqlite3_close(db1);
}

static void exec_owned_sql(sqlite3 *db, char *sql) {
  check(sql != 0, db, "allocate SQL");
  exec_sql(db, sql);
  sqlite3_free(sql);
}

static void test_schema_rebuild(const char *path, const char *aux_path) {
  const int dim = 2;
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  float v[2] = {1.0f, 0.0f};
  double distance = 0.0;
  sqlite3_int64 rowid;
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open schema rebuild db");
  load_extension(db);

  exec_sql(db, "CREATE VIRTUAL TABLE temp.temp_vec USING vec_hnsw("
               "dim=2, metric=l2, m=4, ef_construction=16, ef_search=16, "
               "build_threads=2, build_batch_size=4)");
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO temp.temp_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare temp insert");
  insert_vec(db, stmt, 7, v, dim);
  sqlite3_finalize(stmt);
  exec_sql(db, "SELECT vec_hnsw_rebuild('temp', 'temp_vec')");
  rowid = query_first_sql(db,
                          "SELECT rowid, distance FROM temp.temp_vec "
                          "WHERE embedding MATCH ?1 AND k = 1 "
                          "ORDER BY distance",
                          v, dim, &distance);
  check(rowid == 7, db, "temp schema rebuild query");

  exec_owned_sql(db, sqlite3_mprintf("ATTACH %Q AS aux", aux_path));
  exec_sql(db, "CREATE VIRTUAL TABLE aux.aux_vec USING vec_hnsw("
               "dim=2, metric=l2, m=4, ef_construction=16, ef_search=16, "
               "build_threads=2, build_batch_size=4)");
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO aux.aux_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare aux insert");
  insert_vec(db, stmt, 8, v, dim);
  sqlite3_finalize(stmt);
  exec_sql(db, "SELECT vec_hnsw_rebuild('aux', 'aux_vec')");
  rowid = query_first_sql(db,
                          "SELECT rowid, distance FROM aux.aux_vec "
                          "WHERE embedding MATCH ?1 AND k = 1 "
                          "ORDER BY distance",
                          v, dim, &distance);
  check(rowid == 8, db, "attached schema rebuild query");

  rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('doc_vec')", 0, 0, 0);
  check(rc != SQLITE_OK, db, "single-argument rebuild is rejected");
  rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('', 'doc_vec')", 0, 0, 0);
  check(rc != SQLITE_OK, db, "empty rebuild database is rejected");
  rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('main', '')", 0, 0, 0);
  check(rc != SQLITE_OK, db, "empty rebuild table is rejected");

  sqlite3_close(db);
}

static void create_small_doc_vec(sqlite3 *db) {
  sqlite3_stmt *insert = 0;
  float v[2];
  int rc;
  exec_sql(db, "CREATE VIRTUAL TABLE doc_vec USING vec_hnsw("
               "dim=2, metric=l2, m=4, ef_construction=16, ef_search=16, "
               "build_threads=2, build_batch_size=4, seed=123)");
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db, "prepare small table insert");
  for (int i = 0; i < 6; i++) {
    v[0] = (float)i;
    v[1] = 0.0f;
    insert_vec(db, insert, i + 1, v, 2);
  }
  sqlite3_finalize(insert);
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");
}

static void expect_ann_query_fails(sqlite3 *db) {
  sqlite3_stmt *stmt = 0;
  float q[2] = {0.0f, 0.0f};
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare expected-failing ANN query");
  sqlite3_bind_blob(stmt, 1, q, sizeof(q), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc != SQLITE_ROW && rc != SQLITE_DONE, db,
        "corrupt shadow graph is rejected");
  sqlite3_finalize(stmt);
}

static void test_shadow_corruption(const char *path) {
  sqlite3 *db = 0;
  sqlite3_stmt *stmt = 0;
  int32_t self_edge[1] = {0};
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open shadow corruption db");
  load_extension(db);
  create_small_doc_vec(db);

  rc = sqlite3_prepare_v2(
      db,
      "INSERT OR REPLACE INTO doc_vec_graph(node_id, layer, neighbors) "
      "VALUES(0, 0, ?1)",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare corrupt graph write");
  sqlite3_bind_blob(stmt, 1, self_edge, sizeof(self_edge), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_DONE, db, "write corrupt graph");
  sqlite3_finalize(stmt);
  sqlite3_close(db);

  rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "reopen corrupt graph db");
  load_extension(db);
  expect_ann_query_fails(db);
  sqlite3_close(db);

  remove(path);
  rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open corrupt vector db");
  load_extension(db);
  create_small_doc_vec(db);
  exec_sql(db, "UPDATE doc_vec_vectors SET embedding=zeroblob(4) "
               "WHERE node_id=0");
  rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')", 0, 0, 0);
  check(rc != SQLITE_OK, db, "corrupt vector embedding is rejected");
  sqlite3_close(db);
}

static void test_shadow_meta_corruption(const char *path) {
  sqlite3 *db = 0;
  int rc;
  const char *bad_meta_sql[] = {
      "UPDATE doc_vec_meta SET value='not-an-int' WHERE key='entry_node'",
      "UPDATE doc_vec_meta SET value='abc' WHERE key='generation'",
      "UPDATE doc_vec_meta SET value='true' WHERE key='graph_valid'",
      "UPDATE doc_vec_meta SET value='128x' WHERE key='dim'",
  };
  int i;

  for (i = 0; i < (int)(sizeof(bad_meta_sql) / sizeof(bad_meta_sql[0])); i++) {
    remove(path);
    rc = sqlite3_open(path, &db);
    check(rc == SQLITE_OK, db, "open corrupt meta db");
    load_extension(db);
    create_small_doc_vec(db);
    exec_sql(db, bad_meta_sql[i]);
    if (i == 3) {
      rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')", 0,
                        0, 0);
      check(rc != SQLITE_OK, db, "corrupt dim meta is rejected");
    } else {
      expect_ann_query_fails(db);
    }
    sqlite3_close(db);
    db = 0;
  }
}

static void test_shadow_suffix_tables(const char *path) {
  sqlite3 *db = 0;
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open suffix table db");
  load_extension(db);
  exec_sql(db, "CREATE TABLE user_meta(x INTEGER)");
  exec_sql(db, "CREATE TABLE event_vectors(x INTEGER)");
  exec_sql(db, "CREATE TABLE foo_graph(x INTEGER)");
  exec_sql(db, "INSERT INTO user_meta VALUES(1)");
  exec_sql(db, "INSERT INTO event_vectors VALUES(2)");
  exec_sql(db, "INSERT INTO foo_graph VALUES(3)");
  check(query_i64(db, "SELECT x FROM user_meta") == 1, db,
        "ordinary _meta table works");
  check(query_i64(db, "SELECT x FROM event_vectors") == 2, db,
        "ordinary _vectors table works");
  check(query_i64(db, "SELECT x FROM foo_graph") == 3, db,
        "ordinary _graph table works");
  sqlite3_close(db);
}

static void test_transactions(const char *path) {
  sqlite3 *db = 0;
  sqlite3_stmt *insert = 0;
  float v[2];
  double distance = 0.0;
  sqlite3_int64 rowid;
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open transaction db");
  load_extension(db);
  exec_sql(db, "CREATE VIRTUAL TABLE doc_vec USING vec_hnsw("
               "dim=2, metric=l2, m=4, ef_construction=16, ef_search=16, "
               "build_threads=2, build_batch_size=4, seed=123)");
  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db, "prepare transaction insert");

  exec_sql(db, "BEGIN");
  v[0] = 0.0f;
  v[1] = 0.0f;
  insert_vec(db, insert, 1, v, 2);
  exec_sql(db, "ROLLBACK");
  check(query_i64(db, "SELECT COUNT(*) FROM doc_vec") == 0, db,
        "rolled-back insert is invisible");

  v[0] = 0.0f;
  v[1] = 0.0f;
  insert_vec(db, insert, 1, v, 2);
  v[0] = 5.0f;
  v[1] = 0.0f;
  insert_vec(db, insert, 2, v, 2);
  sqlite3_finalize(insert);
  insert = 0;
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");

  exec_sql(db, "BEGIN");
  exec_sql(db, "DELETE FROM doc_vec WHERE rowid=1");
  exec_sql(db, "ROLLBACK");
  v[0] = 0.0f;
  v[1] = 0.0f;
  rowid = query_first(db, v, 2, &distance);
  check(rowid == 1, db, "rolled-back delete keeps row");

  exec_sql(db, "BEGIN");
  rc = sqlite3_prepare_v2(db, "UPDATE doc_vec SET embedding=?1 WHERE rowid=1",
                          -1, &insert, 0);
  check(rc == SQLITE_OK, db, "prepare transaction update");
  v[0] = 5.0f;
  v[1] = 0.0f;
  sqlite3_bind_blob(insert, 1, v, sizeof(v), SQLITE_STATIC);
  rc = sqlite3_step(insert);
  check(rc == SQLITE_DONE, db, "transaction update");
  sqlite3_finalize(insert);
  insert = 0;
  exec_sql(db, "ROLLBACK");
  rowid = query_first(db, v, 2, &distance);
  check(rowid == 2, db, "rolled-back update keeps old vector");

  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  check(rc == SQLITE_OK, db, "prepare savepoint insert");
  exec_sql(db, "SAVEPOINT sp");
  v[0] = 10.0f;
  v[1] = 0.0f;
  insert_vec(db, insert, 3, v, 2);
  exec_sql(db, "ROLLBACK TO sp");
  exec_sql(db, "RELEASE sp");
  sqlite3_finalize(insert);
  insert = 0;
  check(query_i64(db, "SELECT COUNT(*) FROM doc_vec WHERE rowid=3") == 0, db,
        "rolled-back savepoint insert is invisible");

  exec_sql(db,
           "CREATE TRIGGER fail_doc_vec_graph_insert "
           "BEFORE INSERT ON doc_vec_graph "
           "BEGIN SELECT RAISE(FAIL, 'forced graph insert failure'); END");
  rc = sqlite3_exec(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')", 0, 0, 0);
  check(rc != SQLITE_OK, db, "rebuild failure is reported");
  exec_sql(db, "DROP TRIGGER fail_doc_vec_graph_insert");
  sqlite3_close(db);

  rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "reopen after failed rebuild");
  load_extension(db);
  v[0] = 0.0f;
  v[1] = 0.0f;
  rowid = query_first(db, v, 2, &distance);
  check(rowid == 1, db, "failed rebuild rolled back old graph");
  sqlite3_close(db);
}

static void test_planner_api(const char *path) {
  sqlite3 *db = 0;
  float q[2] = {0.0f, 0.0f};
  float bad_dim[1] = {0.0f};
  float bad_value[2] = {NAN, 0.0f};
  sqlite3_stmt *stmt = 0;
  int rc;
  int count;
  int rc_open = sqlite3_open(path, &db);
  check(rc_open == SQLITE_OK, db, "open planner db");
  load_extension(db);
  create_small_doc_vec(db);

  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec WHERE embedding MATCH ?1 ORDER BY distance",
      q, sizeof(q), "MATCH without k is rejected");
  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 0 ORDER BY distance",
      q, sizeof(q), "k=0 is rejected");
  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
      bad_dim, sizeof(bad_dim), "wrong query blob size is rejected");
  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 ORDER BY distance",
      bad_value, sizeof(bad_value), "non-finite query is rejected");
  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 AND ef_search = 0 "
      "ORDER BY distance",
      q, sizeof(q), "ef_search=0 is rejected");
  expect_step_fails(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 1 AND ef_search = 8193 "
      "ORDER BY distance",
      q, sizeof(q), "ef_search above max is rejected");

  count = count_step_rows(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 3 ORDER BY distance",
      q, sizeof(q));
  check(count > 0 && count <= 3, db, "default ef_search query succeeds");

  count = count_step_rows(
      db,
      "SELECT rowid FROM doc_vec WHERE embedding MATCH ?1 AND k = 3",
      q, sizeof(q));
  check(count > 0 && count <= 3, db, "ANN query without ORDER BY succeeds");

  rc = sqlite3_prepare_v2(
      db,
      "SELECT embedding IS NOT NULL, distance IS NULL FROM doc_vec "
      "ORDER BY rowid LIMIT 1",
      -1, &stmt, 0);
  check(rc == SQLITE_OK, db, "prepare full-scan projection");
  rc = sqlite3_step(stmt);
  check(rc == SQLITE_ROW, db, "full-scan projection row");
  check(sqlite3_column_int(stmt, 0) == 1, db, "full scan embedding is BLOB");
  check(sqlite3_column_int(stmt, 1) == 1, db, "full scan distance is NULL");
  sqlite3_finalize(stmt);

  check_ann_embedding_contract(db, q, 2);
  sqlite3_close(db);
}

static void run_scalar_suite(const char *path) {
  sqlite3 *db = 0;
  int rc = sqlite3_open(path, &db);
  check(rc == SQLITE_OK, db, "open scalar suite db");
  load_extension(db);
  test_scalar_functions(db);
  test_scalar_dimension_limits(db);
  test_table_dimension_limits(db);
  test_parameter_limits(db);
  test_invalid_build_threads(db);
  sqlite3_close(db);
}

static void run_all_suites(void);

static int run_suite(const char *suite) {
  char path[256];
  char aux_path[256];
  make_test_path(path, sizeof(path), suite);
  snprintf(aux_path, sizeof(aux_path), "%s_aux", path);
  remove(path);
  remove(aux_path);

  if (strcmp(suite, "all") == 0) {
    run_all_suites();
  } else if (strcmp(suite, "scalar") == 0) {
    run_scalar_suite(path);
  } else if (strcmp(suite, "virtual") == 0) {
    test_virtual_table(path);
  } else if (strcmp(suite, "cache") == 0) {
    test_cross_connection_cache(path);
  } else if (strcmp(suite, "schema") == 0) {
    test_schema_rebuild(path, aux_path);
  } else if (strcmp(suite, "shadow") == 0) {
    test_shadow_corruption(path);
    remove(path);
    test_shadow_meta_corruption(path);
    remove(path);
    test_shadow_suffix_tables(path);
  } else if (strcmp(suite, "txn") == 0) {
    test_transactions(path);
  } else if (strcmp(suite, "planner") == 0) {
    test_planner_api(path);
  } else {
    fprintf(stderr, "unknown test suite: %s\n", suite);
    return 2;
  }

  remove(path);
  remove(aux_path);
  return 0;
}

static void run_all_suites(void) {
  const char *suites[] = {"scalar", "virtual", "cache", "schema",
                          "shadow", "txn",     "planner"};
  int i;
  for (i = 0; i < (int)(sizeof(suites) / sizeof(suites[0])); i++) {
    int rc = run_suite(suites[i]);
    if (rc != 0) {
      exit(rc);
    }
  }
}

int main(int argc, char **argv) {
  int rc;
  if (argc > 2) {
    fprintf(stderr, "usage: %s [suite]\n", argv[0]);
    return 2;
  }
  rc = argc == 2 ? run_suite(argv[1]) : run_suite("all");
  if (rc != 0) {
    return rc;
  }
  puts("vechnsw tests passed");
  return 0;
}
