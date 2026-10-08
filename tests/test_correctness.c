#include <math.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { METRIC_L2 = 0, METRIC_COSINE = 1, METRIC_IP = 2 };

typedef struct Expected {
  sqlite3_int64 rowid;
  double distance;
} Expected;

typedef struct QueryResult {
  sqlite3_int64 rowid;
  double distance;
  int visited;
} QueryResult;

static int failures = 0;

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (message));    \
      ++failures;                                                              \
      goto done;                                                               \
    }                                                                          \
  } while (0)

static int execute(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "SQL failed: %s\n%s\n", sql,
            error != NULL ? error : sqlite3_errmsg(db));
  }
  sqlite3_free(error);
  return rc;
}

static int expect_sql_error(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
  sqlite3_free(error);
  return rc != SQLITE_OK;
}

static int load(sqlite3 *db, const char *extension) {
  char *error = NULL;
  int rc = sqlite3_enable_load_extension(db, 1);
  if (rc == SQLITE_OK) {
    rc = sqlite3_load_extension(db, extension, NULL, &error);
  }
  if (rc != SQLITE_OK) {
    fprintf(stderr, "load failed: %s\n",
            error != NULL ? error : sqlite3_errmsg(db));
  }
  sqlite3_free(error);
  return rc;
}

static void write_f32_le(unsigned char *out, float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  out[0] = (unsigned char)(bits & 0xffU);
  out[1] = (unsigned char)((bits >> 8U) & 0xffU);
  out[2] = (unsigned char)((bits >> 16U) & 0xffU);
  out[3] = (unsigned char)((bits >> 24U) & 0xffU);
}

static void encode_vector(const float *values, int dims, unsigned char *blob) {
  for (int i = 0; i < dims; ++i) {
    write_f32_le(blob + (size_t)i * 4U, values[i]);
  }
}

static uint64_t next_random(uint64_t *state) {
  uint64_t value = *state;
  value ^= value << 13U;
  value ^= value >> 7U;
  value ^= value << 17U;
  *state = value;
  return value;
}

static float random_float(uint64_t *state) {
  int value = (int)(next_random(state) % UINT64_C(20001)) - 10000;
  return (float)value / 997.0F;
}

static void fill_vectors(float *vectors, int rows, int dims, uint64_t seed) {
  uint64_t state = seed;
  for (int row = 0; row < rows; ++row) {
    double norm = 0.0;
    for (int dim = 0; dim < dims; ++dim) {
      float value = random_float(&state);
      value += (float)(row + 1) * 0.000013F;
      vectors[(size_t)row * (size_t)dims + (size_t)dim] = value;
      norm += (double)value * (double)value;
    }
    if (norm == 0.0) {
      vectors[(size_t)row * (size_t)dims] = 1.0F;
    }
  }
}

static double oracle_distance(const float *a, const float *b, int dims,
                              int metric) {
  double dot = 0.0;
  double a_norm = 0.0;
  double b_norm = 0.0;
  double l2 = 0.0;
  for (int i = 0; i < dims; ++i) {
    double av = a[i];
    double bv = b[i];
    double delta = av - bv;
    dot += av * bv;
    a_norm += av * av;
    b_norm += bv * bv;
    l2 += delta * delta;
  }
  if (metric == METRIC_L2) {
    return sqrt(l2);
  }
  if (metric == METRIC_IP) {
    return -dot;
  }
  return 1.0 - dot / (sqrt(a_norm) * sqrt(b_norm));
}

static int compare_expected(const void *left, const void *right) {
  const Expected *a = (const Expected *)left;
  const Expected *b = (const Expected *)right;
  if (a->distance < b->distance) {
    return -1;
  }
  if (a->distance > b->distance) {
    return 1;
  }
  return a->rowid < b->rowid ? -1 : (a->rowid > b->rowid ? 1 : 0);
}

static void oracle_topk(const float *vectors, int rows, int dims, int metric,
                        const float *query, Expected *items) {
  for (int row = 0; row < rows; ++row) {
    items[row].rowid = (sqlite3_int64)row + 1;
    items[row].distance = oracle_distance(
        query, vectors + (size_t)row * (size_t)dims, dims, metric);
  }
  qsort(items, (size_t)rows, sizeof(*items), compare_expected);
}

static int close_enough(double actual, double expected) {
  double scale = fmax(1.0, fabs(expected));
  return isfinite(actual) && fabs(actual - expected) <= 1e-5 * scale;
}

static const char *metric_name(int metric) {
  if (metric == METRIC_COSINE) {
    return "cosine";
  }
  if (metric == METRIC_IP) {
    return "ip";
  }
  return "l2";
}

static int create_index(sqlite3 *db, const char *table, int dims, int metric,
                        int cache_mb, int threads) {
  char *sql = sqlite3_mprintf(
      "CREATE VIRTUAL TABLE \"%w\" USING hnsw(embedding FLOAT32(%d),"
      "metric=%s,m=8,ef_construction=64,"
      "cache_size_mb=%d,build_memory_mb=256,build_threads=%d)",
      table, dims, metric_name(metric), cache_mb, threads);
  int rc = sql == NULL ? SQLITE_NOMEM : execute(db, sql);
  sqlite3_free(sql);
  return rc;
}

static int drop_index(sqlite3 *db, const char *table) {
  char *sql = sqlite3_mprintf("DROP TABLE \"%w\"", table);
  int rc = sql == NULL ? SQLITE_NOMEM : execute(db, sql);
  sqlite3_free(sql);
  return rc;
}

static int insert_vectors(sqlite3 *db, const char *table, const float *vectors,
                          int rows, int dims) {
  char *sql = sqlite3_mprintf(
      "INSERT INTO \"%w\"(rowid,embedding) VALUES(?,?)", table);
  sqlite3_stmt *statement = NULL;
  unsigned char *blob = NULL;
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  blob = malloc((size_t)dims * 4U);
  if (blob == NULL) {
    sqlite3_finalize(statement);
    return SQLITE_NOMEM;
  }
  rc = execute(db, "BEGIN");
  for (int row = 0; row < rows && rc == SQLITE_OK; ++row) {
    encode_vector(vectors + (size_t)row * (size_t)dims, dims, blob);
    sqlite3_bind_int64(statement, 1, (sqlite3_int64)row + 1);
    sqlite3_bind_blob(statement, 2, blob, dims * 4, SQLITE_TRANSIENT);
    rc = sqlite3_step(statement) == SQLITE_DONE ? SQLITE_OK
                                                : sqlite3_errcode(db);
    sqlite3_reset(statement);
    sqlite3_clear_bindings(statement);
  }
  if (rc == SQLITE_OK) {
    rc = execute(db, "COMMIT");
  } else {
    (void)execute(db, "ROLLBACK");
  }
  free(blob);
  sqlite3_finalize(statement);
  return rc;
}

static int run_query(sqlite3 *db, const char *table, const float *query,
                     int dims, int k, int ef_search, QueryResult *results,
                     int capacity, int *count) {
  char *sql = sqlite3_mprintf(
      "SELECT rowid,distance,visited_count FROM \"%w\" "
      "WHERE embedding MATCH ? AND k=? AND ef_search=? ORDER BY distance",
      table);
  sqlite3_stmt *statement = NULL;
  unsigned char *blob = NULL;
  int rc = SQLITE_OK;
  int found = 0;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  blob = malloc((size_t)dims * 4U);
  if (blob == NULL) {
    sqlite3_finalize(statement);
    return SQLITE_NOMEM;
  }
  encode_vector(query, dims, blob);
  sqlite3_bind_blob(statement, 1, blob, dims * 4, SQLITE_TRANSIENT);
  sqlite3_bind_int(statement, 2, k);
  sqlite3_bind_int(statement, 3, ef_search);
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    if (found >= capacity) {
      rc = SQLITE_TOOBIG;
      break;
    }
    results[found].rowid = sqlite3_column_int64(statement, 0);
    results[found].distance = sqlite3_column_double(statement, 1);
    results[found].visited = sqlite3_column_int(statement, 2);
    ++found;
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  free(blob);
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(db);
  }
  *count = found;
  return rc;
}

static int validate_result_properties(const QueryResult *results, int count,
                                      const float *vectors, int rows, int dims,
                                      int metric, const float *query) {
  for (int i = 0; i < count; ++i) {
    sqlite3_int64 rowid = results[i].rowid;
    if (rowid < 1 || rowid > rows) {
      return 0;
    }
    if (!close_enough(
            results[i].distance,
            oracle_distance(query,
                            vectors + (size_t)(rowid - 1) * (size_t)dims, dims,
                            metric))) {
      return 0;
    }
    if (i > 0 && results[i - 1].distance > results[i].distance) {
      return 0;
    }
    for (int j = 0; j < i; ++j) {
      if (results[j].rowid == rowid) {
        return 0;
      }
    }
  }
  return 1;
}

static int recall_hits(const QueryResult *results, int count,
                       const Expected *expected, int k) {
  int hits = 0;
  for (int i = 0; i < count; ++i) {
    for (int j = 0; j < k; ++j) {
      if (results[i].rowid == expected[j].rowid) {
        ++hits;
        break;
      }
    }
  }
  return hits;
}

static void test_ann_pair(sqlite3 *db, int metric) {
  const int rows = 512;
  const int dims = 16;
  const int queries = 32;
  const int k = 10;
  float *vectors = NULL;
  float *query = NULL;
  Expected *expected = NULL;
  QueryResult page[16];
  QueryResult snapshot[16];
  char page_table[32];
  char snapshot_table[32];
  uint64_t state = UINT64_C(0xc001d00d12345678) ^ (uint64_t)metric;
  int page_created = 0;
  int snapshot_created = 0;
  int hits = 0;
  snprintf(page_table, sizeof(page_table), "ann_page_%d", metric);
  snprintf(snapshot_table, sizeof(snapshot_table), "ann_snapshot_%d", metric);
  vectors = malloc((size_t)rows * (size_t)dims * sizeof(float));
  query = malloc((size_t)dims * sizeof(float));
  expected = malloc((size_t)rows * sizeof(*expected));
  CHECK(vectors != NULL && query != NULL && expected != NULL,
        "allocate ANN correctness buffers");
  fill_vectors(vectors, rows, dims, state);
  CHECK(create_index(db, page_table, dims, metric, 0, 1) == SQLITE_OK,
        "create page-backed ANN index");
  page_created = 1;
  CHECK(create_index(db, snapshot_table, dims, metric, 16, 1) == SQLITE_OK,
        "create snapshot ANN index");
  snapshot_created = 1;
  CHECK(insert_vectors(db, page_table, vectors, rows, dims) == SQLITE_OK,
        "insert page-backed ANN vectors");
  CHECK(insert_vectors(db, snapshot_table, vectors, rows, dims) == SQLITE_OK,
        "insert snapshot ANN vectors");
  for (int query_index = 0; query_index < queries; ++query_index) {
    int page_count = 0;
    int snapshot_count = 0;
    for (int dim = 0; dim < dims; ++dim) {
      query[dim] = random_float(&state) + (float)query_index * 0.000017F;
    }
    if (metric == METRIC_COSINE && query[0] == 0.0F) {
      query[0] = 1.0F;
    }
    oracle_topk(vectors, rows, dims, metric, query, expected);
    CHECK(run_query(db, page_table, query, dims, k, 128, page, 16,
                    &page_count) == SQLITE_OK,
          "run page-backed ANN query");
    CHECK(run_query(db, snapshot_table, query, dims, k, 128, snapshot, 16,
                    &snapshot_count) == SQLITE_OK,
          "run snapshot ANN query");
    CHECK(page_count == k && snapshot_count == k,
          "ANN paths return the requested number of rows");
    CHECK(validate_result_properties(page, page_count, vectors, rows, dims,
                                     metric, query),
          "page-backed ANN result properties");
    CHECK(validate_result_properties(snapshot, snapshot_count, vectors, rows,
                                     dims, metric, query),
          "snapshot ANN result properties");
    for (int i = 0; i < k; ++i) {
      CHECK(page[i].rowid == snapshot[i].rowid,
            "page-backed and snapshot ANN rowids match");
      CHECK(close_enough(page[i].distance, snapshot[i].distance),
            "page-backed and snapshot ANN distances match");
    }
    hits += recall_hits(page, page_count, expected, k);
  }
  CHECK((double)hits / (double)(queries * k) >= 0.95,
        "incremental ANN Recall@10 meets the fixed threshold");
done:
  if (page_created) {
    (void)drop_index(db, page_table);
  }
  if (snapshot_created) {
    (void)drop_index(db, snapshot_table);
  }
  free(vectors);
  free(query);
  free(expected);
}

static void test_arena(sqlite3 *db, int metric) {
  const int rows = 4096;
  const int dims = 8;
  const int queries = 12;
  const int k = 10;
  float *vectors = NULL;
  float query[8];
  Expected *expected = NULL;
  QueryResult results[16];
  char table[32];
  uint64_t state = UINT64_C(0x55aa33cc77ee11dd) ^ (uint64_t)metric;
  int hits = 0;
  int created = 0;
  snprintf(table, sizeof(table), "arena_oracle_%d", metric);
  vectors = malloc((size_t)rows * (size_t)dims * sizeof(float));
  expected = malloc((size_t)rows * sizeof(*expected));
  CHECK(vectors != NULL && expected != NULL, "allocate arena oracle buffers");
  fill_vectors(vectors, rows, dims, state);
  CHECK(create_index(db, table, dims, metric, 0, 4) == SQLITE_OK,
        "create arena oracle index");
  created = 1;
  CHECK(insert_vectors(db, table, vectors, rows, dims) == SQLITE_OK,
        "build arena oracle index");
  for (int query_index = 0; query_index < queries; ++query_index) {
    int count = 0;
    for (int dim = 0; dim < dims; ++dim) {
      query[dim] = random_float(&state) + (float)query_index * 0.000019F;
    }
    if (metric == METRIC_COSINE && query[0] == 0.0F) {
      query[0] = 1.0F;
    }
    oracle_topk(vectors, rows, dims, metric, query, expected);
    CHECK(run_query(db, table, query, dims, k, 128, results, 16, &count) ==
              SQLITE_OK,
          "run arena ANN query");
    CHECK(count == k, "arena ANN returns k rows");
    CHECK(validate_result_properties(results, count, vectors, rows, dims,
                                     metric, query),
          "arena ANN result properties");
    hits += recall_hits(results, count, expected, k);
  }
  CHECK((double)hits / (double)(queries * k) >= 0.95,
        "arena ANN Recall@10 meets the fixed threshold");
done:
  if (created) {
    (void)drop_index(db, table);
  }
  free(vectors);
  free(expected);
}

static int query_expect_error(sqlite3 *db, const char *sql, const float *query,
                              int dims) {
  sqlite3_stmt *statement = NULL;
  unsigned char *blob = malloc((size_t)dims * 4U);
  int rc = SQLITE_NOMEM;
  if (blob == NULL) {
    return 0;
  }
  encode_vector(query, dims, blob);
  rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  if (rc == SQLITE_OK) {
    sqlite3_bind_blob(statement, 1, blob, dims * 4, SQLITE_TRANSIENT);
    rc = sqlite3_step(statement);
  }
  sqlite3_finalize(statement);
  free(blob);
  return rc != SQLITE_ROW && rc != SQLITE_DONE;
}

static void test_inputs(sqlite3 *db) {
  const char *invalid_create[] = {
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(0))",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),metric=nope)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),m=1)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),m=8,"
      "ef_construction=8)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),unknown=1)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),"
      "cache_size_mb=65537)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),"
      "build_memory_mb=127)",
      "CREATE VIRTUAL TABLE bad USING hnsw(embedding FLOAT32(2),"
      "build_threads=65)"};
  float vectors[6] = {1.0F, 1.0F, 2.0F, 2.0F, 3.0F, 3.0F};
  float query[3] = {1.5F, 1.5F, 1.5F};
  QueryResult results[16];
  sqlite3_stmt *statement = NULL;
  unsigned char blob[8];
  int count = 0;
  int created = 0;
  for (size_t i = 0; i < sizeof(invalid_create) / sizeof(invalid_create[0]);
       ++i) {
    CHECK(expect_sql_error(db, invalid_create[i]),
          "invalid CREATE option is rejected");
  }
  CHECK(expect_sql_error(db, "SELECT hnsw_f32('[]')"),
        "empty vector text is rejected");
  CHECK(expect_sql_error(db, "SELECT hnsw_f32('[1,]')"),
        "trailing comma is rejected");
  CHECK(expect_sql_error(db, "SELECT hnsw_f32('[1 2]')"),
        "missing separator is rejected");
  CHECK(expect_sql_error(db, "SELECT hnsw_f32('[1,2')"),
        "unterminated vector is rejected");
  CHECK(expect_sql_error(db, "SELECT hnsw_f32('[1] trailing')"),
        "trailing vector text is rejected");
  CHECK(!expect_sql_error(db, "SELECT hnsw_dims(hnsw_f32('[1e2,-2.5e-1]'))"),
        "exponent vector syntax is accepted");
  CHECK(create_index(db, "bounds", 2, METRIC_L2, 0, 1) == SQLITE_OK,
        "create query-boundary index");
  created = 1;
  CHECK(insert_vectors(db, "bounds", vectors, 3, 2) == SQLITE_OK,
        "insert query-boundary vectors");
  CHECK(run_query(db, "bounds", query, 2, 10, 10, results, 16, &count) ==
            SQLITE_OK &&
            count == 3,
        "k above live count returns every live row");
  CHECK(query_expect_error(
            db, "SELECT rowid FROM bounds WHERE embedding MATCH ? AND k=0", query,
            2),
        "k=0 is rejected");
  CHECK(query_expect_error(db,
                           "SELECT rowid FROM bounds WHERE embedding MATCH ? "
                           "AND k=10001",
                           query, 2),
        "oversized k is rejected");
  CHECK(query_expect_error(db,
                           "SELECT rowid FROM bounds WHERE embedding MATCH ? "
                           "AND k=10 AND ef_search=9",
                           query, 2),
        "ef_search below k is rejected");
  CHECK(query_expect_error(db,
                           "SELECT rowid FROM bounds WHERE embedding MATCH ? "
                           "AND k=1 AND ef_search=1000001",
                           query, 2),
        "oversized ef_search is rejected");
  CHECK(expect_sql_error(
            db, "SELECT rowid FROM bounds WHERE embedding MATCH 'text' AND k=1"),
        "non-BLOB query is rejected");
  CHECK(query_expect_error(db,
                           "SELECT rowid FROM bounds WHERE embedding MATCH ? "
                           "AND k=1",
                           query, 3),
        "query dimension mismatch is rejected");
  CHECK(expect_sql_error(db,
                         "SELECT rowid FROM bounds WHERE embedding MATCH "
                         "hnsw_f32('[1,1]')"),
        "MATCH without k is rejected");
  query[0] = NAN;
  query[1] = 1.0F;
  encode_vector(query, 2, blob);
  CHECK(sqlite3_prepare_v2(db,
                           "INSERT INTO bounds(rowid,embedding) VALUES(99,?)", -1,
                           &statement, NULL) == SQLITE_OK,
        "prepare non-finite insert");
  sqlite3_bind_blob(statement, 1, blob, sizeof(blob), SQLITE_TRANSIENT);
  CHECK(sqlite3_step(statement) != SQLITE_DONE, "NaN vector is rejected");
  sqlite3_finalize(statement);
  statement = NULL;
  CHECK(execute(db, "CREATE VIRTUAL TABLE cosine_zero USING hnsw("
                    "embedding FLOAT32(2),metric=cosine)") == SQLITE_OK,
        "create cosine zero-vector index");
  query[0] = 0.0F;
  query[1] = 0.0F;
  encode_vector(query, 2, blob);
  CHECK(sqlite3_prepare_v2(
            db, "INSERT INTO cosine_zero(rowid,embedding) VALUES(1,?)", -1,
            &statement, NULL) == SQLITE_OK,
        "prepare cosine zero-vector insert");
  sqlite3_bind_blob(statement, 1, blob, sizeof(blob), SQLITE_TRANSIENT);
  CHECK(sqlite3_step(statement) != SQLITE_DONE,
        "cosine zero vector is rejected");
  sqlite3_finalize(statement);
  statement = NULL;
  CHECK(execute(db, "DROP TABLE cosine_zero") == SQLITE_OK,
        "drop cosine zero-vector index");
done:
  sqlite3_finalize(statement);
  if (created) {
    (void)drop_index(db, "bounds");
  }
}

int main(int argc, char **argv) {
  sqlite3 *db = NULL;
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/sqlite_hnsw.so\n", argv[0]);
    return 2;
  }
  if (sqlite3_open(":memory:", &db) != SQLITE_OK ||
      load(db, argv[1]) != SQLITE_OK) {
    sqlite3_close(db);
    return 1;
  }
  for (int metric = METRIC_L2; metric <= METRIC_IP; ++metric) {
    test_ann_pair(db, metric);
    test_arena(db, metric);
  }
  test_inputs(db);
  if (sqlite3_close(db) != SQLITE_OK) {
    fprintf(stderr, "FAIL: sqlite3_close left statements open\n");
    ++failures;
  }
  (void)sqlite3_shutdown();
  if (failures != 0) {
    fprintf(stderr, "%d correctness test(s) failed\n", failures);
    return 1;
  }
  puts("oracle correctness tests passed");
  return 0;
}
