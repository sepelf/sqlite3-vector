#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#ifndef TEST_EXTENSION_PATH
#define TEST_EXTENSION_PATH "./vechnsw.so"
#endif

static double now_sec(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static void die(sqlite3 *db, const char *msg) {
  fprintf(stderr, "error: %s", msg);
  if (db) {
    fprintf(stderr, ": %s", sqlite3_errmsg(db));
  }
  fputc('\n', stderr);
  exit(1);
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

static void load_extension(sqlite3 *db) {
  char *err = 0;
  int rc = sqlite3_enable_load_extension(db, 1);
  if (rc != SQLITE_OK) {
    die(db, "enable load extension");
  }
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
    v[i] = sinf(x * 0.013f) + cosf(x * 0.007f) + (float)(id % 17) * 0.003f;
  }
}

#if defined(__AVX2__)
static float hsum256(__m256 v) {
  float tmp[8];
  _mm256_storeu_ps(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] +
         tmp[7];
}
#endif

static float dot_fast(const float *a, const float *b, int n) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
  }
  float out = hsum256(sum);
  for (; i < n; i++) {
    out += a[i] * b[i];
  }
  return out;
#else
  float out = 0.0f;
  int i;
  for (i = 0; i < n; i++) {
    out += a[i] * b[i];
  }
  return out;
#endif
}

static float cosine_distance(const float *a, const float *b, int dim,
                             float norm_a, float norm_b) {
  if (norm_a <= 0.0f || norm_b <= 0.0f) {
    return 1.0f;
  }
  return 1.0f - dot_fast(a, b, dim) / sqrtf(norm_a * norm_b);
}

static void exact_topk(const float *data, const float *norms, int n, int dim,
                       const float *query, int k, int *ids) {
  float *dists = (float *)malloc((size_t)k * sizeof(float));
  float qnorm = dot_fast(query, query, dim);
  int i, j;
  for (i = 0; i < k; i++) {
    ids[i] = -1;
    dists[i] = INFINITY;
  }
  for (i = 0; i < n; i++) {
    float d = cosine_distance(query, &data[(size_t)i * (size_t)dim], dim, qnorm,
                              norms[i]);
    for (j = 0; j < k; j++) {
      if (d < dists[j]) {
        int m;
        for (m = k - 1; m > j; m--) {
          dists[m] = dists[m - 1];
          ids[m] = ids[m - 1];
        }
        dists[j] = d;
        ids[j] = i + 1;
        break;
      }
    }
  }
  free(dists);
}

static int run_ann(sqlite3 *db, sqlite3_stmt *stmt, const float *query, int dim,
                   int *ids, int k) {
  int rc;
  int count = 0;
  sqlite3_reset(stmt);
  sqlite3_clear_bindings(stmt);
  sqlite3_bind_blob(stmt, 1, query, dim * (int)sizeof(float), SQLITE_STATIC);
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    if (count < k) {
      ids[count++] = sqlite3_column_int(stmt, 0);
    }
  }
  if (rc != SQLITE_DONE) {
    die(db, "run ann query");
  }
  return count;
}

static int contains_id(const int *ids, int n, int id) {
  int i;
  for (i = 0; i < n; i++) {
    if (ids[i] == id) {
      return 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 100000;
  int dim = argc > 2 ? atoi(argv[2]) : 128;
  int queries = argc > 3 ? atoi(argv[3]) : 20;
  int ef_construction = argc > 4 ? atoi(argv[4]) : 128;
  int ef_search = argc > 5 ? atoi(argv[5]) : 64;
  int build_threads = argc > 6 ? atoi(argv[6]) : 8;
  int build_batch_size = argc > 7 ? atoi(argv[7]) : 512;
  int k = 10;
  char db_path[256];
  int db_fd;
  char create_sql[768];
  sqlite3 *db = 0;
  sqlite3_stmt *insert = 0;
  sqlite3_stmt *ann = 0;
  float *data;
  float *norms;
  double t0, t1, insert_sec, build_sec, query_sec;
  double recall = 0.0;
  int qi;
  int rc;

  if (n <= 0 || dim <= 0 || queries <= 0 || ef_construction <= 0 ||
      ef_search <= 0 || build_threads < 2 || build_batch_size < 2) {
    fprintf(stderr,
            "usage: %s [n=100000] [dim=128] [queries=20] "
            "[ef_construction=128] [ef_search=64] [build_threads=8] "
            "[build_batch_size=512]\n",
            argv[0]);
    return 2;
  }

  data = (float *)malloc((size_t)n * (size_t)dim * sizeof(float));
  norms = (float *)malloc((size_t)n * sizeof(float));
  if (!data || !norms) {
    fprintf(stderr, "failed to allocate dataset\n");
    return 1;
  }
  for (int i = 0; i < n; i++) {
    make_vec(&data[(size_t)i * (size_t)dim], dim, i);
    norms[i] = dot_fast(&data[(size_t)i * (size_t)dim],
                        &data[(size_t)i * (size_t)dim], dim);
  }

  snprintf(db_path, sizeof(db_path), "/tmp/vechnsw_bench_XXXXXX");
  db_fd = mkstemp(db_path);
  if (db_fd < 0) {
    perror("mkstemp");
    return 1;
  }
  close(db_fd);
  remove(db_path);
  rc = sqlite3_open(db_path, &db);
  if (rc != SQLITE_OK) {
    die(db, "open database");
  }
  load_extension(db);
  exec_sql(db, "PRAGMA journal_mode=OFF");
  exec_sql(db, "PRAGMA synchronous=OFF");
  exec_sql(db, "PRAGMA temp_store=MEMORY");
  snprintf(create_sql, sizeof(create_sql),
           "CREATE VIRTUAL TABLE doc_vec USING vec_hnsw("
           "dim=%d, metric=cosine, m=16, ef_construction=%d, ef_search=%d, "
           "build_threads=%d, build_batch_size=%d, seed=123)",
           dim, ef_construction, ef_search, build_threads, build_batch_size);
  exec_sql(db, create_sql);

  rc = sqlite3_prepare_v2(
      db, "INSERT INTO doc_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  if (rc != SQLITE_OK) {
    die(db, "prepare insert");
  }
  t0 = now_sec();
  exec_sql(db, "BEGIN");
  for (int i = 0; i < n; i++) {
    sqlite3_reset(insert);
    sqlite3_clear_bindings(insert);
    sqlite3_bind_int64(insert, 1, i + 1);
    sqlite3_bind_blob(insert, 2, &data[(size_t)i * (size_t)dim],
                      dim * (int)sizeof(float), SQLITE_STATIC);
    rc = sqlite3_step(insert);
    if (rc != SQLITE_DONE) {
      die(db, "insert");
    }
  }
  exec_sql(db, "COMMIT");
  t1 = now_sec();
  insert_sec = t1 - t0;
  sqlite3_finalize(insert);

  t0 = now_sec();
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'doc_vec')");
  t1 = now_sec();
  build_sec = t1 - t0;

  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM doc_vec "
      "WHERE embedding MATCH ?1 AND k = 10 AND ef_search = ?2 "
      "ORDER BY distance",
      -1, &ann, 0);
  if (rc != SQLITE_OK) {
    die(db, "prepare ann");
  }

  int *exact_all = (int *)malloc((size_t)queries * (size_t)k * sizeof(int));
  int *ann_all = (int *)malloc((size_t)queries * (size_t)k * sizeof(int));
  if (!exact_all || !ann_all) {
    fprintf(stderr, "failed to allocate query buffers\n");
    return 1;
  }
  for (qi = 0; qi < queries; qi++) {
    int qid = (qi * 7919) % n;
    exact_topk(data, norms, n, dim, &data[(size_t)qid * (size_t)dim], k,
               &exact_all[(size_t)qi * (size_t)k]);
  }
  t0 = now_sec();
  for (qi = 0; qi < queries; qi++) {
    int qid = (qi * 7919) % n;
    int got;
    sqlite3_reset(ann);
    sqlite3_clear_bindings(ann);
    sqlite3_bind_blob(ann, 1, &data[(size_t)qid * (size_t)dim],
                      dim * (int)sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(ann, 2, ef_search);
    got = 0;
    while ((rc = sqlite3_step(ann)) == SQLITE_ROW) {
      if (got < k) {
        ann_all[(size_t)qi * (size_t)k + (size_t)got] =
            sqlite3_column_int(ann, 0);
      }
      got++;
    }
    if (rc != SQLITE_DONE) {
      die(db, "run ann query");
    }
    for (int i = got; i < k; i++) {
      ann_all[(size_t)qi * (size_t)k + (size_t)i] = -1;
    }
  }
  t1 = now_sec();
  query_sec = t1 - t0;
  for (qi = 0; qi < queries; qi++) {
    for (int i = 0; i < k; i++) {
      recall += contains_id(&exact_all[(size_t)qi * (size_t)k], k,
                            ann_all[(size_t)qi * (size_t)k + (size_t)i])
                    ? 1.0
                    : 0.0;
    }
  }
  recall /= (double)(queries * k);

  sqlite3_finalize(ann);
  sqlite3_close(db);

  printf("extension=%s\n", TEST_EXTENSION_PATH);
  printf("database=%s\n", db_path);
  printf("vectors=%d\n", n);
  printf("dim=%d\n", dim);
  printf("metric=cosine\n");
  printf("ef_construction=%d\n", ef_construction);
  printf("ef_search=%d\n", ef_search);
  printf("build_threads=%d\n", build_threads);
  printf("build_batch_size=%d\n", build_batch_size);
  printf("insert_seconds=%.3f\n", insert_sec);
  printf("build_seconds=%.3f\n", build_sec);
  printf("queries=%d\n", queries);
  printf("query_seconds_total=%.3f\n", query_sec);
  printf("query_ms_avg=%.3f\n", query_sec * 1000.0 / (double)queries);
  printf("recall_at_10=%.4f\n", recall);

  free(data);
  free(norms);
  free(exact_all);
  free(ann_all);
  return 0;
}
