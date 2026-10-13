#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <math.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifndef TEST_EXTENSION_PATH
#define TEST_EXTENSION_PATH "./vechnsw.so"
#endif

#define SIFT_DIM 128
#define SIFT_GT_WIDTH 100
#define BENCH_K 10

typedef struct SiftPaths {
  char base[512];
  char query[512];
  char groundtruth[512];
} SiftPaths;

typedef struct QueryStats {
  double *latencies_ms;
  int count;
  double total_seconds;
  double recall_at_1;
  double recall_at_10;
} QueryStats;

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

static void die_errno(const char *msg, const char *path) {
  fprintf(stderr, "error: %s: %s: %s\n", msg, path, strerror(errno));
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

static void make_paths(const char *dir, SiftPaths *paths) {
  snprintf(paths->base, sizeof(paths->base), "%s/sift_base.fvecs", dir);
  snprintf(paths->query, sizeof(paths->query), "%s/sift_query.fvecs", dir);
  snprintf(paths->groundtruth, sizeof(paths->groundtruth),
           "%s/sift_groundtruth.ivecs", dir);
}

static long file_size(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) {
    die_errno("stat failed", path);
  }
  return (long)st.st_size;
}

static int read_i32(FILE *f, const char *path) {
  int32_t v = 0;
  if (fread(&v, sizeof(v), 1, f) != 1) {
    fprintf(stderr, "error: failed to read int32 from %s\n", path);
    exit(1);
  }
  return (int)v;
}

static int fvecs_count(const char *path, int *dim_out) {
  FILE *f = fopen(path, "rb");
  long bytes;
  int dim;
  long record_size;
  if (!f) {
    die_errno("open failed", path);
  }
  dim = read_i32(f, path);
  fclose(f);
  if (dim <= 0) {
    fprintf(stderr, "error: invalid fvecs dimension %d in %s\n", dim, path);
    exit(1);
  }
  bytes = file_size(path);
  record_size = 4L + (long)dim * 4L;
  if (bytes % record_size != 0) {
    fprintf(stderr, "error: %s size is not aligned to fvecs dim %d\n", path,
            dim);
    exit(1);
  }
  *dim_out = dim;
  return (int)(bytes / record_size);
}

static int ivecs_count(const char *path, int *width_out) {
  FILE *f = fopen(path, "rb");
  long bytes;
  int width;
  long record_size;
  if (!f) {
    die_errno("open failed", path);
  }
  width = read_i32(f, path);
  fclose(f);
  if (width <= 0) {
    fprintf(stderr, "error: invalid ivecs width %d in %s\n", width, path);
    exit(1);
  }
  bytes = file_size(path);
  record_size = 4L + (long)width * 4L;
  if (bytes % record_size != 0) {
    fprintf(stderr, "error: %s size is not aligned to ivecs width %d\n", path,
            width);
    exit(1);
  }
  *width_out = width;
  return (int)(bytes / record_size);
}

static void validate_dataset(const SiftPaths *paths, int *base_count,
                             int *query_count, int *gt_count, int *gt_width) {
  int base_dim;
  int query_dim;
  *base_count = fvecs_count(paths->base, &base_dim);
  *query_count = fvecs_count(paths->query, &query_dim);
  *gt_count = ivecs_count(paths->groundtruth, gt_width);
  if (base_dim != SIFT_DIM || query_dim != SIFT_DIM) {
    fprintf(stderr, "error: expected SIFT dim %d, got base=%d query=%d\n",
            SIFT_DIM, base_dim, query_dim);
    exit(1);
  }
  if (*gt_width < BENCH_K) {
    fprintf(stderr, "error: groundtruth width %d is smaller than k=%d\n",
            *gt_width, BENCH_K);
    exit(1);
  }
  if (*gt_count < *query_count) {
    fprintf(stderr, "error: groundtruth rows %d < query rows %d\n", *gt_count,
            *query_count);
    exit(1);
  }
}

static FILE *open_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    die_errno("open failed", path);
  }
  return f;
}

static void load_base_vectors(sqlite3 *db, const char *path, int base_limit) {
  FILE *f = open_file(path);
  sqlite3_stmt *insert = 0;
  float vec[SIFT_DIM];
  int rc = sqlite3_prepare_v2(
      db, "INSERT INTO sift_vec(rowid, embedding) VALUES (?1, ?2)", -1,
      &insert, 0);
  if (rc != SQLITE_OK) {
    die(db, "prepare insert");
  }
  exec_sql(db, "BEGIN");
  for (int i = 0; i < base_limit; i++) {
    int dim = read_i32(f, path);
    if (dim != SIFT_DIM) {
      fprintf(stderr, "error: base vector %d has dim %d, expected %d\n", i, dim,
              SIFT_DIM);
      exit(1);
    }
    if (fread(vec, sizeof(float), SIFT_DIM, f) != SIFT_DIM) {
      fprintf(stderr, "error: failed to read base vector %d\n", i);
      exit(1);
    }
    sqlite3_reset(insert);
    sqlite3_clear_bindings(insert);
    sqlite3_bind_int64(insert, 1, (sqlite3_int64)i + 1);
    sqlite3_bind_blob(insert, 2, vec, sizeof(vec), SQLITE_STATIC);
    rc = sqlite3_step(insert);
    if (rc != SQLITE_DONE) {
      die(db, "insert base vector");
    }
  }
  exec_sql(db, "COMMIT");
  sqlite3_finalize(insert);
  fclose(f);
}

static float *load_query_vectors(const char *path, int query_limit) {
  FILE *f = open_file(path);
  float *queries =
      (float *)malloc((size_t)query_limit * SIFT_DIM * sizeof(float));
  if (!queries) {
    fprintf(stderr, "error: failed to allocate query vectors\n");
    exit(1);
  }
  for (int i = 0; i < query_limit; i++) {
    int dim = read_i32(f, path);
    if (dim != SIFT_DIM) {
      fprintf(stderr, "error: query vector %d has dim %d, expected %d\n", i,
              dim, SIFT_DIM);
      exit(1);
    }
    if (fread(&queries[(size_t)i * SIFT_DIM], sizeof(float), SIFT_DIM, f) !=
        SIFT_DIM) {
      fprintf(stderr, "error: failed to read query vector %d\n", i);
      exit(1);
    }
  }
  fclose(f);
  return queries;
}

static int *load_groundtruth(const char *path, int query_limit, int gt_width) {
  FILE *f = open_file(path);
  int *gt = (int *)malloc((size_t)query_limit * BENCH_K * sizeof(int));
  if (!gt) {
    fprintf(stderr, "error: failed to allocate groundtruth\n");
    exit(1);
  }
  for (int i = 0; i < query_limit; i++) {
    int width = read_i32(f, path);
    if (width != gt_width) {
      fprintf(stderr, "error: groundtruth row %d has width %d, expected %d\n",
              i, width, gt_width);
      exit(1);
    }
    for (int j = 0; j < gt_width; j++) {
      int id = read_i32(f, path);
      if (j < BENCH_K) {
        gt[(size_t)i * (size_t)BENCH_K + (size_t)j] = id;
      }
    }
  }
  fclose(f);
  return gt;
}

static int contains_id(const int *ids, int n, int id) {
  for (int i = 0; i < n; i++) {
    if (ids[i] == id) {
      return 1;
    }
  }
  return 0;
}

static int compare_double(const void *a, const void *b) {
  double da = *(const double *)a;
  double db = *(const double *)b;
  return (da > db) - (da < db);
}

static double percentile(double *values, int count, double p) {
  int idx;
  if (count <= 0) {
    return 0.0;
  }
  idx = (int)ceil((double)count * p) - 1;
  if (idx < 0) {
    idx = 0;
  }
  if (idx >= count) {
    idx = count - 1;
  }
  return values[idx];
}

static QueryStats run_queries(sqlite3 *db, const float *queries, const int *gt,
                              int query_limit, int base_limit, int ef_search) {
  sqlite3_stmt *ann = 0;
  QueryStats stats;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM sift_vec "
      "WHERE embedding MATCH ?1 AND k = 10 AND ef_search = ?2 "
      "ORDER BY distance",
      -1, &ann, 0);
  if (rc != SQLITE_OK) {
    die(db, "prepare ann query");
  }
  memset(&stats, 0, sizeof(stats));
  stats.count = query_limit;
  stats.latencies_ms = (double *)malloc((size_t)query_limit * sizeof(double));
  if (!stats.latencies_ms) {
    fprintf(stderr, "error: failed to allocate latency buffer\n");
    exit(1);
  }

  double t_all0 = now_sec();
  for (int i = 0; i < query_limit; i++) {
    int result_ids[BENCH_K];
    int got = 0;
    double t0;
    double t1;
    for (int j = 0; j < BENCH_K; j++) {
      result_ids[j] = -1;
    }
    sqlite3_reset(ann);
    sqlite3_clear_bindings(ann);
    sqlite3_bind_blob(ann, 1, &queries[(size_t)i * SIFT_DIM],
                      SIFT_DIM * (int)sizeof(float), SQLITE_STATIC);
    sqlite3_bind_int(ann, 2, ef_search);
    t0 = now_sec();
    while ((rc = sqlite3_step(ann)) == SQLITE_ROW) {
      if (got < BENCH_K) {
        sqlite3_int64 rowid = sqlite3_column_int64(ann, 0);
        int base_id = (int)(rowid - 1);
        result_ids[got] = base_id;
      }
      got++;
    }
    t1 = now_sec();
    if (rc != SQLITE_DONE) {
      die(db, "run ann query");
    }
    stats.latencies_ms[i] = (t1 - t0) * 1000.0;
    if (got > 0 && result_ids[0] == gt[(size_t)i * BENCH_K]) {
      stats.recall_at_1 += 1.0;
    }
    for (int j = 0; j < BENCH_K; j++) {
      int id = result_ids[j];
      if (id >= 0 && id < base_limit &&
          contains_id(&gt[(size_t)i * BENCH_K], BENCH_K, id)) {
        stats.recall_at_10 += 1.0;
      }
    }
  }
  stats.total_seconds = now_sec() - t_all0;
  stats.recall_at_1 /= (double)query_limit;
  stats.recall_at_10 /= (double)(query_limit * BENCH_K);
  qsort(stats.latencies_ms, (size_t)query_limit, sizeof(double),
        compare_double);
  sqlite3_finalize(ann);
  return stats;
}

int main(int argc, char **argv) {
  const char *dataset_dir = argc > 1 ? argv[1] : "/tmp/sift1m-fvecs";
  int base_limit = argc > 2 ? atoi(argv[2]) : 1000000;
  int query_limit = argc > 3 ? atoi(argv[3]) : 10000;
  int ef_search = argc > 4 ? atoi(argv[4]) : 64;
  int build_threads = argc > 5 ? atoi(argv[5]) : 8;
  int build_batch_size = argc > 6 ? atoi(argv[6]) : 512;
  int base_count = 0;
  int query_count = 0;
  int gt_count = 0;
  int gt_width = 0;
  int db_fd;
  char db_path[256];
  char create_sql[768];
  double t0;
  double load_seconds;
  double build_seconds;
  long database_bytes;
  SiftPaths paths;
  sqlite3 *db = 0;
  float *queries = 0;
  int *gt = 0;
  QueryStats stats;
  int rc;

  if (base_limit <= 0 || query_limit <= 0 || ef_search <= 0 ||
      build_threads < 2 || build_batch_size < 2) {
    fprintf(stderr,
            "usage: %s [dataset_dir=/tmp/sift1m-fvecs] [base_limit=1000000] "
            "[query_limit=10000] [ef_search=64] [build_threads=8] "
            "[build_batch_size=512]\n",
            argv[0]);
    return 2;
  }

  make_paths(dataset_dir, &paths);
  validate_dataset(&paths, &base_count, &query_count, &gt_count, &gt_width);
  if (base_limit > base_count) {
    base_limit = base_count;
  }
  if (query_limit > query_count) {
    query_limit = query_count;
  }
  if (query_limit > gt_count) {
    query_limit = gt_count;
  }

  queries = load_query_vectors(paths.query, query_limit);
  gt = load_groundtruth(paths.groundtruth, query_limit, gt_width);

  snprintf(db_path, sizeof(db_path), "/tmp/vechnsw_sift_XXXXXX");
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
           "CREATE VIRTUAL TABLE sift_vec USING vec_hnsw("
           "dim=128, metric=l2, m=16, ef_construction=128, ef_search=%d, "
           "build_threads=%d, build_batch_size=%d, seed=123)",
           ef_search, build_threads, build_batch_size);
  exec_sql(db, create_sql);

  t0 = now_sec();
  load_base_vectors(db, paths.base, base_limit);
  load_seconds = now_sec() - t0;

  t0 = now_sec();
  exec_sql(db, "SELECT vec_hnsw_rebuild('main', 'sift_vec')");
  build_seconds = now_sec() - t0;

  stats = run_queries(db, queries, gt, query_limit, base_limit, ef_search);
  sqlite3_close(db);
  database_bytes = file_size(db_path);

  printf("dataset=sift1m\n");
  printf("dataset_dir=%s\n", dataset_dir);
  printf("database=%s\n", db_path);
  printf("base_file_vectors=%d\n", base_count);
  printf("query_file_vectors=%d\n", query_count);
  printf("groundtruth_file_vectors=%d\n", gt_count);
  printf("groundtruth_width=%d\n", gt_width);
  printf("base_vectors=%d\n", base_limit);
  printf("query_vectors=%d\n", query_limit);
  printf("dim=%d\n", SIFT_DIM);
  printf("metric=l2\n");
  printf("m=16\n");
  printf("ef_construction=128\n");
  printf("ef_search=%d\n", ef_search);
  printf("build_threads=%d\n", build_threads);
  printf("build_batch_size=%d\n", build_batch_size);
  printf("k=%d\n", BENCH_K);
  printf("load_seconds=%.3f\n", load_seconds);
  printf("build_seconds=%.3f\n", build_seconds);
  printf("query_seconds_total=%.3f\n", stats.total_seconds);
  printf("query_ms_p50=%.3f\n", percentile(stats.latencies_ms, stats.count, 0.50));
  printf("query_ms_p95=%.3f\n", percentile(stats.latencies_ms, stats.count, 0.95));
  printf("query_ms_p99=%.3f\n", percentile(stats.latencies_ms, stats.count, 0.99));
  printf("qps=%.3f\n", (double)query_limit / stats.total_seconds);
  printf("recall_at_1=%.4f\n", stats.recall_at_1);
  printf("recall_at_10=%.4f\n", stats.recall_at_10);
  printf("recall_scope=full_groundtruth_top10\n");
  printf("database_bytes=%ld\n", database_bytes);

  free(stats.latencies_ms);
  free(queries);
  free(gt);
  return 0;
}
