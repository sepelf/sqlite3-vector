#define _POSIX_C_SOURCE 200809L

#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SQLITE_DETERMINISTIC
#define SQLITE_DETERMINISTIC 0
#endif

#define VH_DEFAULT_M 16
#define VH_DEFAULT_EF_CONSTRUCTION 128
#define VH_DEFAULT_EF_SEARCH 64
#define VH_DEFAULT_BUILD_THREADS 8
#define VH_DEFAULT_BUILD_BATCH_SIZE 512
#define VH_DEFAULT_SEED 0x9e3779b97f4a7c15ULL
#define VH_DEFAULT_SEED_I64 (-7046029254386353131LL)
#define VH_MAX_DIM 4096
#define VH_MAX_M 128
#define VH_MAX_EF 8192
#define VH_MAX_BUILD_THREADS 256
#define VH_MAX_BUILD_BATCH_SIZE 65536
#define VH_MAX_LEVEL 32

#define VH_IDX_MATCH 1
#define VH_IDX_K 2
#define VH_IDX_EF 4

_Static_assert(sizeof(int) == sizeof(int32_t),
               "vechnsw adjacency BLOBs require 32-bit int");

typedef enum VhMetric {
  VH_METRIC_L2 = 0,
  VH_METRIC_COSINE = 1,
  VH_METRIC_IP = 2
} VhMetric;

typedef struct VhConfig {
  int dim;
  int m;
  int ef_construction;
  int ef_search;
  int build_threads;
  int build_batch_size;
  VhMetric metric;
  uint64_t seed;
} VhConfig;

typedef struct VhNeighborList {
  int count;
  int cap;
  int *ids;
  float *dists;
} VhNeighborList;

typedef struct VhCandidate {
  int id;
  float dist;
} VhCandidate;

typedef struct VhHeap {
  VhCandidate *items;
  int count;
  int cap;
  int is_min;
} VhHeap;

typedef struct VhCandidateList {
  VhCandidate *items;
  int count;
  int cap;
} VhCandidateList;

typedef struct VhPlannedEdge {
  int node;
  int layer;
  int neighbor;
  float dist;
} VhPlannedEdge;

typedef struct VhEdgeList {
  VhPlannedEdge *items;
  int count;
  int cap;
} VhEdgeList;

typedef struct VhTouched {
  int node;
  int layer;
} VhTouched;

typedef struct VhTouchedList {
  VhTouched *items;
  int count;
  int cap;
} VhTouchedList;

typedef struct VhScratch {
  int *visited;
  int visit_mark;
  VhHeap work_candidates;
  VhHeap work_best;
  VhCandidateList work_results;
  VhCandidateList work_prune_candidates;
  VhCandidateList work_pruned;
  int *work_selected;
  float *work_selected_dists;
  int work_selected_cap;
  int *work_prune_selected;
  float *work_prune_selected_dists;
  int work_prune_selected_cap;
} VhScratch;

typedef struct VhGraph {
  VhConfig cfg;
  int n;
  int active_count;
  int entry_node;
  int max_level;
  int alloc_levels;
  int rewrite_vectors;
  float *vectors;
  float *stored_vectors;
  sqlite3_int64 *rowids;
  unsigned char *deleted;
  int *levels;
  VhNeighborList *links;
} VhGraph;

typedef enum VhBuildTaskKind {
  VH_BUILD_TASK_PLAN = 0,
  VH_BUILD_TASK_PRUNE = 1
} VhBuildTaskKind;

typedef struct VhBuildPool VhBuildPool;

typedef struct VhBuildWorker {
  VhBuildPool *pool;
  pthread_t thread;
  int started;
  VhScratch scratch;
  VhEdgeList edges;
  const int *active_ids;
  const VhTouched *touched;
  int begin;
  int end;
  int entry_node;
  int max_level;
  int rc;
} VhBuildWorker;

struct VhBuildPool {
  VhGraph *g;
  int worker_count;
  VhBuildWorker *workers;
  pthread_mutex_t mutex;
  pthread_cond_t start_cond;
  pthread_cond_t done_cond;
  int mutex_ready;
  int start_cond_ready;
  int done_cond_ready;
  VhBuildTaskKind task_kind;
  int generation;
  int done_count;
  int stop;
};

typedef struct VhTable {
  sqlite3_vtab base;
  sqlite3 *db;
  char *db_name;
  char *table_name;
  char *meta_name;
  char *vectors_name;
  char *graph_name;
  VhConfig cfg;
  sqlite3_int64 cache_generation;
  sqlite3_int64 cache_data_version;
  VhGraph *cache;
} VhTable;

typedef struct VhCursor {
  sqlite3_vtab_cursor base;
  VhTable *tab;
  int mode;
  int eof;
  sqlite3_stmt *scan_stmt;
  sqlite3_int64 *rowids;
  float *distances;
  int result_count;
  int result_pos;
} VhCursor;

static void *vh_malloc(sqlite3_uint64 n) {
  if (n == 0) {
    n = 1;
  }
  return sqlite3_malloc64(n);
}

static void *vh_realloc(void *p, sqlite3_uint64 n) {
  if (n == 0) {
    n = 1;
  }
  return sqlite3_realloc64(p, n);
}

static int vh_mul_overflow_i64(sqlite3_uint64 a, sqlite3_uint64 b,
                               sqlite3_uint64 *out) {
  if (a != 0 && b > UINT64_MAX / a) {
    return 1;
  }
  *out = a * b;
  return 0;
}

static char *vh_strdup(const char *z) {
  size_t n = strlen(z) + 1;
  char *out = (char *)vh_malloc((sqlite3_uint64)n);
  if (!out) {
    return 0;
  }
  memcpy(out, z, n);
  return out;
}

static int vh_vtab_error(sqlite3_vtab *tab, const char *fmt, ...) {
  va_list ap;
  sqlite3_free(tab->zErrMsg);
  va_start(ap, fmt);
  tab->zErrMsg = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  return SQLITE_ERROR;
}

static int vh_execf(sqlite3 *db, char **pzErr, const char *fmt, ...) {
  va_list ap;
  char *sql;
  int rc;
  va_start(ap, fmt);
  sql = sqlite3_vmprintf(fmt, ap);
  va_end(ap);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_exec(db, sql, 0, 0, pzErr);
  sqlite3_free(sql);
  return rc;
}

static char *vh_quote_ident(const char *z) { return sqlite3_mprintf("\"%w\"", z); }

static char *vh_qualified(const char *db, const char *table) {
  char *qdb = vh_quote_ident(db);
  char *qt = vh_quote_ident(table);
  char *out;
  if (!qdb || !qt) {
    sqlite3_free(qdb);
    sqlite3_free(qt);
    return 0;
  }
  out = sqlite3_mprintf("%s.%s", qdb, qt);
  sqlite3_free(qdb);
  sqlite3_free(qt);
  return out;
}

static char *vh_shadow_name(const char *table, const char *suffix) {
  return sqlite3_mprintf("%s_%s", table, suffix);
}

static const char *vh_metric_name(VhMetric metric) {
  switch (metric) {
  case VH_METRIC_L2:
    return "l2";
  case VH_METRIC_COSINE:
    return "cosine";
  case VH_METRIC_IP:
    return "ip";
  }
  return "cosine";
}

static int vh_parse_metric(const char *z, VhMetric *metric) {
  if (sqlite3_stricmp(z, "l2") == 0 || sqlite3_stricmp(z, "euclidean") == 0) {
    *metric = VH_METRIC_L2;
    return SQLITE_OK;
  }
  if (sqlite3_stricmp(z, "cosine") == 0) {
    *metric = VH_METRIC_COSINE;
    return SQLITE_OK;
  }
  if (sqlite3_stricmp(z, "ip") == 0 ||
      sqlite3_stricmp(z, "inner_product") == 0) {
    *metric = VH_METRIC_IP;
    return SQLITE_OK;
  }
  return SQLITE_ERROR;
}

static int vh_parse_positive_int(const char *z, int *out) {
  char *end = 0;
  long v = strtol(z, &end, 10);
  if (!z[0] || (end && *end)) {
    return SQLITE_ERROR;
  }
  if (v <= 0 || v > 1000000000) {
    return SQLITE_ERROR;
  }
  *out = (int)v;
  return SQLITE_OK;
}

static int vh_validate_dim(int dim) { return dim >= 1 && dim <= VH_MAX_DIM; }

static int vh_config_error(char **pzErr, const char *name, int min, int max) {
  if (pzErr) {
    *pzErr = sqlite3_mprintf("vec_hnsw requires %s between %d and %d", name,
                             min, max);
  }
  return SQLITE_ERROR;
}

static int vh_validate_config(VhConfig *cfg, int adjust_ef_construction,
                              char **pzErr) {
  if (!vh_validate_dim(cfg->dim)) {
    return vh_config_error(pzErr, "dim", 1, VH_MAX_DIM);
  }
  if (cfg->m < 2 || cfg->m > VH_MAX_M) {
    return vh_config_error(pzErr, "m", 2, VH_MAX_M);
  }
  if (adjust_ef_construction && cfg->ef_construction < cfg->m) {
    cfg->ef_construction = cfg->m;
  }
  if (cfg->ef_construction < cfg->m ||
      cfg->ef_construction > VH_MAX_EF) {
    if (pzErr) {
      *pzErr = sqlite3_mprintf(
          "vec_hnsw requires ef_construction between m and %d", VH_MAX_EF);
    }
    return SQLITE_ERROR;
  }
  if (cfg->ef_search < 1 || cfg->ef_search > VH_MAX_EF) {
    return vh_config_error(pzErr, "ef_search", 1, VH_MAX_EF);
  }
  if (cfg->build_threads < 2 ||
      cfg->build_threads > VH_MAX_BUILD_THREADS) {
    return vh_config_error(pzErr, "build_threads", 2,
                           VH_MAX_BUILD_THREADS);
  }
  if (cfg->build_batch_size < 2 ||
      cfg->build_batch_size > VH_MAX_BUILD_BATCH_SIZE) {
    return vh_config_error(pzErr, "build_batch_size", 2,
                           VH_MAX_BUILD_BATCH_SIZE);
  }
  return SQLITE_OK;
}

static int vh_f32_is_finite(float x) {
  uint32_t bits;
  memcpy(&bits, &x, sizeof(bits));
  return (bits & 0x7f800000u) != 0x7f800000u;
}

static int vh_validate_finite_f32(const void *blob, int dim) {
  const float *v = (const float *)blob;
  int i;
  for (i = 0; i < dim; i++) {
    if (!vh_f32_is_finite(v[i])) {
      return SQLITE_ERROR;
    }
  }
  return SQLITE_OK;
}

static int vh_i64_to_int(sqlite3_int64 value, int *out) {
  if (value < INT_MIN || value > INT_MAX) {
    return SQLITE_ERROR;
  }
  *out = (int)value;
  return SQLITE_OK;
}

static int vh_parse_i64_strict(const char *z, sqlite3_int64 *out) {
  sqlite3_int64 value = 0;
  int sign = 1;
  const char *p = z;
  if (!p || !p[0]) {
    return SQLITE_ERROR;
  }
  if (*p == '-') {
    sign = -1;
    p++;
    if (!*p) {
      return SQLITE_ERROR;
    }
  } else if (*p == '+') {
    p++;
    if (!*p) {
      return SQLITE_ERROR;
    }
  }
  for (; *p; p++) {
    int digit;
    if (*p < '0' || *p > '9') {
      return SQLITE_ERROR;
    }
    digit = *p - '0';
    if (sign > 0) {
      if (value > (LLONG_MAX - digit) / 10) {
        return SQLITE_ERROR;
      }
      value = value * 10 + digit;
    } else {
      if (value < (LLONG_MIN + digit) / 10) {
        return SQLITE_ERROR;
      }
      value = value * 10 - digit;
    }
  }
  *out = value;
  return SQLITE_OK;
}

static int vh_parse_args(int argc, const char *const *argv, VhConfig *cfg,
                         char **pzErr) {
  int i;
  cfg->dim = -1;
  cfg->m = VH_DEFAULT_M;
  cfg->ef_construction = VH_DEFAULT_EF_CONSTRUCTION;
  cfg->ef_search = VH_DEFAULT_EF_SEARCH;
  cfg->build_threads = VH_DEFAULT_BUILD_THREADS;
  cfg->build_batch_size = VH_DEFAULT_BUILD_BATCH_SIZE;
  cfg->metric = VH_METRIC_COSINE;
  cfg->seed = VH_DEFAULT_SEED;

  for (i = 3; i < argc; i++) {
    const char *arg = argv[i];
    const char *eq = strchr(arg, '=');
    int key_len;
    const char *value;
    if (!eq) {
      *pzErr = sqlite3_mprintf("vec_hnsw option must be key=value: %s", arg);
      return SQLITE_ERROR;
    }
    key_len = (int)(eq - arg);
    value = eq + 1;
    if (key_len == 3 && sqlite3_strnicmp(arg, "dim", 3) == 0) {
      if (vh_parse_positive_int(value, &cfg->dim) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid dim: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 6 && sqlite3_strnicmp(arg, "metric", 6) == 0) {
      if (vh_parse_metric(value, &cfg->metric) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid metric: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 1 && sqlite3_strnicmp(arg, "m", 1) == 0) {
      if (vh_parse_positive_int(value, &cfg->m) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid m: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 15 &&
               sqlite3_strnicmp(arg, "ef_construction", 15) == 0) {
      if (vh_parse_positive_int(value, &cfg->ef_construction) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid ef_construction: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 9 &&
               sqlite3_strnicmp(arg, "ef_search", 9) == 0) {
      if (vh_parse_positive_int(value, &cfg->ef_search) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid ef_search: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 13 &&
               sqlite3_strnicmp(arg, "build_threads", 13) == 0) {
      if (vh_parse_positive_int(value, &cfg->build_threads) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid build_threads: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 16 &&
               sqlite3_strnicmp(arg, "build_batch_size", 16) == 0) {
      if (vh_parse_positive_int(value, &cfg->build_batch_size) != SQLITE_OK) {
        *pzErr = sqlite3_mprintf("invalid build_batch_size: %s", value);
        return SQLITE_ERROR;
      }
    } else if (key_len == 4 && sqlite3_strnicmp(arg, "seed", 4) == 0) {
      char *end = 0;
      long long seed = strtoll(value, &end, 10);
      if (!value[0] || (end && *end)) {
        *pzErr = sqlite3_mprintf("invalid seed: %s", value);
        return SQLITE_ERROR;
      }
      cfg->seed = seed > 0 ? (uint64_t)seed : VH_DEFAULT_SEED;
    } else {
      *pzErr = sqlite3_mprintf("unknown vec_hnsw option: %s", arg);
      return SQLITE_ERROR;
    }
  }

  if (cfg->dim <= 0) {
    *pzErr = sqlite3_mprintf("vec_hnsw requires dim=N");
    return SQLITE_ERROR;
  }
  return vh_validate_config(cfg, 1, pzErr);
}

static int vh_validate_blob(sqlite3_value *value, int dim, const void **blob) {
  int n = sqlite3_value_bytes(value);
  if (!vh_validate_dim(dim) || sqlite3_value_type(value) != SQLITE_BLOB ||
      n != dim * (int)sizeof(float)) {
    return SQLITE_ERROR;
  }
  *blob = sqlite3_value_blob(value);
  if (!*blob) {
    return SQLITE_NOMEM;
  }
  if (vh_validate_finite_f32(*blob, dim) != SQLITE_OK) {
    return SQLITE_MISMATCH;
  }
  return SQLITE_OK;
}

static float vh_dot_scalar(const float *a, const float *b, int n) {
  float sum = 0.0f;
  int i;
  for (i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

static float vh_l2_scalar(const float *a, const float *b, int n) {
  float sum = 0.0f;
  int i;
  for (i = 0; i < n; i++) {
    float d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

#if defined(__AVX2__)
static float vh_hsum256(__m256 v) {
  float tmp[8];
  _mm256_storeu_ps(tmp, v);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] +
         tmp[7];
}
#endif

static float vh_dot_fast(const float *a, const float *b, int n) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(av, bv));
  }
  float out = vh_hsum256(sum);
  for (; i < n; i++) {
    out += a[i] * b[i];
  }
  return out;
#else
  return vh_dot_scalar(a, b, n);
#endif
}

static float vh_l2_fast(const float *a, const float *b, int n) {
#if defined(__AVX2__)
  __m256 sum = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 av = _mm256_loadu_ps(a + i);
    __m256 bv = _mm256_loadu_ps(b + i);
    __m256 dv = _mm256_sub_ps(av, bv);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(dv, dv));
  }
  float out = vh_hsum256(sum);
  for (; i < n; i++) {
    float d = a[i] - b[i];
    out += d * d;
  }
  return out;
#else
  return vh_l2_scalar(a, b, n);
#endif
}

static void vh_normalize(float *v, int n) {
  float norm2 = vh_dot_fast(v, v, n);
  if (norm2 <= 0.0f) {
    return;
  }
  float inv = 1.0f / sqrtf(norm2);
  int i;
  for (i = 0; i < n; i++) {
    v[i] *= inv;
  }
}

static float vh_metric_rank(const VhGraph *g, const float *a, const float *b) {
  switch (g->cfg.metric) {
  case VH_METRIC_L2:
    return vh_l2_fast(a, b, g->cfg.dim);
  case VH_METRIC_IP:
    return -vh_dot_fast(a, b, g->cfg.dim);
  case VH_METRIC_COSINE:
    return 1.0f - vh_dot_fast(a, b, g->cfg.dim);
  }
  return 0.0f;
}

static float vh_metric_output(const VhGraph *g, float rank) {
  if (g->cfg.metric == VH_METRIC_L2) {
    return sqrtf(rank);
  }
  return rank;
}

static uint64_t vh_xorshift64(uint64_t *state) {
  uint64_t x = *state;
  if (x == 0) {
    x = VH_DEFAULT_SEED;
  }
  x ^= x << 13;
  x ^= x >> 7;
  x ^= x << 17;
  *state = x;
  return x;
}

static float vh_rng_unit(uint64_t *state) {
  uint64_t x = vh_xorshift64(state);
  return (float)((x >> 40) + 1) / (float)(1u << 24);
}

static int vh_random_level(uint64_t *state, int m) {
  int level = 0;
  float p = 1.0f / (float)m;
  while (level < VH_MAX_LEVEL && vh_rng_unit(state) < p) {
    level++;
  }
  return level;
}

static int vh_max_neighbors(const VhConfig *cfg, int layer) {
  return layer == 0 ? cfg->m * 2 : cfg->m;
}

static VhNeighborList *vh_links_at(VhGraph *g, int node, int layer) {
  if (node < 0 || node >= g->n || layer < 0 || layer >= g->alloc_levels) {
    return 0;
  }
  return &g->links[layer * g->n + node];
}

static const VhNeighborList *vh_links_at_const(const VhGraph *g, int node,
                                               int layer) {
  if (node < 0 || node >= g->n || layer < 0 || layer >= g->alloc_levels) {
    return 0;
  }
  return &g->links[layer * g->n + node];
}

static int vh_scratch_init(VhScratch *s, int n) {
  memset(s, 0, sizeof(*s));
  if (n <= 0) {
    return SQLITE_OK;
  }
  s->visited = (int *)vh_malloc((sqlite3_uint64)n * sizeof(int));
  if (!s->visited) {
    return SQLITE_NOMEM;
  }
  memset(s->visited, 0, (size_t)n * sizeof(int));
  return SQLITE_OK;
}

static void vh_scratch_free(VhScratch *s) {
  if (!s) {
    return;
  }
  sqlite3_free(s->visited);
  sqlite3_free(s->work_candidates.items);
  sqlite3_free(s->work_best.items);
  sqlite3_free(s->work_results.items);
  sqlite3_free(s->work_prune_candidates.items);
  sqlite3_free(s->work_pruned.items);
  sqlite3_free(s->work_selected);
  sqlite3_free(s->work_selected_dists);
  sqlite3_free(s->work_prune_selected);
  sqlite3_free(s->work_prune_selected_dists);
  memset(s, 0, sizeof(*s));
}

static void vh_graph_free(VhGraph *g) {
  int i;
  if (!g) {
    return;
  }
  if (g->links) {
    int total = g->n * g->alloc_levels;
    for (i = 0; i < total; i++) {
      sqlite3_free(g->links[i].ids);
      sqlite3_free(g->links[i].dists);
    }
  }
  sqlite3_free(g->links);
  sqlite3_free(g->vectors);
  sqlite3_free(g->stored_vectors);
  sqlite3_free(g->rowids);
  sqlite3_free(g->deleted);
  sqlite3_free(g->levels);
  sqlite3_free(g);
}

static int vh_graph_alloc_links(VhGraph *g, int max_level) {
  sqlite3_uint64 count;
  if (max_level < 0) {
    max_level = 0;
  }
  g->alloc_levels = max_level + 1;
  if (vh_mul_overflow_i64((sqlite3_uint64)g->n,
                          (sqlite3_uint64)g->alloc_levels, &count)) {
    return SQLITE_NOMEM;
  }
  g->links = (VhNeighborList *)vh_malloc(count * sizeof(VhNeighborList));
  if (!g->links) {
    return SQLITE_NOMEM;
  }
  memset(g->links, 0, (size_t)count * sizeof(VhNeighborList));
  return SQLITE_OK;
}

static int vh_neighbor_ensure(VhNeighborList *list, int cap) {
  int *ids;
  float *dists;
  if (list->cap >= cap) {
    return SQLITE_OK;
  }
  ids = (int *)vh_realloc(list->ids, (sqlite3_uint64)cap * sizeof(int));
  dists =
      (float *)vh_realloc(list->dists, (sqlite3_uint64)cap * sizeof(float));
  if (!ids || !dists) {
    if (ids) {
      list->ids = ids;
    }
    if (dists) {
      list->dists = dists;
    }
    return SQLITE_NOMEM;
  }
  list->ids = ids;
  list->dists = dists;
  list->cap = cap;
  return SQLITE_OK;
}

static int vh_rank_less(float a_dist, int a_id, float b_dist, int b_id) {
  if (a_dist < b_dist) {
    return 1;
  }
  if (a_dist > b_dist) {
    return 0;
  }
  return a_id < b_id;
}

static int vh_neighbor_contains(const VhNeighborList *list, int id) {
  int i;
  for (i = 0; i < list->count; i++) {
    if (list->ids[i] == id) {
      return 1;
    }
  }
  return 0;
}

static int vh_neighbor_add_unbounded(VhNeighborList *list, int id,
                                     float dist) {
  int i;
  int cap;
  if (vh_neighbor_contains(list, id)) {
    return SQLITE_OK;
  }
  cap = list->cap ? list->cap : 8;
  while (cap < list->count + 1) {
    cap *= 2;
  }
  if (vh_neighbor_ensure(list, cap) != SQLITE_OK) {
    return SQLITE_NOMEM;
  }
  i = list->count;
  while (i > 0 &&
         vh_rank_less(dist, id, list->dists[i - 1], list->ids[i - 1])) {
    list->ids[i] = list->ids[i - 1];
    list->dists[i] = list->dists[i - 1];
    i--;
  }
  list->ids[i] = id;
  list->dists[i] = dist;
  list->count++;
  return SQLITE_OK;
}

static int vh_heap_less(VhHeap *h, VhCandidate a, VhCandidate b) {
  if (a.dist == b.dist) {
    return h->is_min ? a.id < b.id : a.id > b.id;
  }
  return h->is_min ? a.dist < b.dist : a.dist > b.dist;
}

static int vh_heap_push(VhHeap *h, VhCandidate item) {
  int i;
  if (h->count >= h->cap) {
    int new_cap = h->cap ? h->cap * 2 : 64;
    VhCandidate *items =
        (VhCandidate *)vh_realloc(h->items,
                                  (sqlite3_uint64)new_cap * sizeof(VhCandidate));
    if (!items) {
      return SQLITE_NOMEM;
    }
    h->items = items;
    h->cap = new_cap;
  }
  i = h->count++;
  while (i > 0) {
    int p = (i - 1) / 2;
    if (!vh_heap_less(h, item, h->items[p])) {
      break;
    }
    h->items[i] = h->items[p];
    i = p;
  }
  h->items[i] = item;
  return SQLITE_OK;
}

static VhCandidate vh_heap_pop(VhHeap *h) {
  VhCandidate out = h->items[0];
  VhCandidate x = h->items[--h->count];
  int i = 0;
  while (1) {
    int l = i * 2 + 1;
    int r = l + 1;
    int c = l;
    if (l >= h->count) {
      break;
    }
    if (r < h->count && vh_heap_less(h, h->items[r], h->items[l])) {
      c = r;
    }
    if (!vh_heap_less(h, h->items[c], x)) {
      break;
    }
    h->items[i] = h->items[c];
    i = c;
  }
  if (h->count > 0) {
    h->items[i] = x;
  }
  return out;
}

static void vh_heap_replace_root(VhHeap *h, VhCandidate item) {
  int i = 0;
  while (1) {
    int l = i * 2 + 1;
    int r = l + 1;
    int c = l;
    if (l >= h->count) {
      break;
    }
    if (r < h->count && vh_heap_less(h, h->items[r], h->items[l])) {
      c = r;
    }
    if (!vh_heap_less(h, h->items[c], item)) {
      break;
    }
    h->items[i] = h->items[c];
    i = c;
  }
  h->items[i] = item;
}

static int vh_candidate_list_push(VhCandidateList *list, VhCandidate item) {
  if (list->count >= list->cap) {
    int new_cap = list->cap ? list->cap * 2 : 64;
    VhCandidate *items =
        (VhCandidate *)vh_realloc(list->items,
                                  (sqlite3_uint64)new_cap * sizeof(VhCandidate));
    if (!items) {
      return SQLITE_NOMEM;
    }
    list->items = items;
    list->cap = new_cap;
  }
  list->items[list->count++] = item;
  return SQLITE_OK;
}

static void vh_candidate_list_reset(VhCandidateList *list) { list->count = 0; }

static int vh_edge_list_push(VhEdgeList *list, int node, int layer,
                             int neighbor, float dist) {
  VhPlannedEdge item;
  if (list->count >= list->cap) {
    int new_cap = list->cap ? list->cap * 2 : 256;
    VhPlannedEdge *items =
        (VhPlannedEdge *)vh_realloc(
            list->items, (sqlite3_uint64)new_cap * sizeof(VhPlannedEdge));
    if (!items) {
      return SQLITE_NOMEM;
    }
    list->items = items;
    list->cap = new_cap;
  }
  item.node = node;
  item.layer = layer;
  item.neighbor = neighbor;
  item.dist = dist;
  list->items[list->count++] = item;
  return SQLITE_OK;
}

static void vh_edge_list_free(VhEdgeList *list) {
  sqlite3_free(list->items);
  memset(list, 0, sizeof(*list));
}

static int vh_touched_list_push(VhTouchedList *list, int node, int layer) {
  VhTouched item;
  if (list->count >= list->cap) {
    int new_cap = list->cap ? list->cap * 2 : 256;
    VhTouched *items =
        (VhTouched *)vh_realloc(list->items,
                                (sqlite3_uint64)new_cap * sizeof(VhTouched));
    if (!items) {
      return SQLITE_NOMEM;
    }
    list->items = items;
    list->cap = new_cap;
  }
  item.node = node;
  item.layer = layer;
  list->items[list->count++] = item;
  return SQLITE_OK;
}

static void vh_touched_list_free(VhTouchedList *list) {
  sqlite3_free(list->items);
  memset(list, 0, sizeof(*list));
}

static int vh_selected_ensure(int **ids_out, float **dists_out, int *cap_out,
                              int cap) {
  int *selected;
  float *selected_dists;
  if (*cap_out >= cap) {
    return SQLITE_OK;
  }
  selected = (int *)vh_realloc(*ids_out, (sqlite3_uint64)cap * sizeof(int));
  selected_dists =
      (float *)vh_realloc(*dists_out, (sqlite3_uint64)cap * sizeof(float));
  if (!selected || !selected_dists) {
    if (selected) {
      *ids_out = selected;
    }
    if (selected_dists) {
      *dists_out = selected_dists;
    }
    return SQLITE_NOMEM;
  }
  *ids_out = selected;
  *dists_out = selected_dists;
  *cap_out = cap;
  return SQLITE_OK;
}

static int vh_scratch_selected_ensure(VhScratch *s, int cap) {
  return vh_selected_ensure(&s->work_selected, &s->work_selected_dists,
                            &s->work_selected_cap, cap);
}

static int vh_scratch_prune_selected_ensure(VhScratch *s, int cap) {
  return vh_selected_ensure(&s->work_prune_selected,
                            &s->work_prune_selected_dists,
                            &s->work_prune_selected_cap, cap);
}

static int vh_candidate_cmp_asc(const void *a, const void *b) {
  const VhCandidate *ca = (const VhCandidate *)a;
  const VhCandidate *cb = (const VhCandidate *)b;
  if (ca->dist < cb->dist) {
    return -1;
  }
  if (ca->dist > cb->dist) {
    return 1;
  }
  return ca->id - cb->id;
}

static void vh_candidate_list_sort_asc(VhCandidateList *list) {
  if (list->count <= 1) {
    return;
  }
  qsort(list->items, (size_t)list->count, sizeof(VhCandidate),
        vh_candidate_cmp_asc);
}

static int vh_selected_contains(const int *ids, int count, int id) {
  int i;
  for (i = 0; i < count; i++) {
    if (ids[i] == id) {
      return 1;
    }
  }
  return 0;
}

static void vh_selected_sort(int *ids, float *dists, int count) {
  int i;
  for (i = 1; i < count; i++) {
    int id = ids[i];
    float dist = dists[i];
    int j = i - 1;
    while (j >= 0 && vh_rank_less(dist, id, dists[j], ids[j])) {
      ids[j + 1] = ids[j];
      dists[j + 1] = dists[j];
      j--;
    }
    ids[j + 1] = id;
    dists[j + 1] = dist;
  }
}

static void vh_candidate_list_free(VhCandidateList *list) {
  sqlite3_free(list->items);
  memset(list, 0, sizeof(*list));
}

static void vh_visit_begin(VhScratch *s, int n) {
  s->visit_mark++;
  if (s->visit_mark == 0x7fffffff) {
    memset(s->visited, 0, (size_t)n * sizeof(int));
    s->visit_mark = 1;
  }
}

static const float *vh_vec_at(const VhGraph *g, int id) {
  return &g->vectors[(sqlite3_int64)id * g->cfg.dim];
}

static int vh_search_layer_core(VhGraph *g, VhScratch *s, const float *query,
                                int entry, int ef, int layer,
                                VhCandidateList *out) {
  VhHeap *candidates = &s->work_candidates;
  VhHeap *best = &s->work_best;
  int rc = SQLITE_OK;
  if (entry < 0 || entry >= g->n || ef <= 0 || !s->visited) {
    return SQLITE_OK;
  }
  candidates->count = 0;
  candidates->is_min = 1;
  best->count = 0;
  best->is_min = 0;

  vh_visit_begin(s, g->n);
  s->visited[entry] = s->visit_mark;
  VhCandidate first = {entry, vh_metric_rank(g, query, vh_vec_at(g, entry))};
  rc = vh_heap_push(candidates, first);
  if (rc == SQLITE_OK) {
    rc = vh_heap_push(best, first);
  }

  while (rc == SQLITE_OK && candidates->count > 0) {
    VhCandidate cur = vh_heap_pop(candidates);
    float worst = best->count > 0 ? best->items[0].dist : INFINITY;
    const VhNeighborList *links;
    int i;
    if (best->count >= ef && cur.dist > worst) {
      break;
    }
    links = vh_links_at_const(g, cur.id, layer);
    if (!links) {
      continue;
    }
    for (i = 0; i < links->count; i++) {
      int nb = links->ids[i];
      VhCandidate item;
      if (nb < 0 || nb >= g->n || s->visited[nb] == s->visit_mark) {
        continue;
      }
      s->visited[nb] = s->visit_mark;
      item.id = nb;
      item.dist = vh_metric_rank(g, query, vh_vec_at(g, nb));
      worst = best->count > 0 ? best->items[0].dist : INFINITY;
      if (best->count < ef || item.dist < worst) {
        rc = vh_heap_push(candidates, item);
        if (rc != SQLITE_OK) {
          break;
        }
        if (best->count < ef) {
          rc = vh_heap_push(best, item);
        } else {
          vh_heap_replace_root(best, item);
        }
      }
    }
  }

  if (rc == SQLITE_OK) {
    int i;
    for (i = 0; i < best->count; i++) {
      rc = vh_candidate_list_push(out, best->items[i]);
      if (rc != SQLITE_OK) {
        break;
      }
    }
  }
  return rc;
}

static int vh_build_search_layer(VhGraph *g, VhScratch *s, const float *query,
                                 int entry, int ef, int layer,
                                 VhCandidateList *out) {
  return vh_search_layer_core(g, s, query, entry, ef, layer, out);
}

static int vh_query_search_layer(VhGraph *g, VhScratch *s, const float *query,
                                 int entry, int ef, VhCandidateList *out) {
  return vh_search_layer_core(g, s, query, entry, ef, 0, out);
}

static int vh_greedy_search_core(VhGraph *g, const float *query, int entry,
                                 int layer) {
  int cur = entry;
  float cur_dist;
  int changed = 1;
  if (cur < 0) {
    return cur;
  }
  cur_dist = vh_metric_rank(g, query, vh_vec_at(g, cur));
  while (changed) {
    const VhNeighborList *links = vh_links_at_const(g, cur, layer);
    int i;
    changed = 0;
    if (!links) {
      break;
    }
    for (i = 0; i < links->count; i++) {
      int nb = links->ids[i];
      float d;
      if (nb < 0 || nb >= g->n) {
        continue;
      }
      d = vh_metric_rank(g, query, vh_vec_at(g, nb));
      if (d < cur_dist) {
        cur = nb;
        cur_dist = d;
        changed = 1;
      }
    }
  }
  return cur;
}

static int vh_build_greedy_search(VhGraph *g, const float *query, int entry,
                                  int layer) {
  return vh_greedy_search_core(g, query, entry, layer);
}

static int vh_query_greedy_search(VhGraph *g, const float *query, int entry,
                                  int layer) {
  return vh_greedy_search_core(g, query, entry, layer);
}

static int vh_select_neighbors_heuristic(VhGraph *g, VhScratch *s,
                                         const float *query,
                                         VhCandidateList *candidates, int max,
                                         int *out, float *out_dists,
                                         int *out_count, int already_sorted) {
  VhCandidateList *pruned = &s->work_pruned;
  int i, j;
  int selected = 0;
  int rc = SQLITE_OK;
  if (candidates->count == 0 || max <= 0) {
    *out_count = 0;
    return SQLITE_OK;
  }
  vh_candidate_list_reset(pruned);
  if (!already_sorted) {
    vh_candidate_list_sort_asc(candidates);
  }

  for (i = 0; i < candidates->count && selected < max; i++) {
    VhCandidate cand = candidates->items[i];
    const float *cand_vec;
    int good = 1;
    if (cand.id < 0 || cand.id >= g->n || g->deleted[cand.id] ||
        vh_selected_contains(out, selected, cand.id)) {
      continue;
    }
    cand_vec = vh_vec_at(g, cand.id);
    for (j = 0; j < selected; j++) {
      float pair_dist;
      pair_dist = vh_metric_rank(g, cand_vec, vh_vec_at(g, out[j]));
      if (pair_dist < cand.dist) {
        good = 0;
        break;
      }
    }
    if (!good) {
      rc = vh_candidate_list_push(pruned, cand);
      if (rc != SQLITE_OK) {
        return rc;
      }
      continue;
    }
    out[selected] = cand.id;
    if (out_dists) {
      out_dists[selected] = cand.dist;
    }
    selected++;
  }
  for (i = 0; i < pruned->count && selected < max; i++) {
    VhCandidate cand = pruned->items[i];
    if (vh_selected_contains(out, selected, cand.id)) {
      continue;
    }
    out[selected] = cand.id;
    if (out_dists) {
      out_dists[selected] = cand.dist;
    }
    selected++;
  }
  if (out_dists && selected > 1) {
    vh_selected_sort(out, out_dists, selected);
  }
  *out_count = selected;
  (void)query;
  return SQLITE_OK;
}

static int vh_prune_neighbors(VhGraph *g, VhScratch *s, int node, int layer) {
  VhNeighborList *list = vh_links_at(g, node, layer);
  int max = vh_max_neighbors(&g->cfg, layer);
  VhCandidateList *candidates = &s->work_prune_candidates;
  int selected_count = 0;
  int i;
  int rc;
  if (!list || list->count <= max) {
    return SQLITE_OK;
  }
  rc = vh_scratch_prune_selected_ensure(s, max);
  if (rc != SQLITE_OK) {
    return rc;
  }
  vh_candidate_list_reset(candidates);
  for (i = 0; i < list->count; i++) {
    VhCandidate cand;
    int nb = list->ids[i];
    if (nb < 0 || nb >= g->n || nb == node || g->deleted[nb]) {
      continue;
    }
    cand.id = nb;
    cand.dist = list->dists[i];
    rc = vh_candidate_list_push(candidates, cand);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  rc = vh_select_neighbors_heuristic(g, s, vh_vec_at(g, node), candidates, max,
                                     s->work_prune_selected,
                                     s->work_prune_selected_dists,
                                     &selected_count, 1);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_neighbor_ensure(list, selected_count);
  if (rc != SQLITE_OK) {
    return rc;
  }
  for (i = 0; i < selected_count; i++) {
    list->ids[i] = s->work_prune_selected[i];
    list->dists[i] = s->work_prune_selected_dists[i];
  }
  list->count = selected_count;
  return SQLITE_OK;
}

static int vh_plan_node_edges(VhGraph *g, VhScratch *s, VhEdgeList *edges,
                              int node, int entry_node, int max_level) {
  const float *query = vh_vec_at(g, node);
  int level = g->levels[node];
  int ep = entry_node;
  int layer;
  if (entry_node < 0 || max_level < 0) {
    return SQLITE_OK;
  }
  for (layer = max_level; layer > level; layer--) {
    ep = vh_build_greedy_search(g, query, ep, layer);
  }
  for (layer = level < max_level ? level : max_level; layer >= 0;
       layer--) {
    VhCandidateList *candidates = &s->work_results;
    int max = vh_max_neighbors(&g->cfg, layer);
    int *selected;
    int selected_count = 0;
    int i;
    int rc;
    vh_candidate_list_reset(candidates);
    rc = vh_scratch_selected_ensure(s, max);
    if (rc != SQLITE_OK) {
      return rc;
    }
    selected = s->work_selected;
    rc = vh_build_search_layer(g, s, query, ep, g->cfg.ef_construction, layer,
                               candidates);
    if (rc == SQLITE_OK) {
      rc = vh_select_neighbors_heuristic(g, s, query, candidates, max, selected,
                                         s->work_selected_dists,
                                         &selected_count, 0);
    }
    if (rc == SQLITE_OK && selected_count > 0) {
      ep = selected[0];
    }
    for (i = 0; rc == SQLITE_OK && i < selected_count; i++) {
      rc = vh_edge_list_push(edges, node, layer, selected[i],
                             s->work_selected_dists[i]);
    }
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  return SQLITE_OK;
}

static int vh_mark_touched(VhGraph *g, int *marks, int mark,
                           VhTouchedList *touched, int node, int layer) {
  int idx;
  if (node < 0 || node >= g->n || layer < 0 || layer >= g->alloc_levels) {
    return SQLITE_OK;
  }
  idx = layer * g->n + node;
  if (marks[idx] == mark) {
    return SQLITE_OK;
  }
  marks[idx] = mark;
  return vh_touched_list_push(touched, node, layer);
}

static int vh_merge_planned_edge(VhGraph *g, const VhPlannedEdge *edge,
                                 int *marks, int mark,
                                 VhTouchedList *touched) {
  VhNeighborList *a;
  VhNeighborList *b;
  int rc;
  if (edge->node < 0 || edge->node >= g->n || edge->neighbor < 0 ||
      edge->neighbor >= g->n || edge->layer < 0 ||
      edge->layer >= g->alloc_levels || edge->node == edge->neighbor ||
      g->deleted[edge->node] || g->deleted[edge->neighbor]) {
    return SQLITE_OK;
  }
  a = vh_links_at(g, edge->node, edge->layer);
  b = vh_links_at(g, edge->neighbor, edge->layer);
  if (!a || !b) {
    return SQLITE_OK;
  }
  rc = vh_neighbor_add_unbounded(a, edge->neighbor, edge->dist);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_neighbor_add_unbounded(b, edge->node, edge->dist);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_mark_touched(g, marks, mark, touched, edge->node, edge->layer);
  if (rc == SQLITE_OK) {
    rc = vh_mark_touched(g, marks, mark, touched, edge->neighbor, edge->layer);
  }
  return rc;
}

static void *vh_build_worker_main(void *arg) {
  VhBuildWorker *w = (VhBuildWorker *)arg;
  VhBuildPool *pool = w->pool;
  int seen_generation = 0;
  while (1) {
    VhBuildTaskKind task_kind;
    int i;
    pthread_mutex_lock(&pool->mutex);
    while (!pool->stop && seen_generation == pool->generation) {
      pthread_cond_wait(&pool->start_cond, &pool->mutex);
    }
    if (pool->stop) {
      pthread_mutex_unlock(&pool->mutex);
      break;
    }
    task_kind = pool->task_kind;
    seen_generation = pool->generation;
    pthread_mutex_unlock(&pool->mutex);

    w->rc = SQLITE_OK;
    if (task_kind == VH_BUILD_TASK_PLAN) {
      w->edges.count = 0;
      for (i = w->begin; i < w->end; i++) {
        int node = w->active_ids[i];
        w->rc = vh_plan_node_edges(pool->g, &w->scratch, &w->edges, node,
                                   w->entry_node, w->max_level);
        if (w->rc != SQLITE_OK) {
          break;
        }
      }
    } else if (task_kind == VH_BUILD_TASK_PRUNE) {
      for (i = w->begin; i < w->end; i++) {
        VhTouched item = w->touched[i];
        w->rc = vh_prune_neighbors(pool->g, &w->scratch, item.node, item.layer);
        if (w->rc != SQLITE_OK) {
          break;
        }
      }
    }

    pthread_mutex_lock(&pool->mutex);
    pool->done_count++;
    if (pool->done_count >= pool->worker_count) {
      pthread_cond_signal(&pool->done_cond);
    }
    pthread_mutex_unlock(&pool->mutex);
  }
  return 0;
}

static void vh_build_pool_destroy(VhBuildPool *pool) {
  int i;
  if (!pool) {
    return;
  }
  if (pool->workers && pool->mutex_ready) {
    pthread_mutex_lock(&pool->mutex);
    pool->stop = 1;
    pool->generation++;
    if (pool->start_cond_ready) {
      pthread_cond_broadcast(&pool->start_cond);
    }
    pthread_mutex_unlock(&pool->mutex);
  }
  if (pool->workers) {
    for (i = 0; i < pool->worker_count; i++) {
      if (pool->workers[i].started) {
        pthread_join(pool->workers[i].thread, 0);
      }
      vh_scratch_free(&pool->workers[i].scratch);
      vh_edge_list_free(&pool->workers[i].edges);
    }
  }
  sqlite3_free(pool->workers);
  if (pool->done_cond_ready) {
    pthread_cond_destroy(&pool->done_cond);
  }
  if (pool->start_cond_ready) {
    pthread_cond_destroy(&pool->start_cond);
  }
  if (pool->mutex_ready) {
    pthread_mutex_destroy(&pool->mutex);
  }
  memset(pool, 0, sizeof(*pool));
}

static int vh_build_pool_init(VhBuildPool *pool, VhGraph *g) {
  int i;
  int rc = SQLITE_OK;
  memset(pool, 0, sizeof(*pool));
  pool->g = g;
  pool->worker_count = g->cfg.build_threads;
  pool->workers =
      (VhBuildWorker *)vh_malloc((sqlite3_uint64)pool->worker_count *
                                 sizeof(VhBuildWorker));
  if (!pool->workers) {
    return SQLITE_NOMEM;
  }
  memset(pool->workers, 0,
         (size_t)pool->worker_count * sizeof(VhBuildWorker));
  if (pthread_mutex_init(&pool->mutex, 0) != 0) {
    rc = SQLITE_NOMEM;
    goto fail;
  }
  pool->mutex_ready = 1;
  if (pthread_cond_init(&pool->start_cond, 0) != 0) {
    rc = SQLITE_NOMEM;
    goto fail;
  }
  pool->start_cond_ready = 1;
  if (pthread_cond_init(&pool->done_cond, 0) != 0) {
    rc = SQLITE_NOMEM;
    goto fail;
  }
  pool->done_cond_ready = 1;
  for (i = 0; i < pool->worker_count; i++) {
    VhBuildWorker *w = &pool->workers[i];
    w->pool = pool;
    rc = vh_scratch_init(&w->scratch, g->n);
    if (rc != SQLITE_OK) {
      goto fail;
    }
    if (pthread_create(&w->thread, 0, vh_build_worker_main, w) != 0) {
      rc = SQLITE_NOMEM;
      goto fail;
    }
    w->started = 1;
  }
  return SQLITE_OK;

fail:
  vh_build_pool_destroy(pool);
  return rc;
}

static int vh_build_pool_run_plan(VhBuildPool *pool, const int *active_ids,
                                  int begin, int count, int entry_node,
                                  int max_level) {
  int i;
  int rc = SQLITE_OK;
  pthread_mutex_lock(&pool->mutex);
  for (i = 0; i < pool->worker_count; i++) {
    int worker_begin =
        begin + (int)(((sqlite3_int64)count * i) / pool->worker_count);
    int worker_end =
        begin + (int)(((sqlite3_int64)count * (i + 1)) / pool->worker_count);
    VhBuildWorker *w = &pool->workers[i];
    w->active_ids = active_ids;
    w->touched = 0;
    w->begin = worker_begin;
    w->end = worker_end;
    w->entry_node = entry_node;
    w->max_level = max_level;
    w->rc = SQLITE_OK;
    w->edges.count = 0;
  }
  pool->task_kind = VH_BUILD_TASK_PLAN;
  pool->done_count = 0;
  pool->generation++;
  pthread_cond_broadcast(&pool->start_cond);
  while (pool->done_count < pool->worker_count) {
    pthread_cond_wait(&pool->done_cond, &pool->mutex);
  }
  for (i = 0; i < pool->worker_count; i++) {
    if (pool->workers[i].rc != SQLITE_OK) {
      rc = pool->workers[i].rc;
      break;
    }
  }
  pthread_mutex_unlock(&pool->mutex);
  return rc;
}

static int vh_build_pool_run_prune(VhBuildPool *pool,
                                   const VhTouchedList *touched) {
  int i;
  int rc = SQLITE_OK;
  if (touched->count == 0) {
    return SQLITE_OK;
  }
  pthread_mutex_lock(&pool->mutex);
  for (i = 0; i < pool->worker_count; i++) {
    int worker_begin =
        (int)(((sqlite3_int64)touched->count * i) / pool->worker_count);
    int worker_end =
        (int)(((sqlite3_int64)touched->count * (i + 1)) / pool->worker_count);
    VhBuildWorker *w = &pool->workers[i];
    w->active_ids = 0;
    w->touched = touched->items;
    w->begin = worker_begin;
    w->end = worker_end;
    w->entry_node = -1;
    w->max_level = -1;
    w->rc = SQLITE_OK;
  }
  pool->task_kind = VH_BUILD_TASK_PRUNE;
  pool->done_count = 0;
  pool->generation++;
  pthread_cond_broadcast(&pool->start_cond);
  while (pool->done_count < pool->worker_count) {
    pthread_cond_wait(&pool->done_cond, &pool->mutex);
  }
  for (i = 0; i < pool->worker_count; i++) {
    if (pool->workers[i].rc != SQLITE_OK) {
      rc = pool->workers[i].rc;
      break;
    }
  }
  pthread_mutex_unlock(&pool->mutex);
  return rc;
}

static int vh_collect_active_ids(VhGraph *g, int **out) {
  int *ids;
  int id;
  int count = 0;
  ids = (int *)vh_malloc((sqlite3_uint64)g->active_count * sizeof(int));
  if (!ids && g->active_count > 0) {
    return SQLITE_NOMEM;
  }
  for (id = 0; id < g->n; id++) {
    if (!g->deleted[id]) {
      ids[count++] = id;
    }
  }
  *out = ids;
  return SQLITE_OK;
}

static int vh_build_graph(VhGraph *g) {
  int *active_ids = 0;
  int *touch_marks = 0;
  VhTouchedList touched = {0, 0, 0};
  VhBuildPool pool;
  int pool_ready = 0;
  int pos;
  int mark = 1;
  int rc;
  sqlite3_uint64 mark_count;
  memset(&pool, 0, sizeof(pool));

  if (g->cfg.build_threads < 2 || g->cfg.build_batch_size < 2) {
    return SQLITE_ERROR;
  }
  g->entry_node = -1;
  g->max_level = -1;
  if (g->active_count <= 0) {
    return SQLITE_OK;
  }
  rc = vh_collect_active_ids(g, &active_ids);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (vh_mul_overflow_i64((sqlite3_uint64)g->n,
                          (sqlite3_uint64)g->alloc_levels, &mark_count)) {
    sqlite3_free(active_ids);
    return SQLITE_NOMEM;
  }
  touch_marks = (int *)vh_malloc(mark_count * sizeof(int));
  if (!touch_marks) {
    sqlite3_free(active_ids);
    return SQLITE_NOMEM;
  }
  memset(touch_marks, 0, (size_t)mark_count * sizeof(int));

  g->entry_node = active_ids[0];
  g->max_level = g->levels[g->entry_node];
  rc = vh_build_pool_init(&pool, g);
  if (rc != SQLITE_OK) {
    goto done;
  }
  pool_ready = 1;
  pos = 1;
  while (pos < g->active_count && rc == SQLITE_OK) {
    int remaining = g->active_count - pos;
    int batch_count = g->cfg.build_batch_size;
    int dynamic_limit = pos < 2 ? 2 : pos;
    int frozen_entry = g->entry_node;
    int frozen_max_level = g->max_level;
    int i;

    if (batch_count > dynamic_limit) {
      batch_count = dynamic_limit;
    }
    if (batch_count > remaining) {
      batch_count = remaining;
    }
    rc = vh_build_pool_run_plan(&pool, active_ids, pos, batch_count,
                                frozen_entry, frozen_max_level);
    touched.count = 0;
    if (rc == SQLITE_OK) {
      mark++;
      if (mark == 0x7fffffff) {
        memset(touch_marks, 0, (size_t)mark_count * sizeof(int));
        mark = 1;
      }
      for (i = 0; rc == SQLITE_OK && i < pool.worker_count; i++) {
        int j;
        VhBuildWorker *worker = &pool.workers[i];
        for (j = 0; j < worker->edges.count; j++) {
          rc = vh_merge_planned_edge(g, &worker->edges.items[j],
                                     touch_marks, mark, &touched);
        }
      }
    }
    if (rc == SQLITE_OK) {
      rc = vh_build_pool_run_prune(&pool, &touched);
    }
    if (rc == SQLITE_OK) {
      int end = pos + batch_count;
      for (i = pos; i < end; i++) {
        int node = active_ids[i];
        if (g->levels[node] > g->max_level) {
          g->entry_node = node;
          g->max_level = g->levels[node];
        }
      }
      pos += batch_count;
    }
  }
done:
  if (pool_ready) {
    vh_build_pool_destroy(&pool);
  }
  vh_touched_list_free(&touched);
  sqlite3_free(touch_marks);
  sqlite3_free(active_ids);
  return rc;
}

static int vh_meta_set(sqlite3 *db, const char *db_name, const char *meta_name,
                       const char *key, const char *value) {
  char *qmeta = vh_qualified(db_name, meta_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qmeta) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("REPLACE INTO %s(key,value) VALUES(?1,?2)", qmeta);
  sqlite3_free(qmeta);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, value, -1, SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt) == SQLITE_DONE ? SQLITE_OK : sqlite3_errcode(db);
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_meta_set_int(sqlite3 *db, const char *db_name,
                           const char *meta_name, const char *key,
                           sqlite3_int64 value) {
  char buf[64];
  sqlite3_snprintf(sizeof(buf), buf, "%lld", (long long)value);
  return vh_meta_set(db, db_name, meta_name, key, buf);
}

static int vh_meta_get_int(sqlite3 *db, const char *db_name,
                           const char *meta_name, const char *key,
                           sqlite3_int64 def, sqlite3_int64 *out) {
  char *qmeta = vh_qualified(db_name, meta_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qmeta) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT value FROM %s WHERE key=?1", qmeta);
  sqlite3_free(qmeta);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    sqlite3_finalize(stmt);
    return rc;
  }
  sqlite3_bind_text(stmt, 1, key, -1, SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const unsigned char *text = sqlite3_column_text(stmt, 0);
    if (!text || vh_parse_i64_strict((const char *)text, out) != SQLITE_OK) {
      rc = SQLITE_CORRUPT_VTAB;
    } else {
      rc = SQLITE_OK;
    }
  } else if (rc == SQLITE_DONE) {
    *out = def;
    rc = SQLITE_OK;
  } else {
    rc = sqlite3_errcode(db);
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_get_data_version(sqlite3 *db, const char *db_name,
                               sqlite3_int64 *out) {
  char *qdb = vh_quote_ident(db_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qdb) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("PRAGMA %s.data_version", qdb);
  sqlite3_free(qdb);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    int step_rc = sqlite3_step(stmt);
    if (step_rc == SQLITE_ROW) {
      *out = sqlite3_column_int64(stmt, 0);
    } else if (step_rc == SQLITE_DONE) {
      rc = SQLITE_ERROR;
    } else {
      rc = sqlite3_errcode(db);
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_create_shadow(sqlite3 *db, const char *db_name,
                            const char *table_name, const VhConfig *cfg,
                            int reset, char **pzErr) {
  char *meta = vh_shadow_name(table_name, "meta");
  char *vectors = vh_shadow_name(table_name, "vectors");
  char *graph = vh_shadow_name(table_name, "graph");
  char *index = vh_shadow_name(table_name, "vectors_rowid_idx");
  char *qmeta = 0;
  char *qvectors = 0;
  char *qgraph = 0;
  char *qindex = 0;
  char *uvectors = 0;
  int rc = SQLITE_OK;
  if (!meta || !vectors || !graph || !index) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  qmeta = vh_qualified(db_name, meta);
  qvectors = vh_qualified(db_name, vectors);
  qgraph = vh_qualified(db_name, graph);
  qindex = vh_qualified(db_name, index);
  uvectors = vh_quote_ident(vectors);
  if (!qmeta || !qvectors || !qgraph || !qindex || !uvectors) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  if (reset) {
    rc = vh_execf(db, pzErr, "DROP TABLE IF EXISTS %s", qgraph);
    if (rc != SQLITE_OK) {
      goto done;
    }
    rc = vh_execf(db, pzErr, "DROP TABLE IF EXISTS %s", qvectors);
    if (rc != SQLITE_OK) {
      goto done;
    }
    rc = vh_execf(db, pzErr, "DROP TABLE IF EXISTS %s", qmeta);
    if (rc != SQLITE_OK) {
      goto done;
    }
  }
  rc = vh_execf(db, pzErr,
                "CREATE TABLE IF NOT EXISTS %s(key TEXT PRIMARY KEY, value "
                "TEXT NOT NULL)",
                qmeta);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_execf(db, pzErr,
                "CREATE TABLE IF NOT EXISTS %s(node_id INTEGER PRIMARY KEY, "
                "rowid_value INTEGER NOT NULL, level INTEGER NOT NULL, "
                "deleted INTEGER NOT NULL DEFAULT 0, embedding BLOB NOT NULL)",
                qvectors);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_execf(db, pzErr,
                "CREATE INDEX IF NOT EXISTS %s ON %s(rowid_value, deleted)",
                qindex, uvectors);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_execf(db, pzErr,
                "CREATE TABLE IF NOT EXISTS %s(node_id INTEGER NOT NULL, "
                "layer INTEGER NOT NULL, neighbors BLOB NOT NULL, PRIMARY "
                "KEY(node_id, layer))",
                qgraph);
  if (rc != SQLITE_OK) {
    goto done;
  }
  if (reset) {
    rc = vh_meta_set_int(db, db_name, meta, "dim", cfg->dim);
    if (rc == SQLITE_OK) {
      rc = vh_meta_set(db, db_name, meta, "metric", vh_metric_name(cfg->metric));
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "m", cfg->m);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "ef_construction",
                           cfg->ef_construction);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "ef_search", cfg->ef_search);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "build_threads",
                           cfg->build_threads);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "build_batch_size",
                           cfg->build_batch_size);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "seed", (sqlite3_int64)cfg->seed);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "generation", 0);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "entry_node", -1);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "max_level", -1);
    }
    if (rc == SQLITE_OK) {
      rc = vh_meta_set_int(db, db_name, meta, "graph_valid", 0);
    }
  }

done:
  sqlite3_free(meta);
  sqlite3_free(vectors);
  sqlite3_free(graph);
  sqlite3_free(index);
  sqlite3_free(qmeta);
  sqlite3_free(qvectors);
  sqlite3_free(qgraph);
  sqlite3_free(qindex);
  sqlite3_free(uvectors);
  return rc;
}

static int vh_mark_graph_stale(VhTable *tab) {
  char *qmeta = vh_qualified(tab->db_name, tab->meta_name);
  char *sql;
  int rc;
  if (!qmeta) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("UPDATE %s SET value='0' "
                        "WHERE key='graph_valid' AND value<>'0'",
                        qmeta);
  sqlite3_free(qmeta);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_exec(tab->db, sql, 0, 0, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    tab->cache_generation = -1;
    tab->cache_data_version = -1;
  }
  return rc;
}

static int vh_load_vectors(sqlite3 *db, const char *db_name,
                           const char *vectors_name, const VhConfig *cfg,
                           int assign_levels, VhGraph **out) {
  char *qvectors = vh_qualified(db_name, vectors_name);
  char *sql = 0;
  sqlite3_stmt *stmt = 0;
  sqlite3_int64 max_id = -1;
  int rc;
  VhGraph *g = 0;
  sqlite3_uint64 bytes;
  uint64_t rng = cfg->seed;

  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  rc = vh_validate_config((VhConfig *)cfg, 0, 0);
  if (rc != SQLITE_OK) {
    sqlite3_free(qvectors);
    return SQLITE_CORRUPT_VTAB;
  }
  sql = sqlite3_mprintf("SELECT COALESCE(MAX(node_id), -1) FROM %s", qvectors);
  if (!sql) {
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    int step_rc = sqlite3_step(stmt);
    if (step_rc == SQLITE_ROW) {
      max_id = sqlite3_column_int64(stmt, 0);
      if (max_id > INT_MAX - 1) {
        rc = SQLITE_CORRUPT_VTAB;
      }
    } else if (step_rc != SQLITE_DONE) {
      rc = sqlite3_errcode(db);
    }
  }
  sqlite3_finalize(stmt);
  stmt = 0;
  if (rc != SQLITE_OK) {
    sqlite3_free(qvectors);
    return rc;
  }

  g = (VhGraph *)vh_malloc(sizeof(VhGraph));
  if (!g) {
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  memset(g, 0, sizeof(*g));
  g->cfg = *cfg;
  g->n = (int)(max_id + 1);
  g->entry_node = -1;
  g->max_level = -1;
  if (g->n <= 0) {
    *out = g;
    sqlite3_free(qvectors);
    return SQLITE_OK;
  }

  if (vh_mul_overflow_i64((sqlite3_uint64)g->n, (sqlite3_uint64)cfg->dim,
                          &bytes)) {
    vh_graph_free(g);
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  g->vectors = (float *)vh_malloc(bytes * sizeof(float));
  g->rowids =
      (sqlite3_int64 *)vh_malloc((sqlite3_uint64)g->n * sizeof(sqlite3_int64));
  g->deleted = (unsigned char *)vh_malloc((sqlite3_uint64)g->n);
  g->levels = (int *)vh_malloc((sqlite3_uint64)g->n * sizeof(int));
  if (!g->vectors || !g->rowids || !g->deleted || !g->levels) {
    vh_graph_free(g);
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  memset(g->vectors, 0, (size_t)bytes * sizeof(float));
  memset(g->rowids, 0, (size_t)g->n * sizeof(sqlite3_int64));
  memset(g->deleted, 1, (size_t)g->n);
  memset(g->levels, 0, (size_t)g->n * sizeof(int));

  sql = sqlite3_mprintf("SELECT node_id,rowid_value,level,embedding FROM %s "
                        "WHERE deleted=0 ORDER BY node_id",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    vh_graph_free(g);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    sqlite3_int64 id64 = sqlite3_column_int64(stmt, 0);
    sqlite3_int64 rowid = sqlite3_column_int64(stmt, 1);
    sqlite3_int64 level64 = sqlite3_column_int64(stmt, 2);
    int id;
    int level;
    const void *blob = sqlite3_column_blob(stmt, 3);
    int blob_bytes = sqlite3_column_bytes(stmt, 3);
    float *dst;
    if (id64 < 0 || id64 > INT_MAX || level64 < 0 ||
        level64 > VH_MAX_LEVEL) {
      sqlite3_finalize(stmt);
      vh_graph_free(g);
      return SQLITE_CORRUPT_VTAB;
    }
    id = (int)id64;
    level = (int)level64;
    if (id < 0 || id >= g->n || blob_bytes != cfg->dim * (int)sizeof(float) ||
        !blob || vh_validate_finite_f32(blob, cfg->dim) != SQLITE_OK) {
      sqlite3_finalize(stmt);
      vh_graph_free(g);
      return SQLITE_CORRUPT_VTAB;
    }
    dst = &g->vectors[(sqlite3_int64)id * cfg->dim];
    memcpy(dst, blob, (size_t)blob_bytes);
    if (cfg->metric == VH_METRIC_COSINE) {
      vh_normalize(dst, cfg->dim);
    }
    if (assign_levels) {
      level = vh_random_level(&rng, cfg->m);
    }
    if (level > VH_MAX_LEVEL) {
      level = VH_MAX_LEVEL;
    }
    g->rowids[id] = rowid;
    g->levels[id] = level;
    g->deleted[id] = 0;
    g->active_count++;
    if (level > g->max_level) {
      g->max_level = level;
    }
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  } else {
    rc = sqlite3_errcode(db);
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  if (g->active_count == 0) {
    g->max_level = -1;
  }
  rc = vh_graph_alloc_links(g, g->max_level < 0 ? 0 : g->max_level);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  *out = g;
  return SQLITE_OK;
}

static int vh_load_vectors_compact(sqlite3 *db, const char *db_name,
                                   const char *vectors_name,
                                   const VhConfig *cfg, VhGraph **out) {
  char *qvectors = vh_qualified(db_name, vectors_name);
  char *sql = 0;
  sqlite3_stmt *stmt = 0;
  sqlite3_int64 count64 = 0;
  int rc;
  VhGraph *g = 0;
  sqlite3_uint64 bytes;
  uint64_t rng = cfg->seed;
  int id = 0;

  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  rc = vh_validate_config((VhConfig *)cfg, 0, 0);
  if (rc != SQLITE_OK) {
    sqlite3_free(qvectors);
    return SQLITE_CORRUPT_VTAB;
  }

  sql = sqlite3_mprintf("SELECT COUNT(*) FROM %s WHERE deleted=0", qvectors);
  if (!sql) {
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  sql = 0;
  if (rc == SQLITE_OK) {
    int step_rc = sqlite3_step(stmt);
    if (step_rc == SQLITE_ROW) {
      count64 = sqlite3_column_int64(stmt, 0);
      if (count64 < 0 || count64 > INT_MAX) {
        rc = SQLITE_CORRUPT_VTAB;
      }
    } else if (step_rc != SQLITE_DONE) {
      rc = sqlite3_errcode(db);
    }
  }
  sqlite3_finalize(stmt);
  stmt = 0;
  if (rc != SQLITE_OK) {
    sqlite3_free(qvectors);
    return rc;
  }

  g = (VhGraph *)vh_malloc(sizeof(VhGraph));
  if (!g) {
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  memset(g, 0, sizeof(*g));
  g->cfg = *cfg;
  g->n = (int)count64;
  g->active_count = g->n;
  g->rewrite_vectors = 1;
  g->entry_node = -1;
  g->max_level = -1;
  if (g->n <= 0) {
    *out = g;
    sqlite3_free(qvectors);
    return SQLITE_OK;
  }

  if (vh_mul_overflow_i64((sqlite3_uint64)g->n, (sqlite3_uint64)cfg->dim,
                          &bytes)) {
    vh_graph_free(g);
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  g->vectors = (float *)vh_malloc(bytes * sizeof(float));
  g->stored_vectors = (float *)vh_malloc(bytes * sizeof(float));
  g->rowids =
      (sqlite3_int64 *)vh_malloc((sqlite3_uint64)g->n * sizeof(sqlite3_int64));
  g->deleted = (unsigned char *)vh_malloc((sqlite3_uint64)g->n);
  g->levels = (int *)vh_malloc((sqlite3_uint64)g->n * sizeof(int));
  if (!g->vectors || !g->stored_vectors || !g->rowids || !g->deleted ||
      !g->levels) {
    vh_graph_free(g);
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }
  memset(g->deleted, 0, (size_t)g->n);

  sql = sqlite3_mprintf("SELECT rowid_value,embedding FROM %s "
                        "WHERE deleted=0 ORDER BY node_id",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    vh_graph_free(g);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    sqlite3_int64 rowid = sqlite3_column_int64(stmt, 0);
    const void *blob = sqlite3_column_blob(stmt, 1);
    int blob_bytes = sqlite3_column_bytes(stmt, 1);
    float *raw;
    float *vec;
    int level;
    if (id >= g->n || blob_bytes != cfg->dim * (int)sizeof(float) || !blob ||
        vh_validate_finite_f32(blob, cfg->dim) != SQLITE_OK) {
      sqlite3_finalize(stmt);
      vh_graph_free(g);
      return SQLITE_CORRUPT_VTAB;
    }
    raw = &g->stored_vectors[(sqlite3_int64)id * cfg->dim];
    vec = &g->vectors[(sqlite3_int64)id * cfg->dim];
    memcpy(raw, blob, (size_t)blob_bytes);
    memcpy(vec, blob, (size_t)blob_bytes);
    if (cfg->metric == VH_METRIC_COSINE) {
      vh_normalize(vec, cfg->dim);
    }
    level = vh_random_level(&rng, cfg->m);
    if (level > VH_MAX_LEVEL) {
      level = VH_MAX_LEVEL;
    }
    g->rowids[id] = rowid;
    g->levels[id] = level;
    if (level > g->max_level) {
      g->max_level = level;
    }
    id++;
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  } else {
    rc = sqlite3_errcode(db);
  }
  sqlite3_finalize(stmt);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  if (id != g->n) {
    vh_graph_free(g);
    return SQLITE_CORRUPT_VTAB;
  }
  rc = vh_graph_alloc_links(g, g->max_level < 0 ? 0 : g->max_level);
  if (rc != SQLITE_OK) {
    vh_graph_free(g);
    return rc;
  }
  *out = g;
  return SQLITE_OK;
}

static int vh_load_graph_edges(sqlite3 *db, const char *db_name,
                               const char *graph_name, VhGraph *g) {
  char *qgraph = vh_qualified(db_name, graph_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qgraph) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT node_id,layer,neighbors FROM %s", qgraph);
  sqlite3_free(qgraph);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
    sqlite3_int64 node64 = sqlite3_column_int64(stmt, 0);
    sqlite3_int64 layer64 = sqlite3_column_int64(stmt, 1);
    const void *blob = sqlite3_column_blob(stmt, 2);
    int bytes = sqlite3_column_bytes(stmt, 2);
    VhNeighborList *list;
    const int32_t *ids;
    int node;
    int layer;
    int count;
    int i, j;
    if (node64 < 0 || node64 > INT_MAX || layer64 < 0 ||
        layer64 > INT_MAX) {
      sqlite3_finalize(stmt);
      return SQLITE_CORRUPT_VTAB;
    }
    node = (int)node64;
    layer = (int)layer64;
    if (node < 0 || node >= g->n || g->deleted[node] || layer < 0 ||
        layer >= g->alloc_levels || layer > g->levels[node] || bytes < 0 ||
        bytes % (int)sizeof(int32_t) != 0) {
      sqlite3_finalize(stmt);
      return SQLITE_CORRUPT_VTAB;
    }
    count = bytes / (int)sizeof(int32_t);
    if (count > vh_max_neighbors(&g->cfg, layer) || (count > 0 && !blob)) {
      sqlite3_finalize(stmt);
      return SQLITE_CORRUPT_VTAB;
    }
    ids = (const int32_t *)blob;
    for (i = 0; i < count; i++) {
      int nb = ids[i];
      if (nb < 0 || nb >= g->n || g->deleted[nb] || nb == node ||
          layer > g->levels[nb]) {
        sqlite3_finalize(stmt);
        return SQLITE_CORRUPT_VTAB;
      }
      for (j = i + 1; j < count; j++) {
        if (ids[j] == nb) {
          sqlite3_finalize(stmt);
          return SQLITE_CORRUPT_VTAB;
        }
      }
    }
    list = vh_links_at(g, node, layer);
    if (vh_neighbor_ensure(list, count) != SQLITE_OK) {
      sqlite3_finalize(stmt);
      return SQLITE_NOMEM;
    }
    if (count > 0) {
      memcpy(list->ids, blob, (size_t)bytes);
    }
    list->count = count;
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  } else {
    rc = sqlite3_errcode(db);
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_persist_graph(sqlite3 *db, const char *db_name,
                            const char *meta_name, const char *vectors_name,
                            const char *graph_name, VhGraph *g) {
  char *qgraph = vh_qualified(db_name, graph_name);
  char *qvectors = vh_qualified(db_name, vectors_name);
  char *sql = 0;
  sqlite3_stmt *edge_stmt = 0;
  sqlite3_stmt *level_stmt = 0;
  sqlite3_stmt *vector_stmt = 0;
  sqlite3_int64 gen = 0;
  int compact = g->rewrite_vectors;
  int rc;
  int id, layer;
  if (!qgraph || !qvectors) {
    sqlite3_free(qgraph);
    sqlite3_free(qvectors);
    return SQLITE_NOMEM;
  }

  rc = sqlite3_exec(db, "SAVEPOINT vechnsw_rebuild", 0, 0, 0);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_execf(db, 0, "DELETE FROM %s", qgraph);
  if (rc != SQLITE_OK) {
    goto rollback;
  }
  if (compact) {
    rc = vh_execf(db, 0, "DELETE FROM %s", qvectors);
    if (rc != SQLITE_OK) {
      goto rollback;
    }
    sql = sqlite3_mprintf(
        "INSERT INTO %s(node_id,rowid_value,level,deleted,embedding) "
        "VALUES(?1,?2,?3,0,?4)",
        qvectors);
    if (!sql) {
      rc = SQLITE_NOMEM;
      goto rollback;
    }
    rc = sqlite3_prepare_v2(db, sql, -1, &vector_stmt, 0);
    sqlite3_free(sql);
    sql = 0;
    if (rc != SQLITE_OK) {
      goto rollback;
    }
  } else {
    sql = sqlite3_mprintf("UPDATE %s SET level=?1 WHERE node_id=?2", qvectors);
    if (!sql) {
      rc = SQLITE_NOMEM;
      goto rollback;
    }
    rc = sqlite3_prepare_v2(db, sql, -1, &level_stmt, 0);
    sqlite3_free(sql);
    sql = 0;
    if (rc != SQLITE_OK) {
      goto rollback;
    }
  }
  sql = sqlite3_mprintf("INSERT INTO %s(node_id,layer,neighbors) "
                        "VALUES(?1,?2,?3)",
                        qgraph);
  if (!sql) {
    rc = SQLITE_NOMEM;
    goto rollback;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &edge_stmt, 0);
  sqlite3_free(sql);
  sql = 0;
  if (rc != SQLITE_OK) {
    goto rollback;
  }
  for (id = 0; id < g->n; id++) {
    if (g->deleted[id]) {
      continue;
    }
    if (compact) {
      sqlite3_bind_int(vector_stmt, 1, id);
      sqlite3_bind_int64(vector_stmt, 2, g->rowids[id]);
      sqlite3_bind_int(vector_stmt, 3, g->levels[id]);
      sqlite3_bind_blob(vector_stmt, 4,
                        &g->stored_vectors[(sqlite3_int64)id * g->cfg.dim],
                        g->cfg.dim * (int)sizeof(float), SQLITE_TRANSIENT);
      rc = sqlite3_step(vector_stmt) == SQLITE_DONE ? SQLITE_OK
                                                    : sqlite3_errcode(db);
      sqlite3_reset(vector_stmt);
      sqlite3_clear_bindings(vector_stmt);
    } else {
      sqlite3_bind_int(level_stmt, 1, g->levels[id]);
      sqlite3_bind_int(level_stmt, 2, id);
      rc = sqlite3_step(level_stmt) == SQLITE_DONE ? SQLITE_OK
                                                   : sqlite3_errcode(db);
      sqlite3_reset(level_stmt);
      sqlite3_clear_bindings(level_stmt);
    }
    if (rc != SQLITE_OK) {
      goto rollback;
    }
    for (layer = 0; layer <= g->levels[id] && layer < g->alloc_levels; layer++) {
      VhNeighborList *list = vh_links_at(g, id, layer);
      if (!list || list->count == 0) {
        continue;
      }
      sqlite3_bind_int(edge_stmt, 1, id);
      sqlite3_bind_int(edge_stmt, 2, layer);
      sqlite3_bind_blob(edge_stmt, 3, list->ids,
                        list->count * (int)sizeof(int32_t), SQLITE_TRANSIENT);
      rc = sqlite3_step(edge_stmt) == SQLITE_DONE ? SQLITE_OK : sqlite3_errcode(db);
      sqlite3_reset(edge_stmt);
      sqlite3_clear_bindings(edge_stmt);
      if (rc != SQLITE_OK) {
        goto rollback;
      }
    }
  }
  rc = vh_meta_get_int(db, db_name, meta_name, "generation", 0, &gen);
  if (rc == SQLITE_OK && (gen < 0 || gen == LLONG_MAX)) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_set_int(db, db_name, meta_name, "generation", gen + 1);
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_set_int(db, db_name, meta_name, "entry_node", g->entry_node);
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_set_int(db, db_name, meta_name, "max_level", g->max_level);
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_set_int(db, db_name, meta_name, "graph_valid", 1);
  }
  if (rc != SQLITE_OK) {
    goto rollback;
  }
  rc = sqlite3_exec(db, "RELEASE vechnsw_rebuild", 0, 0, 0);
  goto done;

rollback:
  sqlite3_exec(db, "ROLLBACK TO vechnsw_rebuild", 0, 0, 0);
  sqlite3_exec(db, "RELEASE vechnsw_rebuild", 0, 0, 0);

done:
  sqlite3_finalize(edge_stmt);
  sqlite3_finalize(level_stmt);
  sqlite3_finalize(vector_stmt);
  sqlite3_free(qgraph);
  sqlite3_free(qvectors);
  sqlite3_free(sql);
  return rc;
}

static int vh_rebuild_table(sqlite3 *db, const char *db_name,
                            const char *table_name, char **pzErr) {
  VhConfig cfg;
  char *meta = vh_shadow_name(table_name, "meta");
  char *vectors = vh_shadow_name(table_name, "vectors");
  char *graph_name = vh_shadow_name(table_name, "graph");
  sqlite3_int64 v = 0;
  char *qmeta = 0;
  char *sql = 0;
  sqlite3_stmt *stmt = 0;
  int rc = SQLITE_OK;
  VhGraph *g = 0;
  if (!meta || !vectors || !graph_name) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  qmeta = vh_qualified(db_name, meta);
  if (!qmeta) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  sql = sqlite3_mprintf("SELECT value FROM %s WHERE key='metric'", qmeta);
  if (!sql) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  sql = 0;
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const char *metric = (const char *)sqlite3_column_text(stmt, 0);
    if (!metric || vh_parse_metric(metric, &cfg.metric) != SQLITE_OK) {
      rc = SQLITE_ERROR;
      if (pzErr) {
        *pzErr = sqlite3_mprintf("invalid metric metadata for %s", table_name);
      }
      goto done;
    }
    rc = SQLITE_OK;
  } else {
    rc = SQLITE_ERROR;
    if (pzErr) {
      *pzErr = sqlite3_mprintf("vec_hnsw table not found: %s", table_name);
    }
    goto done;
  }
  sqlite3_finalize(stmt);
  stmt = 0;
  rc = vh_meta_get_int(db, db_name, meta, "dim", 0, &v);
  if (rc == SQLITE_OK) {
    rc = vh_i64_to_int(v, &cfg.dim);
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "m", VH_DEFAULT_M, &v);
    if (rc == SQLITE_OK) {
      rc = vh_i64_to_int(v, &cfg.m);
    }
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "ef_construction",
                         VH_DEFAULT_EF_CONSTRUCTION, &v);
    if (rc == SQLITE_OK) {
      rc = vh_i64_to_int(v, &cfg.ef_construction);
    }
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "ef_search", VH_DEFAULT_EF_SEARCH,
                         &v);
    if (rc == SQLITE_OK) {
      rc = vh_i64_to_int(v, &cfg.ef_search);
    }
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "build_threads",
                         VH_DEFAULT_BUILD_THREADS, &v);
    if (rc == SQLITE_OK) {
      rc = vh_i64_to_int(v, &cfg.build_threads);
    }
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "build_batch_size",
                         VH_DEFAULT_BUILD_BATCH_SIZE, &v);
    if (rc == SQLITE_OK) {
      rc = vh_i64_to_int(v, &cfg.build_batch_size);
    }
  }
  if (rc == SQLITE_OK) {
    rc = vh_meta_get_int(db, db_name, meta, "seed", VH_DEFAULT_SEED_I64, &v);
    cfg.seed = (uint64_t)v;
  }
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_validate_config(&cfg, 0, pzErr);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_load_vectors_compact(db, db_name, vectors, &cfg, &g);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_build_graph(g);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = vh_persist_graph(db, db_name, meta, vectors, graph_name, g);

done:
  sqlite3_finalize(stmt);
  sqlite3_free(meta);
  sqlite3_free(vectors);
  sqlite3_free(graph_name);
  sqlite3_free(qmeta);
  sqlite3_free(sql);
  vh_graph_free(g);
  return rc;
}

static int vh_load_or_build_cache(VhTable *tab) {
  sqlite3_int64 generation = 0;
  sqlite3_int64 data_version = 0;
  sqlite3_int64 graph_valid = 0;
  sqlite3_int64 entry = -1;
  sqlite3_int64 max_level = -1;
  int rc;
  rc = vh_get_data_version(tab->db, tab->db_name, &data_version);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_meta_get_int(tab->db, tab->db_name, tab->meta_name, "generation", 0,
                       &generation);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (generation < 0) {
    return SQLITE_CORRUPT_VTAB;
  }
  if (tab->cache && tab->cache_generation == generation &&
      tab->cache_data_version == data_version) {
    return SQLITE_OK;
  }
  vh_graph_free(tab->cache);
  tab->cache = 0;
  tab->cache_generation = -1;
  tab->cache_data_version = -1;

  rc = vh_meta_get_int(tab->db, tab->db_name, tab->meta_name, "graph_valid", 0,
                       &graph_valid);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (graph_valid != 0 && graph_valid != 1) {
    return SQLITE_CORRUPT_VTAB;
  }
  rc = vh_meta_get_int(tab->db, tab->db_name, tab->meta_name, "entry_node", -1,
                       &entry);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_meta_get_int(tab->db, tab->db_name, tab->meta_name, "max_level", -1,
                       &max_level);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_load_vectors(tab->db, tab->db_name, tab->vectors_name, &tab->cfg,
                       graph_valid ? 0 : 1, &tab->cache);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (graph_valid && tab->cache->active_count > 0) {
    tab->cache->entry_node = (int)entry;
    tab->cache->max_level = (int)max_level;
    if (entry < 0 || entry >= tab->cache->n ||
        tab->cache->deleted[(int)entry] || max_level < 0 ||
        max_level > INT_MAX ||
        tab->cache->max_level + 1 > tab->cache->alloc_levels) {
      vh_graph_free(tab->cache);
      tab->cache = 0;
      return SQLITE_CORRUPT_VTAB;
    }
    rc = vh_load_graph_edges(tab->db, tab->db_name, tab->graph_name,
                             tab->cache);
  } else {
    rc = vh_build_graph(tab->cache);
  }
  if (rc != SQLITE_OK) {
    vh_graph_free(tab->cache);
    tab->cache = 0;
    return rc;
  }
  tab->cache_generation = generation;
  tab->cache_data_version = data_version;
  return SQLITE_OK;
}

static int vh_ann_search(VhTable *tab, sqlite3_value *query_value, int k,
                         int ef_search, VhCursor *cur) {
  const void *blob = 0;
  float *query = 0;
  VhCandidateList candidates = {0, 0, 0};
  VhScratch scratch;
  VhGraph *g;
  int rc;
  int i;
  memset(&scratch, 0, sizeof(scratch));
  if (k <= 0) {
    return vh_vtab_error(&tab->base, "k must be positive");
  }
  rc = vh_validate_blob(query_value, tab->cfg.dim, &blob);
  if (rc != SQLITE_OK) {
    if (rc == SQLITE_MISMATCH) {
      return vh_vtab_error(&tab->base, "query vector values must be finite");
    }
    return vh_vtab_error(&tab->base,
                         "query vector must be a float32 BLOB of %d bytes",
                         tab->cfg.dim * 4);
  }
  rc = vh_load_or_build_cache(tab);
  if (rc != SQLITE_OK) {
    return rc;
  }
  g = tab->cache;
  if (!g || g->active_count == 0 || g->entry_node < 0) {
    cur->eof = 1;
    return SQLITE_OK;
  }
  query = (float *)vh_malloc((sqlite3_uint64)tab->cfg.dim * sizeof(float));
  if (!query) {
    return SQLITE_NOMEM;
  }
  memcpy(query, blob, (size_t)tab->cfg.dim * sizeof(float));
  if (tab->cfg.metric == VH_METRIC_COSINE) {
    vh_normalize(query, tab->cfg.dim);
  }
  if (ef_search < 1 || ef_search > VH_MAX_EF) {
    sqlite3_free(query);
    return vh_vtab_error(&tab->base, "ef_search must be between 1 and %d",
                         VH_MAX_EF);
  }
  if (ef_search < k) {
    if (k > VH_MAX_EF) {
      sqlite3_free(query);
      return vh_vtab_error(&tab->base, "k must be no greater than %d",
                           VH_MAX_EF);
    }
    ef_search = k;
  }
  rc = vh_scratch_init(&scratch, g->n);
  if (rc != SQLITE_OK) {
    sqlite3_free(query);
    return rc;
  }

  int ep = g->entry_node;
  for (i = g->max_level; i > 0; i--) {
    ep = vh_query_greedy_search(g, query, ep, i);
  }
  rc = vh_query_search_layer(g, &scratch, query, ep, ef_search, &candidates);
  if (rc == SQLITE_OK && candidates.count > 1) {
    qsort(candidates.items, (size_t)candidates.count, sizeof(VhCandidate),
          vh_candidate_cmp_asc);
  }
  if (rc == SQLITE_OK) {
    int limit = candidates.count < k ? candidates.count : k;
    cur->rowids =
        (sqlite3_int64 *)vh_malloc((sqlite3_uint64)limit * sizeof(sqlite3_int64));
    cur->distances = (float *)vh_malloc((sqlite3_uint64)limit * sizeof(float));
    if ((!cur->rowids || !cur->distances) && limit > 0) {
      rc = SQLITE_NOMEM;
    } else {
      cur->result_count = limit;
      for (i = 0; i < limit; i++) {
        cur->rowids[i] = g->rowids[candidates.items[i].id];
        cur->distances[i] = vh_metric_output(g, candidates.items[i].dist);
      }
      cur->eof = limit == 0;
    }
  }
  sqlite3_free(query);
  vh_scratch_free(&scratch);
  vh_candidate_list_free(&candidates);
  return rc;
}

static int vh_configure_table(VhTable *tab, sqlite3 *db, const char *db_name,
                              const char *table_name, const VhConfig *cfg) {
  tab->db = db;
  tab->db_name = vh_strdup(db_name);
  tab->table_name = vh_strdup(table_name);
  tab->meta_name = vh_shadow_name(table_name, "meta");
  tab->vectors_name = vh_shadow_name(table_name, "vectors");
  tab->graph_name = vh_shadow_name(table_name, "graph");
  tab->cfg = *cfg;
  tab->cache_generation = -1;
  tab->cache_data_version = -1;
  if (!tab->db_name || !tab->table_name || !tab->meta_name ||
      !tab->vectors_name || !tab->graph_name) {
    return SQLITE_NOMEM;
  }
  return SQLITE_OK;
}

static void vh_table_free(VhTable *tab) {
  if (!tab) {
    return;
  }
  vh_graph_free(tab->cache);
  sqlite3_free(tab->db_name);
  sqlite3_free(tab->table_name);
  sqlite3_free(tab->meta_name);
  sqlite3_free(tab->vectors_name);
  sqlite3_free(tab->graph_name);
  sqlite3_free(tab);
}

static int vhCreateConnect(sqlite3 *db, void *pAux, int argc,
                           const char *const *argv, sqlite3_vtab **ppVtab,
                           char **pzErr, int is_create) {
  VhConfig cfg;
  VhTable *tab = 0;
  int rc;
  (void)pAux;
  rc = vh_parse_args(argc, argv, &cfg, pzErr);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_declare_vtab(
      db, "CREATE TABLE x(embedding BLOB, distance REAL HIDDEN, k INTEGER "
          "HIDDEN, ef_search INTEGER HIDDEN)");
  if (rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_vtab_config(db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);

  tab = (VhTable *)vh_malloc(sizeof(VhTable));
  if (!tab) {
    return SQLITE_NOMEM;
  }
  memset(tab, 0, sizeof(*tab));
  rc = vh_configure_table(tab, db, argv[1], argv[2], &cfg);
  if (rc != SQLITE_OK) {
    vh_table_free(tab);
    return rc;
  }
  if (is_create) {
    rc = vh_create_shadow(db, argv[1], argv[2], &cfg, 1, pzErr);
    if (rc != SQLITE_OK) {
      vh_table_free(tab);
      return rc;
    }
  }
  *ppVtab = &tab->base;
  return SQLITE_OK;
}

static int vhCreate(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                    sqlite3_vtab **ppVtab, char **pzErr) {
  return vhCreateConnect(db, pAux, argc, argv, ppVtab, pzErr, 1);
}

static int vhConnect(sqlite3 *db, void *pAux, int argc, const char *const *argv,
                     sqlite3_vtab **ppVtab, char **pzErr) {
  return vhCreateConnect(db, pAux, argc, argv, ppVtab, pzErr, 0);
}

static int vhBestIndex(sqlite3_vtab *pVTab, sqlite3_index_info *pIdxInfo) {
  int i;
  int argv_index = 1;
  int flags = 0;
  int match_idx = -1;
  int k_idx = -1;
  int ef_idx = -1;
  (void)pVTab;
  for (i = 0; i < pIdxInfo->nConstraint; i++) {
    struct sqlite3_index_constraint *c = &pIdxInfo->aConstraint[i];
    if (!c->usable) {
      continue;
    }
    if (c->iColumn == 0 && c->op == SQLITE_INDEX_CONSTRAINT_MATCH) {
      flags |= VH_IDX_MATCH;
      match_idx = i;
    } else if (c->iColumn == 2 && c->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      flags |= VH_IDX_K;
      k_idx = i;
    } else if (c->iColumn == 3 && c->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      flags |= VH_IDX_EF;
      ef_idx = i;
    }
  }
  if (match_idx >= 0) {
    pIdxInfo->aConstraintUsage[match_idx].argvIndex = argv_index++;
    pIdxInfo->aConstraintUsage[match_idx].omit = 1;
  }
  if (k_idx >= 0) {
    pIdxInfo->aConstraintUsage[k_idx].argvIndex = argv_index++;
    pIdxInfo->aConstraintUsage[k_idx].omit = 1;
  }
  if (ef_idx >= 0) {
    pIdxInfo->aConstraintUsage[ef_idx].argvIndex = argv_index++;
    pIdxInfo->aConstraintUsage[ef_idx].omit = 1;
  }
  if ((flags & (VH_IDX_MATCH | VH_IDX_K)) == (VH_IDX_MATCH | VH_IDX_K)) {
    pIdxInfo->idxNum = flags;
    pIdxInfo->estimatedCost = 100.0;
    pIdxInfo->estimatedRows = 10;
    if (pIdxInfo->nOrderBy == 1 && pIdxInfo->aOrderBy[0].iColumn == 1 &&
        !pIdxInfo->aOrderBy[0].desc) {
      pIdxInfo->orderByConsumed = 1;
    }
  } else {
    pIdxInfo->idxNum = flags;
    pIdxInfo->estimatedCost = 1000000000.0;
    pIdxInfo->estimatedRows = 1000000;
  }
  return SQLITE_OK;
}

static int vhDisconnect(sqlite3_vtab *pVTab) {
  vh_table_free((VhTable *)pVTab);
  return SQLITE_OK;
}

static int vhDestroy(sqlite3_vtab *pVTab) {
  VhTable *tab = (VhTable *)pVTab;
  char *qmeta = vh_qualified(tab->db_name, tab->meta_name);
  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *qgraph = vh_qualified(tab->db_name, tab->graph_name);
  if (qgraph) {
    vh_execf(tab->db, 0, "DROP TABLE IF EXISTS %s", qgraph);
  }
  if (qvectors) {
    vh_execf(tab->db, 0, "DROP TABLE IF EXISTS %s", qvectors);
  }
  if (qmeta) {
    vh_execf(tab->db, 0, "DROP TABLE IF EXISTS %s", qmeta);
  }
  sqlite3_free(qmeta);
  sqlite3_free(qvectors);
  sqlite3_free(qgraph);
  vh_table_free(tab);
  return SQLITE_OK;
}

static int vhOpen(sqlite3_vtab *pVTab, sqlite3_vtab_cursor **ppCursor) {
  VhCursor *cur = (VhCursor *)vh_malloc(sizeof(VhCursor));
  if (!cur) {
    return SQLITE_NOMEM;
  }
  memset(cur, 0, sizeof(*cur));
  cur->tab = (VhTable *)pVTab;
  *ppCursor = &cur->base;
  return SQLITE_OK;
}

static void vh_cursor_clear(VhCursor *cur) {
  sqlite3_finalize(cur->scan_stmt);
  cur->scan_stmt = 0;
  sqlite3_free(cur->rowids);
  sqlite3_free(cur->distances);
  cur->rowids = 0;
  cur->distances = 0;
  cur->result_count = 0;
  cur->result_pos = 0;
  cur->eof = 0;
  cur->mode = 0;
}

static int vhClose(sqlite3_vtab_cursor *pCursor) {
  VhCursor *cur = (VhCursor *)pCursor;
  vh_cursor_clear(cur);
  sqlite3_free(cur);
  return SQLITE_OK;
}

static int vhFilter(sqlite3_vtab_cursor *pCursor, int idxNum,
                    const char *idxStr, int argc, sqlite3_value **argv) {
  VhCursor *cur = (VhCursor *)pCursor;
  VhTable *tab = cur->tab;
  int arg = 0;
  (void)idxStr;
  (void)argc;
  vh_cursor_clear(cur);
  if ((idxNum & (VH_IDX_MATCH | VH_IDX_K)) == (VH_IDX_MATCH | VH_IDX_K)) {
    int k;
    int ef = tab->cfg.ef_search;
    sqlite3_value *query = argv[arg++];
    k = sqlite3_value_int(argv[arg++]);
    if (idxNum & VH_IDX_EF) {
      ef = sqlite3_value_int(argv[arg++]);
    }
    cur->mode = 1;
    return vh_ann_search(tab, query, k, ef, cur);
  }
  if (idxNum & VH_IDX_MATCH) {
    return vh_vtab_error(&tab->base,
                         "ANN query requires embedding MATCH ? and k = ?");
  }

  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *sql;
  int rc;
  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT rowid_value,embedding FROM %s "
                        "WHERE deleted=0 ORDER BY rowid_value",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(tab->db, sql, -1, &cur->scan_stmt, 0);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  cur->mode = 2;
  rc = sqlite3_step(cur->scan_stmt);
  if (rc == SQLITE_ROW) {
    cur->eof = 0;
    return SQLITE_OK;
  }
  if (rc == SQLITE_DONE) {
    cur->eof = 1;
    return SQLITE_OK;
  }
  return sqlite3_errcode(tab->db);
}

static int vhNext(sqlite3_vtab_cursor *pCursor) {
  VhCursor *cur = (VhCursor *)pCursor;
  if (cur->mode == 1) {
    cur->result_pos++;
    if (cur->result_pos >= cur->result_count) {
      cur->eof = 1;
    }
    return SQLITE_OK;
  }
  if (cur->mode == 2) {
    int rc = sqlite3_step(cur->scan_stmt);
    if (rc == SQLITE_ROW) {
      cur->eof = 0;
      return SQLITE_OK;
    }
    if (rc == SQLITE_DONE) {
      cur->eof = 1;
      return SQLITE_OK;
    }
    return sqlite3_errcode(cur->tab->db);
  }
  cur->eof = 1;
  return SQLITE_OK;
}

static int vhEof(sqlite3_vtab_cursor *pCursor) {
  return ((VhCursor *)pCursor)->eof;
}

static int vhColumn(sqlite3_vtab_cursor *pCursor, sqlite3_context *ctx, int i) {
  VhCursor *cur = (VhCursor *)pCursor;
  if (cur->mode == 1) {
    if (i == 1 && cur->result_pos < cur->result_count) {
      sqlite3_result_double(ctx, cur->distances[cur->result_pos]);
    } else {
      sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
  }
  if (cur->mode == 2 && cur->scan_stmt) {
    if (i == 0) {
      const void *blob = sqlite3_column_blob(cur->scan_stmt, 1);
      int bytes = sqlite3_column_bytes(cur->scan_stmt, 1);
      sqlite3_result_blob(ctx, blob, bytes, SQLITE_TRANSIENT);
    } else {
      sqlite3_result_null(ctx);
    }
    return SQLITE_OK;
  }
  sqlite3_result_null(ctx);
  return SQLITE_OK;
}

static int vhRowid(sqlite3_vtab_cursor *pCursor, sqlite3_int64 *pRowid) {
  VhCursor *cur = (VhCursor *)pCursor;
  if (cur->mode == 1 && cur->result_pos < cur->result_count) {
    *pRowid = cur->rowids[cur->result_pos];
    return SQLITE_OK;
  }
  if (cur->mode == 2 && cur->scan_stmt) {
    *pRowid = sqlite3_column_int64(cur->scan_stmt, 0);
    return SQLITE_OK;
  }
  *pRowid = 0;
  return SQLITE_OK;
}

static int vh_next_node_id(VhTable *tab, sqlite3_int64 *out) {
  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT COALESCE(MAX(node_id), -1)+1 FROM %s",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(tab->db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    int step_rc = sqlite3_step(stmt);
    if (step_rc == SQLITE_ROW) {
      *out = sqlite3_column_int64(stmt, 0);
    } else if (step_rc != SQLITE_DONE) {
      rc = sqlite3_errcode(tab->db);
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_assign_rowid(VhTable *tab, sqlite3_int64 *out) {
  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT COALESCE(MAX(rowid_value), 0)+1 FROM %s "
                        "WHERE deleted=0",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(tab->db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    int step_rc = sqlite3_step(stmt);
    if (step_rc == SQLITE_ROW) {
      *out = sqlite3_column_int64(stmt, 0);
    } else if (step_rc != SQLITE_DONE) {
      rc = sqlite3_errcode(tab->db);
    }
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_tombstone_rowid(VhTable *tab, sqlite3_int64 rowid) {
  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  int rc;
  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("UPDATE %s SET deleted=1 WHERE rowid_value=?1 AND "
                        "deleted=0",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(tab->db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, rowid);
    rc = sqlite3_step(stmt) == SQLITE_DONE ? SQLITE_OK : sqlite3_errcode(tab->db);
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vh_insert_vector(VhTable *tab, sqlite3_int64 rowid,
                            const void *blob) {
  char *qvectors = vh_qualified(tab->db_name, tab->vectors_name);
  char *sql;
  sqlite3_stmt *stmt = 0;
  sqlite3_int64 node_id = 0;
  int rc;
  if (!qvectors) {
    return SQLITE_NOMEM;
  }
  rc = vh_next_node_id(tab, &node_id);
  if (rc != SQLITE_OK) {
    sqlite3_free(qvectors);
    return rc;
  }
  sql = sqlite3_mprintf("INSERT INTO %s(node_id,rowid_value,level,deleted,"
                        "embedding) VALUES(?1,?2,0,0,?3)",
                        qvectors);
  sqlite3_free(qvectors);
  if (!sql) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(tab->db, sql, -1, &stmt, 0);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(stmt, 1, node_id);
    sqlite3_bind_int64(stmt, 2, rowid);
    sqlite3_bind_blob(stmt, 3, blob, tab->cfg.dim * (int)sizeof(float),
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(stmt) == SQLITE_DONE ? SQLITE_OK : sqlite3_errcode(tab->db);
  }
  sqlite3_finalize(stmt);
  return rc;
}

static int vhUpdate(sqlite3_vtab *pVTab, int argc, sqlite3_value **argv,
                    sqlite3_int64 *pRowid) {
  VhTable *tab = (VhTable *)pVTab;
  int rc;
  if (argc == 1) {
    sqlite3_int64 old_rowid = sqlite3_value_int64(argv[0]);
    rc = vh_tombstone_rowid(tab, old_rowid);
    if (rc == SQLITE_OK) {
      rc = vh_mark_graph_stale(tab);
    }
    return rc;
  }

  sqlite3_int64 old_rowid = sqlite3_value_type(argv[0]) == SQLITE_NULL
                                ? 0
                                : sqlite3_value_int64(argv[0]);
  sqlite3_int64 new_rowid;
  const void *blob = 0;
  if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    rc = vh_assign_rowid(tab, &new_rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  } else {
    new_rowid = sqlite3_value_int64(argv[1]);
  }
  rc = vh_validate_blob(argv[2], tab->cfg.dim, &blob);
  if (rc != SQLITE_OK) {
    if (rc == SQLITE_MISMATCH) {
      return vh_vtab_error(pVTab, "embedding vector values must be finite");
    }
    return vh_vtab_error(pVTab, "embedding must be a float32 BLOB of %d bytes",
                         tab->cfg.dim * 4);
  }
  if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
    rc = vh_tombstone_rowid(tab, old_rowid);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  rc = vh_tombstone_rowid(tab, new_rowid);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = vh_insert_vector(tab, new_rowid, blob);
  if (rc == SQLITE_OK) {
    rc = vh_mark_graph_stale(tab);
  }
  if (rc == SQLITE_OK && pRowid) {
    *pRowid = new_rowid;
  }
  return rc;
}

static int vhRename(sqlite3_vtab *pVTab, const char *zNew) {
  (void)zNew;
  return vh_vtab_error(pVTab, "ALTER TABLE RENAME is not supported by vec_hnsw");
}

static sqlite3_module VhModule = {
    3,          /* iVersion */
    vhCreate,   /* xCreate */
    vhConnect,  /* xConnect */
    vhBestIndex,
    vhDisconnect,
    vhDestroy,
    vhOpen,
    vhClose,
    vhFilter,
    vhNext,
    vhEof,
    vhColumn,
    vhRowid,
    vhUpdate,
    0,        /* xBegin */
    0,        /* xSync */
    0,        /* xCommit */
    0,        /* xRollback */
    0,        /* xFindFunction */
    vhRename, /* xRename */
    0,        /* xSavepoint */
    0,        /* xRelease */
    0,        /* xRollbackTo */
    0,        /* xShadowName */
    0,        /* xIntegrity */
};

static void vh_vec_dim(sqlite3_context *ctx, int argc, sqlite3_value **argv) {
  int bytes;
  (void)argc;
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB) {
    sqlite3_result_null(ctx);
    return;
  }
  bytes = sqlite3_value_bytes(argv[0]);
  if (bytes % (int)sizeof(float) != 0) {
    sqlite3_result_error(ctx, "vector blob size is not divisible by 4", -1);
    return;
  }
  if (!vh_validate_dim(bytes / (int)sizeof(float))) {
    sqlite3_result_error(ctx, "vector dimension exceeds maximum 4096", -1);
    return;
  }
  if (vh_validate_finite_f32(sqlite3_value_blob(argv[0]),
                             bytes / (int)sizeof(float)) != SQLITE_OK) {
    sqlite3_result_error(ctx, "vector values must be finite", -1);
    return;
  }
  sqlite3_result_int(ctx, bytes / (int)sizeof(float));
}

static void vh_vec_distance(sqlite3_context *ctx, int argc,
                            sqlite3_value **argv) {
  const float *a;
  const float *b;
  const char *metric_text;
  VhMetric metric;
  int bytes_a;
  int bytes_b;
  int dim;
  float d;
  (void)argc;
  if (sqlite3_value_type(argv[0]) != SQLITE_BLOB ||
      sqlite3_value_type(argv[1]) != SQLITE_BLOB ||
      sqlite3_value_type(argv[2]) == SQLITE_NULL) {
    sqlite3_result_null(ctx);
    return;
  }
  bytes_a = sqlite3_value_bytes(argv[0]);
  bytes_b = sqlite3_value_bytes(argv[1]);
  if (bytes_a != bytes_b || bytes_a % (int)sizeof(float) != 0) {
    sqlite3_result_error(ctx, "vectors must have equal float32 blob sizes", -1);
    return;
  }
  dim = bytes_a / (int)sizeof(float);
  if (!vh_validate_dim(dim)) {
    sqlite3_result_error(ctx, "vector dimension exceeds maximum 4096", -1);
    return;
  }
  if (vh_validate_finite_f32(sqlite3_value_blob(argv[0]), dim) != SQLITE_OK ||
      vh_validate_finite_f32(sqlite3_value_blob(argv[1]), dim) != SQLITE_OK) {
    sqlite3_result_error(ctx, "vector values must be finite", -1);
    return;
  }
  metric_text = (const char *)sqlite3_value_text(argv[2]);
  if (!metric_text || vh_parse_metric(metric_text, &metric) != SQLITE_OK) {
    sqlite3_result_error(ctx, "invalid metric", -1);
    return;
  }
  a = (const float *)sqlite3_value_blob(argv[0]);
  b = (const float *)sqlite3_value_blob(argv[1]);
  if (metric == VH_METRIC_L2) {
    sqlite3_result_double(ctx, sqrtf(vh_l2_fast(a, b, dim)));
  } else if (metric == VH_METRIC_IP) {
    sqlite3_result_double(ctx, -vh_dot_fast(a, b, dim));
  } else {
    float na = vh_dot_fast(a, a, dim);
    float nb = vh_dot_fast(b, b, dim);
    if (na <= 0.0f || nb <= 0.0f) {
      d = 1.0f;
    } else {
      d = 1.0f - vh_dot_fast(a, b, dim) / sqrtf(na * nb);
    }
    sqlite3_result_double(ctx, d);
  }
}

static void vh_rebuild_func(sqlite3_context *ctx, int argc,
                            sqlite3_value **argv) {
  const char *db_name;
  const char *table_name;
  sqlite3 *db = sqlite3_context_db_handle(ctx);
  char *err = 0;
  int rc;
  if (argc != 2 || sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    sqlite3_result_error(ctx, "database name and table name are required", -1);
    return;
  }
  db_name = (const char *)sqlite3_value_text(argv[0]);
  table_name = (const char *)sqlite3_value_text(argv[1]);
  if (!db_name || !db_name[0] || !table_name || !table_name[0]) {
    sqlite3_result_error(ctx, "database name and table name are required", -1);
    return;
  }
  rc = vh_rebuild_table(db, db_name, table_name, &err);
  if (rc != SQLITE_OK) {
    if (err) {
      sqlite3_result_error(ctx, err, -1);
      sqlite3_free(err);
    } else {
      sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    }
    return;
  }
  sqlite3_result_int(ctx, 1);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_vechnsw_init(sqlite3 *db, char **pzErrMsg,
                         const sqlite3_api_routines *pApi) {
  int rc;
  SQLITE_EXTENSION_INIT2(pApi);
  rc = sqlite3_create_module(db, "vec_hnsw", &VhModule, 0);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function(db, "vec_dim", 1,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0,
                               vh_vec_dim, 0, 0);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function(db, "vec_distance", 3,
                               SQLITE_UTF8 | SQLITE_DETERMINISTIC, 0,
                               vh_vec_distance, 0, 0);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_create_function(db, "vec_hnsw_rebuild", 2, SQLITE_UTF8, 0,
                               vh_rebuild_func, 0, 0);
  if (rc != SQLITE_OK && pzErrMsg) {
    *pzErrMsg = sqlite3_mprintf("failed to register vec_hnsw functions");
  }
  return rc;
}
