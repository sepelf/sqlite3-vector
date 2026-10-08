#include "sqlite3ext.h"
SQLITE_EXTENSION_INIT1

#include "build_arena.h"
#include "integrity.h"
#include "sqlite_hnsw_version.h"
#include "vector.h"

#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HNSW_FORMAT_VERSION 1
#define HNSW_MAX_DIMENSIONS 4096
#define HNSW_MAX_LEVEL 32
#define HNSW_ARENA_THRESHOLD 4096

enum {
  COL_VECTOR = 0,
  COL_DISTANCE,
  COL_K,
  COL_EF_SEARCH,
  COL_VISITED_COUNT,
  COL_COMMAND,
  HNSW_COLUMN_COUNT
};

enum { PLAN_SCAN = 1, PLAN_ROWID = 2, PLAN_ANN = 3 };

typedef struct HnswVtab HnswVtab;
typedef struct HnswCursor HnswCursor;

typedef struct HnswState {
  sqlite3_int64 entry_node;
  int max_level;
  sqlite3_int64 next_node_id;
  sqlite3_int64 live_count;
  sqlite3_int64 tombstone_count;
  sqlite3_int64 change_seq;
} HnswState;

typedef struct NodeCacheEntry NodeCacheEntry;
typedef struct EdgeCacheEntry EdgeCacheEntry;

struct NodeCacheEntry {
  sqlite3_int64 node_id;
  float *vector;
  double norm;
  int deleted;
  int level;
  int pins;
  size_t bytes;
  NodeCacheEntry *hash_next;
  NodeCacheEntry *lru_previous;
  NodeCacheEntry *lru_next;
};

typedef struct NodeView {
  const float *vector;
  double norm;
  int deleted;
  int level;
  NodeCacheEntry *cached;
  float *owned;
} NodeView;

struct EdgeCacheEntry {
  sqlite3_int64 node_id;
  int layer;
  sqlite3_int64 *ids;
  int count;
  int dirty;
  int pins;
  size_t bytes;
  EdgeCacheEntry *hash_next;
  EdgeCacheEntry *lru_previous;
  EdgeCacheEntry *lru_next;
};

typedef struct EdgeView {
  const sqlite3_int64 *ids;
  const uint32_t *compact_ids;
  int count;
  EdgeCacheEntry *cached;
  sqlite3_int64 *owned;
} EdgeView;

typedef struct HnswQuerySnapshot {
  float *vectors;
  double *norms;
  sqlite3_int64 *rowids;
  size_t *neighbor_offsets;
  size_t *count_offsets;
  uint32_t *neighbors;
  uint16_t *neighbor_counts;
  unsigned char *levels;
  unsigned char *deleted;
  unsigned char *valid;
  unsigned char *has_rowid;
  size_t node_capacity;
  size_t neighbor_slots;
  size_t count_slots;
  sqlite3_uint64 estimated_bytes;
  sqlite3_int64 change_seq;
  int ready;
  int attempted;
} HnswQuerySnapshot;

typedef struct ResultRow {
  sqlite3_int64 rowid;
  sqlite3_int64 node_id;
  double distance;
} ResultRow;

struct HnswVtab {
  sqlite3_vtab base;
  sqlite3 *db;
  char *schema;
  char *name;
  char *vector_name;
  int dims;
  int metric;
  int m;
  int ef_construction;
  int cache_size_mb;
  int build_memory_mb;
  int build_threads;
  sqlite3_stmt *read_state_stmt;
  sqlite3_stmt *write_state_stmt;
  sqlite3_stmt *read_node_stmt;
  sqlite3_stmt *read_edges_stmt;
  sqlite3_stmt *write_edges_stmt;
  sqlite3_stmt *read_rowid_stmt;
  sqlite3_stmt *read_node_id_stmt;
  sqlite3_stmt *insert_node_stmt;
  sqlite3_stmt *insert_row_stmt;
  NodeCacheEntry **cache_buckets;
  size_t cache_bucket_count;
  size_t cache_bytes;
  size_t cache_budget;
  NodeCacheEntry *cache_head;
  NodeCacheEntry *cache_tail;
  EdgeCacheEntry **edge_cache_buckets;
  size_t edge_cache_bucket_count;
  EdgeCacheEntry *edge_cache_head;
  EdgeCacheEntry *edge_cache_tail;
  sqlite3_int64 cache_seq;
  int defer_edge_writes;
  int write_dirty;
  HnswQuerySnapshot snapshot;
};

struct HnswCursor {
  sqlite3_vtab_cursor base;
  ResultRow *rows;
  int count;
  int capacity;
  int position;
  int k;
  int ef_search;
  int visited_count;
  uint32_t *visited_epochs;
  size_t visited_capacity;
  uint32_t visited_epoch;
};

typedef struct HeapItem {
  sqlite3_int64 id;
  double distance;
} HeapItem;

typedef struct Heap {
  HeapItem *items;
  int count;
  int capacity;
  int is_min;
} Heap;

typedef struct IdSet {
  sqlite3_int64 *slots;
  size_t capacity;
  size_t count;
} IdSet;

typedef struct IdList {
  sqlite3_int64 *ids;
  int count;
} IdList;

typedef struct SearchOutput {
  HeapItem *items;
  int count;
  int visited;
} SearchOutput;

static int hnsw_create(sqlite3 *, void *, int, const char *const *,
                       sqlite3_vtab **, char **);
static int hnsw_connect(sqlite3 *, void *, int, const char *const *,
                        sqlite3_vtab **, char **);
static int hnsw_best_index(sqlite3_vtab *, sqlite3_index_info *);
static int hnsw_disconnect(sqlite3_vtab *);
static int hnsw_destroy(sqlite3_vtab *);
static int hnsw_open(sqlite3_vtab *, sqlite3_vtab_cursor **);
static int hnsw_close(sqlite3_vtab_cursor *);
static int hnsw_filter(sqlite3_vtab_cursor *, int, const char *, int,
                       sqlite3_value **);
static int hnsw_next(sqlite3_vtab_cursor *);
static int hnsw_eof(sqlite3_vtab_cursor *);
static int hnsw_column(sqlite3_vtab_cursor *, sqlite3_context *, int);
static int hnsw_rowid(sqlite3_vtab_cursor *, sqlite3_int64 *);
static int hnsw_update(sqlite3_vtab *, int, sqlite3_value **, sqlite3_int64 *);
static int hnsw_rename(sqlite3_vtab *, const char *);
static int hnsw_begin(sqlite3_vtab *);
static int hnsw_sync(sqlite3_vtab *);
static int hnsw_commit(sqlite3_vtab *);
static int hnsw_rollback(sqlite3_vtab *);
static int hnsw_savepoint(sqlite3_vtab *, int);
static int hnsw_release(sqlite3_vtab *, int);
static int hnsw_rollback_to(sqlite3_vtab *, int);
static int hnsw_shadow_name(const char *);
static void query_snapshot_clear(HnswVtab *);
int sqlite3_hnsw_init(sqlite3 *, char **, const sqlite3_api_routines *);
int sqlite3_sqlitehnsw_init(sqlite3 *, char **, const sqlite3_api_routines *);
int sqlite3_extension_init(sqlite3 *, char **, const sqlite3_api_routines *);

static const sqlite3_module hnsw_module = {3,
                                           hnsw_create,
                                           hnsw_connect,
                                           hnsw_best_index,
                                           hnsw_disconnect,
                                           hnsw_destroy,
                                           hnsw_open,
                                           hnsw_close,
                                           hnsw_filter,
                                           hnsw_next,
                                           hnsw_eof,
                                           hnsw_column,
                                           hnsw_rowid,
                                           hnsw_update,
                                           hnsw_begin,
                                           hnsw_sync,
                                           hnsw_commit,
                                           hnsw_rollback,
                                           NULL,
                                           hnsw_rename,
                                           hnsw_savepoint,
                                           hnsw_release,
                                           hnsw_rollback_to,
                                           hnsw_shadow_name
#if SQLITE_VERSION_NUMBER >= 3044000
                                           ,
                                           NULL
#endif
};

static int ascii_equal(const char *a, const char *b) {
  while (*a != '\0' && *b != '\0') {
    if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
      return 0;
    }
    ++a;
    ++b;
  }
  return *a == '\0' && *b == '\0';
}

static char *trim_copy(const char *input) {
  const char *start = input;
  const char *end = input + strlen(input);
  while (start < end && isspace((unsigned char)*start)) {
    ++start;
  }
  while (end > start && isspace((unsigned char)end[-1])) {
    --end;
  }
  return sqlite3_mprintf("%.*s", (int)(end - start), start);
}

static int parse_int_option(const char *text, int minimum, int maximum,
                            int *out) {
  char *end = NULL;
  long value = 0;
  if (text == NULL || *text == '\0') {
    return SQLITE_ERROR;
  }
  value = strtol(text, &end, 10);
  if (*end != '\0' || value < minimum || value > maximum) {
    return SQLITE_ERROR;
  }
  *out = (int)value;
  return SQLITE_OK;
}

static int valid_identifier(const char *name) {
  const unsigned char *p = (const unsigned char *)name;
  if (*p == '\0' || !(isalpha(*p) || *p == '_')) {
    return 0;
  }
  ++p;
  while (*p != '\0') {
    if (!(isalnum(*p) || *p == '_')) {
      return 0;
    }
    ++p;
  }
  return 1;
}

static int parse_vector_declaration(const char *argument, char **name,
                                    int *dims) {
  char *copy = trim_copy(argument);
  char *space = NULL;
  char *type = NULL;
  char *close = NULL;
  int rc = SQLITE_ERROR;
  if (copy == NULL) {
    return SQLITE_NOMEM;
  }
  space = copy;
  while (*space != '\0' && !isspace((unsigned char)*space)) {
    ++space;
  }
  if (*space == '\0') {
    goto done;
  }
  *space++ = '\0';
  while (isspace((unsigned char)*space)) {
    ++space;
  }
  type = space;
  if (sqlite3_strnicmp(type, "FLOAT32(", 8) != 0) {
    goto done;
  }
  close = strchr(type + 8, ')');
  if (close == NULL || close[1] != '\0') {
    goto done;
  }
  *close = '\0';
  if (!valid_identifier(copy) ||
      parse_int_option(type + 8, 1, HNSW_MAX_DIMENSIONS, dims) != SQLITE_OK) {
    goto done;
  }
  *name = sqlite3_mprintf("%s", copy);
  rc = *name == NULL ? SQLITE_NOMEM : SQLITE_OK;
done:
  sqlite3_free(copy);
  return rc;
}

static int parse_arguments(int argc, const char *const *argv, HnswVtab *vtab,
                           char **error) {
  int i = 0;
  int have_vector = 0;
  vtab->metric = HNSW_METRIC_L2;
  vtab->m = 16;
  vtab->ef_construction = 128;
  vtab->cache_size_mb = 256;
  vtab->build_memory_mb = 1024;
  vtab->build_threads = 0;

  for (i = 3; i < argc; ++i) {
    char *arg = trim_copy(argv[i]);
    char *equals = NULL;
    char *key = NULL;
    char *value = NULL;
    int rc = SQLITE_OK;
    if (arg == NULL) {
      return SQLITE_NOMEM;
    }
    equals = strchr(arg, '=');
    if (equals == NULL) {
      if (have_vector) {
        *error = sqlite3_mprintf("hnsw accepts exactly one vector column");
        sqlite3_free(arg);
        return SQLITE_ERROR;
      }
      rc = parse_vector_declaration(arg, &vtab->vector_name, &vtab->dims);
      sqlite3_free(arg);
      if (rc != SQLITE_OK) {
        *error = sqlite3_mprintf(
            "expected a declaration such as embedding FLOAT32(768)");
        return rc;
      }
      have_vector = 1;
      continue;
    }
    *equals = '\0';
    key = trim_copy(arg);
    value = trim_copy(equals + 1);
    sqlite3_free(arg);
    if (key == NULL || value == NULL) {
      sqlite3_free(key);
      sqlite3_free(value);
      return SQLITE_NOMEM;
    }
    if (ascii_equal(key, "metric")) {
      if (ascii_equal(value, "l2")) {
        vtab->metric = HNSW_METRIC_L2;
      } else if (ascii_equal(value, "cosine")) {
        vtab->metric = HNSW_METRIC_COSINE;
      } else if (ascii_equal(value, "ip") ||
                 ascii_equal(value, "inner_product")) {
        vtab->metric = HNSW_METRIC_INNER_PRODUCT;
      } else {
        rc = SQLITE_ERROR;
      }
    } else if (ascii_equal(key, "m")) {
      rc = parse_int_option(value, 2, 64, &vtab->m);
    } else if (ascii_equal(key, "ef_construction")) {
      rc = parse_int_option(value, 4, 1000, &vtab->ef_construction);
    } else if (ascii_equal(key, "cache_size_mb")) {
      rc = parse_int_option(value, 0, 65536, &vtab->cache_size_mb);
    } else if (ascii_equal(key, "build_memory_mb")) {
      rc = parse_int_option(value, 128, 16384, &vtab->build_memory_mb);
    } else if (ascii_equal(key, "build_threads")) {
      rc = parse_int_option(value, 0, 64, &vtab->build_threads);
    } else {
      *error = sqlite3_mprintf("unknown hnsw option: %s", key);
      sqlite3_free(key);
      sqlite3_free(value);
      return SQLITE_ERROR;
    }
    if (rc != SQLITE_OK) {
      *error = sqlite3_mprintf("invalid value for hnsw option %s", key);
      sqlite3_free(key);
      sqlite3_free(value);
      return SQLITE_ERROR;
    }
    sqlite3_free(key);
    sqlite3_free(value);
  }
  if (!have_vector) {
    *error = sqlite3_mprintf("hnsw requires one FLOAT32(N) vector column");
    return SQLITE_ERROR;
  }
  if (vtab->ef_construction < vtab->m * 2) {
    *error = sqlite3_mprintf("ef_construction must be at least 2*m");
    return SQLITE_ERROR;
  }
  return SQLITE_OK;
}

static int exec_sql(sqlite3 *db, char **error, const char *sql) {
  return sqlite3_exec(db, sql, NULL, NULL, error);
}

static char *shadow_table(const HnswVtab *vtab, const char *suffix) {
  return sqlite3_mprintf("\"%w\".\"%w_%w\"", vtab->schema, vtab->name, suffix);
}

static int finish_cached_statement(sqlite3_stmt *statement, int rc) {
  int reset_rc = sqlite3_reset(statement);
  sqlite3_clear_bindings(statement);
  return rc == SQLITE_OK && reset_rc != SQLITE_OK ? reset_rc : rc;
}

static int prepare_cached(HnswVtab *vtab, sqlite3_stmt **slot,
                          const char *sql) {
  if (*slot != NULL) {
    return SQLITE_OK;
  }
  return sqlite3_prepare_v3(vtab->db, sql, -1, SQLITE_PREPARE_PERSISTENT, slot,
                            NULL);
}

static void finalize_cached_statements(HnswVtab *vtab) {
  sqlite3_finalize(vtab->read_state_stmt);
  sqlite3_finalize(vtab->write_state_stmt);
  sqlite3_finalize(vtab->read_node_stmt);
  sqlite3_finalize(vtab->read_edges_stmt);
  sqlite3_finalize(vtab->write_edges_stmt);
  sqlite3_finalize(vtab->read_rowid_stmt);
  sqlite3_finalize(vtab->read_node_id_stmt);
  sqlite3_finalize(vtab->insert_node_stmt);
  sqlite3_finalize(vtab->insert_row_stmt);
  vtab->read_state_stmt = NULL;
  vtab->write_state_stmt = NULL;
  vtab->read_node_stmt = NULL;
  vtab->read_edges_stmt = NULL;
  vtab->write_edges_stmt = NULL;
  vtab->read_rowid_stmt = NULL;
  vtab->read_node_id_stmt = NULL;
  vtab->insert_node_stmt = NULL;
  vtab->insert_row_stmt = NULL;
}

static size_t node_cache_hash(const HnswVtab *vtab, sqlite3_int64 node_id) {
  uint64_t value = (uint64_t)node_id;
  value ^= value >> 33U;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33U;
  return (size_t)(value & (vtab->cache_bucket_count - 1U));
}

static void node_cache_unlink_lru(HnswVtab *vtab, NodeCacheEntry *entry) {
  if (entry->lru_previous != NULL) {
    entry->lru_previous->lru_next = entry->lru_next;
  } else {
    vtab->cache_head = entry->lru_next;
  }
  if (entry->lru_next != NULL) {
    entry->lru_next->lru_previous = entry->lru_previous;
  } else {
    vtab->cache_tail = entry->lru_previous;
  }
}

static void node_cache_link_head(HnswVtab *vtab, NodeCacheEntry *entry) {
  entry->lru_previous = NULL;
  entry->lru_next = vtab->cache_head;
  if (vtab->cache_head != NULL) {
    vtab->cache_head->lru_previous = entry;
  } else {
    vtab->cache_tail = entry;
  }
  vtab->cache_head = entry;
}

static void node_cache_remove(HnswVtab *vtab, NodeCacheEntry *entry) {
  size_t bucket = node_cache_hash(vtab, entry->node_id);
  NodeCacheEntry **link = &vtab->cache_buckets[bucket];
  while (*link != NULL && *link != entry) {
    link = &(*link)->hash_next;
  }
  if (*link == entry) {
    *link = entry->hash_next;
  }
  node_cache_unlink_lru(vtab, entry);
  vtab->cache_bytes -= entry->bytes;
  sqlite3_free(entry->vector);
  sqlite3_free(entry);
}

static void node_cache_clear(HnswVtab *vtab) {
  NodeCacheEntry *entry = vtab->cache_head;
  while (entry != NULL) {
    NodeCacheEntry *next = entry->lru_next;
    vtab->cache_bytes -= entry->bytes;
    sqlite3_free(entry->vector);
    sqlite3_free(entry);
    entry = next;
  }
  sqlite3_free(vtab->cache_buckets);
  vtab->cache_buckets = NULL;
  vtab->cache_bucket_count = 0;
  vtab->cache_head = NULL;
  vtab->cache_tail = NULL;
}

static size_t edge_cache_hash(const HnswVtab *vtab, sqlite3_int64 node_id,
                              int layer) {
  uint64_t value = (uint64_t)node_id ^ ((uint64_t)(unsigned int)layer << 48U);
  value ^= value >> 33U;
  value *= UINT64_C(0xff51afd7ed558ccd);
  value ^= value >> 33U;
  return (size_t)(value & (vtab->edge_cache_bucket_count - 1U));
}

static void edge_cache_unlink_lru(HnswVtab *vtab, EdgeCacheEntry *entry) {
  if (entry->lru_previous != NULL) {
    entry->lru_previous->lru_next = entry->lru_next;
  } else {
    vtab->edge_cache_head = entry->lru_next;
  }
  if (entry->lru_next != NULL) {
    entry->lru_next->lru_previous = entry->lru_previous;
  } else {
    vtab->edge_cache_tail = entry->lru_previous;
  }
}

static void edge_cache_link_head(HnswVtab *vtab, EdgeCacheEntry *entry) {
  entry->lru_previous = NULL;
  entry->lru_next = vtab->edge_cache_head;
  if (vtab->edge_cache_head != NULL) {
    vtab->edge_cache_head->lru_previous = entry;
  } else {
    vtab->edge_cache_tail = entry;
  }
  vtab->edge_cache_head = entry;
}

static EdgeCacheEntry *edge_cache_find(HnswVtab *vtab, sqlite3_int64 node_id,
                                       int layer) {
  EdgeCacheEntry *entry = NULL;
  size_t bucket = 0;
  if (vtab->edge_cache_buckets == NULL) {
    return NULL;
  }
  bucket = edge_cache_hash(vtab, node_id, layer);
  entry = vtab->edge_cache_buckets[bucket];
  while (entry != NULL &&
         (entry->node_id != node_id || entry->layer != layer)) {
    entry = entry->hash_next;
  }
  if (entry != NULL && entry != vtab->edge_cache_head) {
    edge_cache_unlink_lru(vtab, entry);
    edge_cache_link_head(vtab, entry);
  }
  return entry;
}

static void edge_cache_remove(HnswVtab *vtab, EdgeCacheEntry *entry) {
  size_t bucket = edge_cache_hash(vtab, entry->node_id, entry->layer);
  EdgeCacheEntry **link = &vtab->edge_cache_buckets[bucket];
  while (*link != NULL && *link != entry) {
    link = &(*link)->hash_next;
  }
  if (*link == entry) {
    *link = entry->hash_next;
  }
  edge_cache_unlink_lru(vtab, entry);
  vtab->cache_bytes -= entry->bytes;
  sqlite3_free(entry->ids);
  sqlite3_free(entry);
}

static void edge_cache_clear(HnswVtab *vtab) {
  EdgeCacheEntry *entry = vtab->edge_cache_head;
  while (entry != NULL) {
    EdgeCacheEntry *next = entry->lru_next;
    vtab->cache_bytes -= entry->bytes;
    sqlite3_free(entry->ids);
    sqlite3_free(entry);
    entry = next;
  }
  sqlite3_free(vtab->edge_cache_buckets);
  vtab->edge_cache_buckets = NULL;
  vtab->edge_cache_bucket_count = 0;
  vtab->edge_cache_head = NULL;
  vtab->edge_cache_tail = NULL;
}

static void cache_clear_all(HnswVtab *vtab) {
  node_cache_clear(vtab);
  edge_cache_clear(vtab);
}

static void query_snapshot_clear(HnswVtab *vtab) {
  HnswQuerySnapshot *snapshot = &vtab->snapshot;
  sqlite3_free(snapshot->vectors);
  sqlite3_free(snapshot->norms);
  sqlite3_free(snapshot->rowids);
  sqlite3_free(snapshot->neighbor_offsets);
  sqlite3_free(snapshot->count_offsets);
  sqlite3_free(snapshot->neighbors);
  sqlite3_free(snapshot->neighbor_counts);
  sqlite3_free(snapshot->levels);
  sqlite3_free(snapshot->deleted);
  sqlite3_free(snapshot->valid);
  sqlite3_free(snapshot->has_rowid);
  memset(snapshot, 0, sizeof(*snapshot));
}

static NodeCacheEntry *node_cache_find(HnswVtab *vtab, sqlite3_int64 node_id) {
  NodeCacheEntry *entry = NULL;
  size_t bucket = 0;
  if (vtab->cache_buckets == NULL) {
    return NULL;
  }
  bucket = node_cache_hash(vtab, node_id);
  entry = vtab->cache_buckets[bucket];
  while (entry != NULL && entry->node_id != node_id) {
    entry = entry->hash_next;
  }
  if (entry != NULL && entry != vtab->cache_head) {
    node_cache_unlink_lru(vtab, entry);
    node_cache_link_head(vtab, entry);
  }
  return entry;
}

static void node_cache_invalidate(HnswVtab *vtab, sqlite3_int64 node_id) {
  NodeCacheEntry *entry = node_cache_find(vtab, node_id);
  if (entry != NULL) {
    node_cache_remove(vtab, entry);
  }
}

static void node_cache_put(HnswVtab *vtab, sqlite3_int64 node_id,
                           const float *vector, double norm, int deleted,
                           int level) {
  NodeCacheEntry *entry = NULL;
  size_t vector_bytes = (size_t)vtab->dims * sizeof(float);
  size_t bytes = sizeof(NodeCacheEntry) + vector_bytes;
  size_t bucket = 0;
  if (vtab->cache_budget == 0 || bytes > vtab->cache_budget) {
    return;
  }
  if (vtab->cache_buckets == NULL) {
    size_t bucket_bytes = 4096U * sizeof(NodeCacheEntry *);
    vtab->cache_buckets = sqlite3_malloc64((sqlite3_uint64)bucket_bytes);
    if (vtab->cache_buckets == NULL) {
      return;
    }
    memset(vtab->cache_buckets, 0, bucket_bytes);
    vtab->cache_bucket_count = 4096U;
  }
  node_cache_invalidate(vtab, node_id);
  while (vtab->cache_tail != NULL &&
         vtab->cache_bytes + bytes > vtab->cache_budget) {
    NodeCacheEntry *victim = vtab->cache_tail;
    while (victim != NULL && victim->pins > 0) {
      victim = victim->lru_previous;
    }
    if (victim == NULL) {
      return;
    }
    node_cache_remove(vtab, victim);
  }
  if (vtab->cache_bytes + bytes > vtab->cache_budget) {
    return;
  }
  entry = sqlite3_malloc64(sizeof(*entry));
  if (entry == NULL) {
    return;
  }
  memset(entry, 0, sizeof(*entry));
  entry->vector = sqlite3_malloc64((sqlite3_uint64)vector_bytes);
  if (entry->vector == NULL) {
    sqlite3_free(entry);
    return;
  }
  memcpy(entry->vector, vector, vector_bytes);
  entry->node_id = node_id;
  entry->norm = norm;
  entry->deleted = deleted;
  entry->level = level;
  entry->bytes = bytes;
  bucket = node_cache_hash(vtab, node_id);
  entry->hash_next = vtab->cache_buckets[bucket];
  vtab->cache_buckets[bucket] = entry;
  node_cache_link_head(vtab, entry);
  vtab->cache_bytes += bytes;
}

static int declare_vtab(HnswVtab *vtab, char **error) {
  char *sql = sqlite3_mprintf(
      "CREATE TABLE x(\"%w\" BLOB NOT NULL,"
      "distance REAL HIDDEN,k INTEGER HIDDEN,ef_search INTEGER HIDDEN,"
      "visited_count INTEGER HIDDEN,command TEXT HIDDEN)",
      vtab->vector_name);
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_declare_vtab(vtab->db, sql);
  if (rc != SQLITE_OK && error != NULL) {
    *error = sqlite3_mprintf("failed to declare hnsw table: %s",
                             sqlite3_errmsg(vtab->db));
  }
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    rc = sqlite3_vtab_config(vtab->db, SQLITE_VTAB_CONSTRAINT_SUPPORT, 1);
  }
  return rc;
}

static void set_cache_budget(HnswVtab *vtab) {
  if ((sqlite3_uint64)vtab->cache_size_mb * 1024U * 1024U >
      (sqlite3_uint64)SIZE_MAX) {
    vtab->cache_budget = SIZE_MAX;
  } else {
    vtab->cache_budget = (size_t)vtab->cache_size_mb * 1024U * 1024U;
  }
}

static void free_vtab(HnswVtab *vtab) {
  if (vtab == NULL) {
    return;
  }
  finalize_cached_statements(vtab);
  cache_clear_all(vtab);
  query_snapshot_clear(vtab);
  sqlite3_free(vtab->schema);
  sqlite3_free(vtab->name);
  sqlite3_free(vtab->vector_name);
  sqlite3_free(vtab);
}

static int allocate_vtab(sqlite3 *db, int argc, const char *const *argv,
                         HnswVtab **out, char **error) {
  HnswVtab *vtab = sqlite3_malloc64(sizeof(*vtab));
  int rc = SQLITE_OK;
  if (vtab == NULL) {
    return SQLITE_NOMEM;
  }
  memset(vtab, 0, sizeof(*vtab));
  vtab->db = db;
  vtab->schema = sqlite3_mprintf("%s", argv[1]);
  vtab->name = sqlite3_mprintf("%s", argv[2]);
  if (vtab->schema == NULL || vtab->name == NULL) {
    free_vtab(vtab);
    return SQLITE_NOMEM;
  }
  rc = parse_arguments(argc, argv, vtab, error);
  if (rc != SQLITE_OK) {
    free_vtab(vtab);
    return rc;
  }
  vtab->cache_seq = -1;
  set_cache_budget(vtab);
  *out = vtab;
  return SQLITE_OK;
}

static int load_state(HnswVtab *vtab, HnswState *state) {
  char *table = shadow_table(vtab, "meta");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT entry_node,max_level,next_node_id,live_count,"
                        "tombstone_count,change_seq FROM %s WHERE id=1",
                        table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->read_state_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->read_state_stmt;
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    state->entry_node = sqlite3_column_int64(statement, 0);
    state->max_level = sqlite3_column_int(statement, 1);
    state->next_node_id = sqlite3_column_int64(statement, 2);
    state->live_count = sqlite3_column_int64(statement, 3);
    state->tombstone_count = sqlite3_column_int64(statement, 4);
    state->change_seq = sqlite3_column_int64(statement, 5);
    rc = SQLITE_OK;
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  return finish_cached_statement(statement, rc);
}

static int sync_node_cache(HnswVtab *vtab) {
  HnswState state;
  int rc = load_state(vtab, &state);
  if (rc == SQLITE_OK && state.change_seq != vtab->cache_seq) {
    cache_clear_all(vtab);
    query_snapshot_clear(vtab);
    vtab->cache_seq = state.change_seq;
  }
  return rc;
}

static int save_state(HnswVtab *vtab, const HnswState *state) {
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  char *table = NULL;
  char *sql = NULL;
  table = shadow_table(vtab, "meta");
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql =
      sqlite3_mprintf("UPDATE %s SET entry_node=?,max_level=?,next_node_id=?,"
                      "live_count=?,tombstone_count=?,change_seq=? WHERE id=1",
                      table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->write_state_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->write_state_stmt;
  sqlite3_bind_int64(statement, 1, state->entry_node);
  sqlite3_bind_int(statement, 2, state->max_level);
  sqlite3_bind_int64(statement, 3, state->next_node_id);
  sqlite3_bind_int64(statement, 4, state->live_count);
  sqlite3_bind_int64(statement, 5, state->tombstone_count);
  sqlite3_bind_int64(statement, 6, state->change_seq);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  return finish_cached_statement(statement, rc);
}

static uint64_t read_u64_le(const unsigned char *p) {
  uint64_t value = 0;
  int i = 0;
  for (i = 7; i >= 0; --i) {
    value = (value << 8U) | p[i];
  }
  return value;
}

static void write_u64_le(unsigned char *p, uint64_t value) {
  int i = 0;
  for (i = 0; i < 8; ++i) {
    p[i] = (unsigned char)(value & 0xffU);
    value >>= 8U;
  }
}

static uint32_t read_u32_le_local(const unsigned char *p) {
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8U) | ((uint32_t)p[2] << 16U) |
         ((uint32_t)p[3] << 24U);
}

static void write_u32_le_local(unsigned char *p, uint32_t value) {
  p[0] = (unsigned char)(value & 0xffU);
  p[1] = (unsigned char)((value >> 8U) & 0xffU);
  p[2] = (unsigned char)((value >> 16U) & 0xffU);
  p[3] = (unsigned char)((value >> 24U) & 0xffU);
}

static void free_id_list(IdList *list) {
  sqlite3_free(list->ids);
  list->ids = NULL;
  list->count = 0;
}

static int decode_id_list(const void *blob, int bytes, int maximum,
                          IdList *out) {
  const unsigned char *data = (const unsigned char *)blob;
  uint32_t count = 0;
  int i = 0;
  if (blob == NULL || bytes < 4) {
    return SQLITE_CORRUPT_VTAB;
  }
  count = read_u32_le_local(data);
  if (maximum < 0 || count > (uint32_t)maximum ||
      bytes != 4 + (int)((sqlite3_uint64)count * 8U)) {
    return SQLITE_CORRUPT_VTAB;
  }
  if (count == 0) {
    return SQLITE_OK;
  }
  out->ids = sqlite3_malloc64((sqlite3_uint64)count * sizeof(sqlite3_int64));
  if (out->ids == NULL) {
    return SQLITE_NOMEM;
  }
  out->count = (int)count;
  for (i = 0; i < out->count; ++i) {
    uint64_t raw = read_u64_le(data + 4U + (size_t)i * 8U);
    if (raw == 0 || raw > (uint64_t)INT64_MAX) {
      free_id_list(out);
      return SQLITE_CORRUPT_VTAB;
    }
    out->ids[i] = (sqlite3_int64)raw;
  }
  return SQLITE_OK;
}

static unsigned char *encode_id_list(const IdList *list, int *bytes) {
  unsigned char *blob = NULL;
  int i = 0;
  sqlite3_uint64 size = 4U + (sqlite3_uint64)list->count * 8U;
  if (list->count < 0 || size > HNSW_ALLOCATION_LIMIT ||
      size > (sqlite3_uint64)INT_MAX) {
    return NULL;
  }
  blob = sqlite3_malloc64(size);
  if (blob == NULL) {
    return NULL;
  }
  write_u32_le_local(blob, (uint32_t)list->count);
  for (i = 0; i < list->count; ++i) {
    write_u64_le(blob + 4U + (size_t)i * 8U, (uint64_t)list->ids[i]);
  }
  *bytes = (int)size;
  return blob;
}

static int snapshot_add_array(sqlite3_uint64 *total, sqlite3_uint64 count,
                              sqlite3_uint64 item_size) {
  sqlite3_uint64 bytes = 0;
  if (item_size != 0 && count > HNSW_ALLOCATION_LIMIT / item_size) {
    return SQLITE_TOOBIG;
  }
  bytes = count * item_size;
  if (*total > UINT64_MAX - bytes) {
    return SQLITE_TOOBIG;
  }
  *total += bytes;
  return SQLITE_OK;
}

static int query_snapshot_required_bytes(size_t node_capacity, int dims,
                                         size_t neighbor_slots,
                                         size_t count_slots,
                                         sqlite3_uint64 *required) {
  sqlite3_uint64 total = 0;
  int rc = SQLITE_OK;
#define SNAPSHOT_ARRAY(target, count, type)                                    \
  do {                                                                         \
    rc = snapshot_add_array(&(target), (sqlite3_uint64)(count), sizeof(type)); \
    if (rc != SQLITE_OK) {                                                      \
      return rc;                                                               \
    }                                                                          \
  } while (0)
  if (dims <= 0 || (sqlite3_uint64)node_capacity >
                       HNSW_ALLOCATION_LIMIT /
                           ((sqlite3_uint64)dims * sizeof(float))) {
    return SQLITE_TOOBIG;
  }
  SNAPSHOT_ARRAY(total, (sqlite3_uint64)node_capacity * (sqlite3_uint64)dims,
                 float);
  SNAPSHOT_ARRAY(total, node_capacity, double);
  SNAPSHOT_ARRAY(total, node_capacity, sqlite3_int64);
  SNAPSHOT_ARRAY(total, node_capacity + 1U, size_t);
  SNAPSHOT_ARRAY(total, node_capacity + 1U, size_t);
  SNAPSHOT_ARRAY(total, neighbor_slots, uint32_t);
  SNAPSHOT_ARRAY(total, count_slots, uint16_t);
  SNAPSHOT_ARRAY(total, node_capacity, unsigned char);
  SNAPSHOT_ARRAY(total, node_capacity, unsigned char);
  SNAPSHOT_ARRAY(total, node_capacity, unsigned char);
  SNAPSHOT_ARRAY(total, node_capacity, unsigned char);
  SNAPSHOT_ARRAY(total, node_capacity, uint32_t);
#undef SNAPSHOT_ARRAY
  *required = total;
  return SQLITE_OK;
}

static void *snapshot_allocate(sqlite3_uint64 count, sqlite3_uint64 item_size,
                               int zero) {
  sqlite3_uint64 bytes = count * item_size;
  void *memory = NULL;
  if (bytes == 0) {
    return NULL;
  }
  memory = sqlite3_malloc64(bytes);
  if (memory != NULL && zero) {
    memset(memory, 0, (size_t)bytes);
  }
  return memory;
}

static int query_snapshot_estimate(HnswVtab *vtab, const HnswState *state,
                                   size_t *node_capacity,
                                   size_t *neighbor_slots,
                                   size_t *count_slots,
                                   sqlite3_uint64 *required, int *pending) {
  char *nodes = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  sqlite3_int64 rows = 0;
  sqlite3_int64 maximum_id = 0;
  sqlite3_int64 linked = 0;
  sqlite3_int64 level_sum = 0;
  sqlite3_uint64 slots = 0;
  sqlite3_uint64 counts = 0;
  int rc = SQLITE_OK;
  if (nodes == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT count(*),coalesce(max(node_id),0),"
      "coalesce(sum(CASE WHEN level>=0 THEN 1 ELSE 0 END),0),"
      "coalesce(sum(CASE WHEN level>=0 THEN level ELSE 0 END),0),"
      "coalesce(sum(CASE WHEN level<=-2 AND deleted=0 THEN 1 ELSE 0 END),0) "
      "FROM %s",
      nodes);
  sqlite3_free(nodes);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    rows = sqlite3_column_int64(statement, 0);
    maximum_id = sqlite3_column_int64(statement, 1);
    linked = sqlite3_column_int64(statement, 2);
    level_sum = sqlite3_column_int64(statement, 3);
    *pending = sqlite3_column_int(statement, 4);
    rc = SQLITE_OK;
  } else {
    rc = rc == SQLITE_DONE ? SQLITE_CORRUPT_VTAB : rc;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (state->next_node_id <= 0 || maximum_id >= state->next_node_id ||
      rows < 0 || linked < 0 || level_sum < 0 ||
      (uint64_t)(state->next_node_id - 1) > UINT32_MAX) {
    return SQLITE_TOOBIG;
  }
  *node_capacity = (size_t)(state->next_node_id - 1);
  slots = (sqlite3_uint64)linked * (sqlite3_uint64)(vtab->m * 2) +
          (sqlite3_uint64)level_sum * (sqlite3_uint64)vtab->m;
  counts = (sqlite3_uint64)linked + (sqlite3_uint64)level_sum;
  if (slots > SIZE_MAX || counts > SIZE_MAX) {
    return SQLITE_TOOBIG;
  }
  *neighbor_slots = (size_t)slots;
  *count_slots = (size_t)counts;
  return query_snapshot_required_bytes(*node_capacity, vtab->dims,
                                       *neighbor_slots, *count_slots,
                                       required);
}

static int query_snapshot_load_nodes(HnswVtab *vtab,
                                     HnswQuerySnapshot *snapshot) {
  char *nodes = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  size_t next_index = 0;
  size_t neighbor_offset = 0;
  size_t count_offset = 0;
  int rc = SQLITE_OK;
  if (nodes == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT node_id,level,vector,norm,deleted FROM %s ORDER BY node_id",
      nodes);
  sqlite3_free(nodes);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 node_id = sqlite3_column_int64(statement, 0);
    size_t index = 0;
    int level = sqlite3_column_int(statement, 1);
    int deleted = sqlite3_column_int(statement, 4);
    double stored_norm = sqlite3_column_double(statement, 3);
    char *error = NULL;
    if (node_id <= 0 || (uint64_t)node_id > snapshot->node_capacity ||
        level < 0 || level > HNSW_MAX_LEVEL ||
        (deleted != 0 && deleted != 1)) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    index = (size_t)(node_id - 1);
    if (index < next_index) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    while (next_index <= index) {
      snapshot->neighbor_offsets[next_index] = neighbor_offset;
      snapshot->count_offsets[next_index] = count_offset;
      ++next_index;
    }
    rc = hnsw_vector_validate(sqlite3_column_blob(statement, 2),
                              sqlite3_column_bytes(statement, 2), vtab->dims,
                              vtab->metric, &error);
    if (rc == SQLITE_OK) {
      rc = hnsw_vector_decode_into(
          sqlite3_column_blob(statement, 2),
          sqlite3_column_bytes(statement, 2),
          snapshot->vectors + index * (size_t)vtab->dims, vtab->dims, &error);
    }
    sqlite3_free(error);
    if (rc != SQLITE_OK) {
      if (rc != SQLITE_NOMEM) {
        rc = SQLITE_CORRUPT_VTAB;
      }
      break;
    }
    {
      double computed = hnsw_vector_norm(
          snapshot->vectors + index * (size_t)vtab->dims, vtab->dims);
      double tolerance = fmax(1e-12, fabs(computed) * 1e-12);
      if (!isfinite(stored_norm) || stored_norm < 0.0 ||
          fabs(stored_norm - computed) > tolerance) {
        rc = SQLITE_CORRUPT_VTAB;
        break;
      }
    }
    snapshot->norms[index] = stored_norm;
    snapshot->levels[index] = (unsigned char)level;
    snapshot->deleted[index] = (unsigned char)deleted;
    snapshot->valid[index] = 1;
    neighbor_offset += (size_t)vtab->m * (size_t)(level + 2);
    count_offset += (size_t)level + 1U;
    if (sqlite3_is_interrupted(vtab->db)) {
      rc = SQLITE_INTERRUPT;
      break;
    }
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  while (rc == SQLITE_OK && next_index <= snapshot->node_capacity) {
    snapshot->neighbor_offsets[next_index] = neighbor_offset;
    snapshot->count_offsets[next_index] = count_offset;
    ++next_index;
  }
  if (rc == SQLITE_OK &&
      (neighbor_offset != snapshot->neighbor_slots ||
       count_offset != snapshot->count_slots)) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  return rc;
}

static int query_snapshot_load_rowids(HnswVtab *vtab,
                                      HnswQuerySnapshot *snapshot) {
  char *rows = shadow_table(vtab, "rows");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (rows == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT rowid,node_id FROM %s", rows);
  sqlite3_free(rows);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 node_id = sqlite3_column_int64(statement, 1);
    size_t index = 0;
    if (node_id <= 0 || (uint64_t)node_id > snapshot->node_capacity) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    index = (size_t)(node_id - 1);
    if (!snapshot->valid[index] || snapshot->deleted[index]) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    snapshot->rowids[index] = sqlite3_column_int64(statement, 0);
    snapshot->has_rowid[index] = 1;
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  if (rc == SQLITE_OK) {
    size_t index = 0;
    for (index = 0; index < snapshot->node_capacity; ++index) {
      if (snapshot->valid[index] && !snapshot->deleted[index] &&
          !snapshot->has_rowid[index]) {
        return SQLITE_CORRUPT_VTAB;
      }
    }
  }
  return rc;
}

static int query_snapshot_load_edges(HnswVtab *vtab,
                                     HnswQuerySnapshot *snapshot) {
  char *edges = shadow_table(vtab, "edges");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  size_t loaded_rows = 0;
  int rc = SQLITE_OK;
  if (edges == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT node_id,layer,neighbors FROM %s ORDER BY node_id,layer", edges);
  sqlite3_free(edges);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 node_id = sqlite3_column_int64(statement, 0);
    int layer = sqlite3_column_int(statement, 1);
    const unsigned char *data = sqlite3_column_blob(statement, 2);
    int bytes = sqlite3_column_bytes(statement, 2);
    size_t index = 0;
    size_t slot = 0;
    size_t count_index = 0;
    int capacity = 0;
    uint32_t count = 0;
    uint32_t i = 0;
    if (node_id <= 0 || (uint64_t)node_id > snapshot->node_capacity) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    index = (size_t)(node_id - 1);
    if (!snapshot->valid[index] || layer < 0 ||
        layer > (int)snapshot->levels[index] || data == NULL || bytes < 4) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    capacity = layer == 0 ? vtab->m * 2 : vtab->m;
    slot = snapshot->neighbor_offsets[index] +
           (layer == 0 ? 0U
                       : (size_t)vtab->m * (size_t)(layer + 1));
    count_index = snapshot->count_offsets[index] + (size_t)layer;
    count = read_u32_le_local(data);
    if (count > (uint32_t)capacity ||
        bytes != 4 + (int)((sqlite3_uint64)count * 8U) ||
        count_index >= snapshot->count_slots ||
        slot + count > snapshot->neighbor_slots) {
      rc = SQLITE_CORRUPT_VTAB;
      break;
    }
    snapshot->neighbor_counts[count_index] = (uint16_t)count;
    for (i = 0; i < count; ++i) {
      uint64_t raw = read_u64_le(data + 4U + (size_t)i * 8U);
      uint32_t j = 0;
      if (raw == 0 || raw > snapshot->node_capacity ||
          !snapshot->valid[(size_t)raw - 1U] ||
          snapshot->levels[(size_t)raw - 1U] < (unsigned char)layer ||
          raw == (uint64_t)node_id) {
        rc = SQLITE_CORRUPT_VTAB;
        break;
      }
      for (j = 0; j < i; ++j) {
        if (snapshot->neighbors[slot + j] == (uint32_t)raw) {
          rc = SQLITE_CORRUPT_VTAB;
          break;
        }
      }
      if (rc != SQLITE_ROW && rc != SQLITE_OK) {
        break;
      }
      snapshot->neighbors[slot + i] = (uint32_t)raw;
    }
    if (rc != SQLITE_ROW && rc != SQLITE_OK) {
      break;
    }
    ++loaded_rows;
    if (sqlite3_is_interrupted(vtab->db)) {
      rc = SQLITE_INTERRUPT;
      break;
    }
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  if (rc == SQLITE_OK && loaded_rows != snapshot->count_slots) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  return rc;
}

static int query_snapshot_ensure(HnswVtab *vtab) {
  HnswQuerySnapshot *snapshot = &vtab->snapshot;
  HnswState state;
  sqlite3_uint64 required = 0;
  size_t node_capacity = 0;
  size_t neighbor_slots = 0;
  size_t count_slots = 0;
  int pending = 0;
  int rc = SQLITE_OK;
  if (vtab->write_dirty || vtab->cache_budget == 0) {
    return SQLITE_OK;
  }
  rc = load_state(vtab, &state);
  if (rc != SQLITE_OK || state.live_count == 0) {
    return rc;
  }
  if (snapshot->attempted && snapshot->change_seq == state.change_seq) {
    return SQLITE_OK;
  }
  query_snapshot_clear(vtab);
  snapshot->attempted = 1;
  snapshot->change_seq = state.change_seq;
  if (state.next_node_id <= 0 ||
      (uint64_t)(state.next_node_id - 1) > UINT32_MAX ||
      (sqlite3_uint64)(state.next_node_id - 1) >
          HNSW_ALLOCATION_LIMIT /
              ((sqlite3_uint64)vtab->dims * sizeof(float))) {
    return SQLITE_OK;
  }
  snapshot->estimated_bytes =
      (sqlite3_uint64)(state.next_node_id - 1) *
      (sqlite3_uint64)vtab->dims * sizeof(float);
  if (snapshot->estimated_bytes > vtab->cache_budget) {
    return SQLITE_OK;
  }
  rc = query_snapshot_estimate(vtab, &state, &node_capacity, &neighbor_slots,
                               &count_slots, &required, &pending);
  snapshot->estimated_bytes = required;
  if (rc == SQLITE_TOOBIG || pending != 0 || required > vtab->cache_budget) {
    return SQLITE_OK;
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  snapshot->node_capacity = node_capacity;
  snapshot->neighbor_slots = neighbor_slots;
  snapshot->count_slots = count_slots;
  cache_clear_all(vtab);
#define SNAPSHOT_ALLOC(member, count, type, zero)                              \
  do {                                                                         \
    snapshot->member = snapshot_allocate((sqlite3_uint64)(count), sizeof(type),\
                                         (zero));                              \
    if ((count) != 0 && snapshot->member == NULL) {                            \
      rc = SQLITE_NOMEM;                                                       \
      goto failed;                                                             \
    }                                                                          \
  } while (0)
  SNAPSHOT_ALLOC(vectors, node_capacity * (size_t)vtab->dims, float, 0);
  SNAPSHOT_ALLOC(norms, node_capacity, double, 0);
  SNAPSHOT_ALLOC(rowids, node_capacity, sqlite3_int64, 0);
  SNAPSHOT_ALLOC(neighbor_offsets, node_capacity + 1U, size_t, 0);
  SNAPSHOT_ALLOC(count_offsets, node_capacity + 1U, size_t, 0);
  SNAPSHOT_ALLOC(neighbors, neighbor_slots, uint32_t, 0);
  SNAPSHOT_ALLOC(neighbor_counts, count_slots, uint16_t, 1);
  SNAPSHOT_ALLOC(levels, node_capacity, unsigned char, 1);
  SNAPSHOT_ALLOC(deleted, node_capacity, unsigned char, 1);
  SNAPSHOT_ALLOC(valid, node_capacity, unsigned char, 1);
  SNAPSHOT_ALLOC(has_rowid, node_capacity, unsigned char, 1);
#undef SNAPSHOT_ALLOC
  rc = query_snapshot_load_nodes(vtab, snapshot);
  if (rc == SQLITE_OK) {
    rc = query_snapshot_load_rowids(vtab, snapshot);
  }
  if (rc == SQLITE_OK) {
    rc = query_snapshot_load_edges(vtab, snapshot);
  }
  if (rc == SQLITE_OK) {
    snapshot->ready = 1;
    return SQLITE_OK;
  }
failed:
  query_snapshot_clear(vtab);
  return rc;
}

static int load_edges_storage(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                              IdList *out) {
  char *table = shadow_table(vtab, "edges");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  memset(out, 0, sizeof(*out));
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT neighbors FROM %s WHERE node_id=? AND layer=?",
                        table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->read_edges_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->read_edges_stmt;
  sqlite3_bind_int64(statement, 1, node_id);
  sqlite3_bind_int(statement, 2, layer);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    int maximum = layer == 0 ? vtab->m * 2 : vtab->m;
    int i = 0;
    rc = decode_id_list(sqlite3_column_blob(statement, 0),
                        sqlite3_column_bytes(statement, 0), maximum, out);
    for (i = 0; i < out->count && rc == SQLITE_OK; ++i) {
      int j = 0;
      if (out->ids[i] == node_id) {
        rc = SQLITE_CORRUPT_VTAB;
        break;
      }
      for (j = 0; j < i; ++j) {
        if (out->ids[j] == out->ids[i]) {
          rc = SQLITE_CORRUPT_VTAB;
          break;
        }
      }
    }
    if (rc != SQLITE_OK) {
      free_id_list(out);
    }
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  return finish_cached_statement(statement, rc);
}

static int write_edges_storage(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                               const IdList *list) {
  char *table = shadow_table(vtab, "edges");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  unsigned char *blob = NULL;
  int bytes = 0;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  blob = encode_id_list(list, &bytes);
  if (blob == NULL) {
    sqlite3_free(table);
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "INSERT INTO %s(node_id,layer,neighbors) VALUES(?,?,?) "
      "ON CONFLICT(node_id,layer) DO UPDATE SET neighbors=excluded.neighbors",
      table);
  sqlite3_free(table);
  if (sql == NULL) {
    sqlite3_free(blob);
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->write_edges_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    sqlite3_free(blob);
    return rc;
  }
  statement = vtab->write_edges_stmt;
  sqlite3_bind_int64(statement, 1, node_id);
  sqlite3_bind_int(statement, 2, layer);
  sqlite3_bind_blob(statement, 3, blob, bytes, sqlite3_free);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  return finish_cached_statement(statement, rc);
}

static int edge_cache_make_room(HnswVtab *vtab, size_t bytes) {
  while (vtab->cache_bytes + bytes > vtab->cache_budget) {
    EdgeCacheEntry *edge = vtab->edge_cache_tail;
    NodeCacheEntry *node = NULL;
    while (edge != NULL && edge->pins > 0) {
      edge = edge->lru_previous;
    }
    if (edge != NULL) {
      int rc = SQLITE_OK;
      if (edge->dirty) {
        IdList list = {edge->ids, edge->count};
        rc = write_edges_storage(vtab, edge->node_id, edge->layer, &list);
      }
      if (rc != SQLITE_OK) {
        return rc;
      }
      edge_cache_remove(vtab, edge);
      continue;
    }
    node = vtab->cache_tail;
    while (node != NULL && node->pins > 0) {
      node = node->lru_previous;
    }
    if (node == NULL) {
      return SQLITE_FULL;
    }
    node_cache_remove(vtab, node);
  }
  return SQLITE_OK;
}

static int edge_cache_store(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                            const IdList *list, int dirty, int *stored) {
  EdgeCacheEntry *entry = NULL;
  size_t list_bytes = (size_t)list->count * sizeof(sqlite3_int64);
  size_t bytes = sizeof(EdgeCacheEntry) + list_bytes;
  size_t bucket = 0;
  int rc = SQLITE_OK;
  *stored = 0;
  if (vtab->cache_budget == 0 || bytes > vtab->cache_budget) {
    return SQLITE_OK;
  }
  if (vtab->edge_cache_buckets == NULL) {
    size_t bucket_bytes = 4096U * sizeof(EdgeCacheEntry *);
    vtab->edge_cache_buckets = sqlite3_malloc64((sqlite3_uint64)bucket_bytes);
    if (vtab->edge_cache_buckets == NULL) {
      return SQLITE_OK;
    }
    memset(vtab->edge_cache_buckets, 0, bucket_bytes);
    vtab->edge_cache_bucket_count = 4096U;
  }
  entry = edge_cache_find(vtab, node_id, layer);
  if (entry != NULL) {
    if (entry->pins > 0) {
      return SQLITE_BUSY;
    }
    if (entry->dirty && !dirty) {
      IdList old = {entry->ids, entry->count};
      rc = write_edges_storage(vtab, entry->node_id, entry->layer, &old);
      if (rc != SQLITE_OK) {
        return rc;
      }
    }
    edge_cache_remove(vtab, entry);
  }
  rc = edge_cache_make_room(vtab, bytes);
  if (rc == SQLITE_FULL) {
    return SQLITE_OK;
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  entry = sqlite3_malloc64(sizeof(*entry));
  if (entry == NULL) {
    return SQLITE_OK;
  }
  memset(entry, 0, sizeof(*entry));
  if (list->count > 0) {
    entry->ids = sqlite3_malloc64((sqlite3_uint64)list_bytes);
    if (entry->ids == NULL) {
      sqlite3_free(entry);
      return SQLITE_OK;
    }
    memcpy(entry->ids, list->ids, list_bytes);
  }
  entry->node_id = node_id;
  entry->layer = layer;
  entry->count = list->count;
  entry->dirty = dirty;
  entry->bytes = bytes;
  bucket = edge_cache_hash(vtab, node_id, layer);
  entry->hash_next = vtab->edge_cache_buckets[bucket];
  vtab->edge_cache_buckets[bucket] = entry;
  edge_cache_link_head(vtab, entry);
  vtab->cache_bytes += bytes;
  *stored = 1;
  return SQLITE_OK;
}

static int acquire_edges(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                         EdgeView *view) {
  EdgeCacheEntry *entry = NULL;
  memset(view, 0, sizeof(*view));
  if (vtab->snapshot.ready && !vtab->write_dirty) {
    HnswQuerySnapshot *snapshot = &vtab->snapshot;
    size_t index = 0;
    size_t slot = 0;
    size_t count_index = 0;
    if (node_id <= 0 || (uint64_t)node_id > snapshot->node_capacity) {
      return SQLITE_CORRUPT_VTAB;
    }
    index = (size_t)(node_id - 1);
    if (!snapshot->valid[index] || layer < 0 ||
        layer > (int)snapshot->levels[index]) {
      return SQLITE_CORRUPT_VTAB;
    }
    slot = snapshot->neighbor_offsets[index] +
           (layer == 0 ? 0U
                       : (size_t)vtab->m * (size_t)(layer + 1));
    count_index = snapshot->count_offsets[index] + (size_t)layer;
    if (count_index >= snapshot->count_slots) {
      return SQLITE_CORRUPT_VTAB;
    }
    view->compact_ids = snapshot->neighbors + slot;
    view->count = (int)snapshot->neighbor_counts[count_index];
    return SQLITE_OK;
  }
  entry = edge_cache_find(vtab, node_id, layer);
  if (entry != NULL) {
    ++entry->pins;
    view->ids = entry->ids;
    view->count = entry->count;
    view->cached = entry;
    return SQLITE_OK;
  }
  {
    IdList loaded = {0};
    int stored = 0;
    int rc = load_edges_storage(vtab, node_id, layer, &loaded);
    if (rc != SQLITE_OK) {
      return rc;
    }
    rc = edge_cache_store(vtab, node_id, layer, &loaded, 0, &stored);
    if (rc != SQLITE_OK) {
      free_id_list(&loaded);
      return rc;
    }
    if (stored) {
      entry = edge_cache_find(vtab, node_id, layer);
      ++entry->pins;
      view->ids = entry->ids;
      view->count = entry->count;
      view->cached = entry;
      free_id_list(&loaded);
    } else {
      view->ids = loaded.ids;
      view->count = loaded.count;
      view->owned = loaded.ids;
    }
  }
  return SQLITE_OK;
}

static sqlite3_int64 edge_view_id(const EdgeView *view, int index) {
  return view->compact_ids != NULL ? (sqlite3_int64)view->compact_ids[index]
                                   : view->ids[index];
}

static void release_edges(EdgeView *view) {
  if (view->cached != NULL) {
    --view->cached->pins;
  }
  sqlite3_free(view->owned);
  memset(view, 0, sizeof(*view));
}

static int fetch_edges(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                       IdList *out) {
  EdgeView view = {0};
  int rc = acquire_edges(vtab, node_id, layer, &view);
  memset(out, 0, sizeof(*out));
  if (rc == SQLITE_OK && view.count > 0) {
    out->ids = sqlite3_malloc64((sqlite3_uint64)view.count * sizeof(*out->ids));
    if (out->ids == NULL) {
      rc = SQLITE_NOMEM;
    } else {
      int i = 0;
      for (i = 0; i < view.count; ++i) {
        out->ids[i] = edge_view_id(&view, i);
      }
      out->count = view.count;
    }
  }
  release_edges(&view);
  return rc;
}

static int write_edges(HnswVtab *vtab, sqlite3_int64 node_id, int layer,
                       const IdList *list) {
  int stored = 0;
  int rc = SQLITE_OK;
  if (!vtab->defer_edge_writes) {
    rc = write_edges_storage(vtab, node_id, layer, list);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  rc = edge_cache_store(vtab, node_id, layer, list, vtab->defer_edge_writes,
                        &stored);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (vtab->defer_edge_writes && !stored) {
    return write_edges_storage(vtab, node_id, layer, list);
  }
  return SQLITE_OK;
}

static int flush_edge_cache(HnswVtab *vtab) {
  EdgeCacheEntry *entry = vtab->edge_cache_head;
  while (entry != NULL) {
    if (entry->dirty) {
      IdList list = {entry->ids, entry->count};
      int rc = write_edges_storage(vtab, entry->node_id, entry->layer, &list);
      if (rc != SQLITE_OK) {
        return rc;
      }
      entry->dirty = 0;
    }
    entry = entry->lru_next;
  }
  return SQLITE_OK;
}

static int fetch_node(HnswVtab *vtab, sqlite3_int64 node_id, float **vector,
                      double *norm, int *deleted, int *level) {
  char *table = NULL;
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  char *error = NULL;
  int dims = 0;
  int rc = SQLITE_OK;
  NodeCacheEntry *cached = node_cache_find(vtab, node_id);
  *vector = NULL;
  if (cached != NULL) {
    *vector = sqlite3_malloc64((sqlite3_uint64)vtab->dims * sizeof(float));
    if (*vector == NULL) {
      return SQLITE_NOMEM;
    }
    memcpy(*vector, cached->vector, (size_t)vtab->dims * sizeof(float));
    *norm = cached->norm;
    *deleted = cached->deleted;
    *level = cached->level;
    return SQLITE_OK;
  }
  table = shadow_table(vtab, "nodes");
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT vector,norm,deleted,level FROM %s WHERE node_id=?", table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->read_node_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->read_node_stmt;
  sqlite3_bind_int64(statement, 1, node_id);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    const void *blob = sqlite3_column_blob(statement, 0);
    int bytes = sqlite3_column_bytes(statement, 0);
    rc = hnsw_vector_validate(blob, bytes, vtab->dims, vtab->metric, &error);
    if (rc == SQLITE_OK) {
      rc = hnsw_vector_decode(blob, bytes, vector, &dims, &error);
    }
    if (rc == SQLITE_OK && dims != vtab->dims) {
      rc = SQLITE_CORRUPT_VTAB;
    }
    if (rc != SQLITE_OK && rc != SQLITE_NOMEM) {
      rc = SQLITE_CORRUPT_VTAB;
    }
    if (rc == SQLITE_OK) {
      *norm = sqlite3_column_double(statement, 1);
      *deleted = sqlite3_column_int(statement, 2);
      *level = sqlite3_column_int(statement, 3);
      {
        double computed = hnsw_vector_norm(*vector, dims);
        double tolerance = fmax(1e-12, fabs(computed) * 1e-12);
        if (!isfinite(*norm) || *norm < 0.0 ||
            fabs(*norm - computed) > tolerance ||
            (*deleted != 0 && *deleted != 1) ||
            !((*level >= 0 && *level <= HNSW_MAX_LEVEL) ||
              (*level <= -2 && *level >= -HNSW_MAX_LEVEL - 2))) {
          rc = SQLITE_CORRUPT_VTAB;
        }
      }
      if (rc == SQLITE_OK) {
        node_cache_put(vtab, node_id, *vector, *norm, *deleted, *level);
      }
    }
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  sqlite3_free(error);
  return finish_cached_statement(statement, rc);
}

static int acquire_node(HnswVtab *vtab, sqlite3_int64 node_id, NodeView *view) {
  NodeCacheEntry *cached = NULL;
  memset(view, 0, sizeof(*view));
  if (vtab->snapshot.ready && !vtab->write_dirty) {
    HnswQuerySnapshot *snapshot = &vtab->snapshot;
    size_t index = 0;
    if (node_id <= 0 || (uint64_t)node_id > snapshot->node_capacity) {
      return SQLITE_CORRUPT_VTAB;
    }
    index = (size_t)(node_id - 1);
    if (!snapshot->valid[index]) {
      return SQLITE_CORRUPT_VTAB;
    }
    view->vector = snapshot->vectors + index * (size_t)vtab->dims;
    view->norm = snapshot->norms[index];
    view->deleted = snapshot->deleted[index];
    view->level = snapshot->levels[index];
    return SQLITE_OK;
  }
  cached = node_cache_find(vtab, node_id);
  if (cached != NULL) {
    ++cached->pins;
    view->vector = cached->vector;
    view->norm = cached->norm;
    view->deleted = cached->deleted;
    view->level = cached->level;
    view->cached = cached;
    return SQLITE_OK;
  }
  {
    float *decoded = NULL;
    int rc = fetch_node(vtab, node_id, &decoded, &view->norm, &view->deleted,
                        &view->level);
    if (rc != SQLITE_OK) {
      sqlite3_free(decoded);
      return rc;
    }
    cached = node_cache_find(vtab, node_id);
    if (cached != NULL) {
      ++cached->pins;
      view->vector = cached->vector;
      view->norm = cached->norm;
      view->deleted = cached->deleted;
      view->level = cached->level;
      view->cached = cached;
      sqlite3_free(decoded);
    } else {
      view->vector = decoded;
      view->owned = decoded;
    }
  }
  return SQLITE_OK;
}

static void release_node(NodeView *view) {
  if (view->cached != NULL) {
    --view->cached->pins;
  }
  sqlite3_free(view->owned);
  memset(view, 0, sizeof(*view));
}

static int node_distance(HnswVtab *vtab, const float *query, double query_norm,
                         sqlite3_int64 node_id, double *distance, int *deleted,
                         int *level, int *visited) {
  NodeView view;
  int rc = acquire_node(vtab, node_id, &view);
  if (rc != SQLITE_OK) {
    return rc;
  }
  *distance = hnsw_vector_distance(query, query_norm, view.vector, view.norm,
                                   vtab->dims, vtab->metric);
  *deleted = view.deleted;
  if (level != NULL) {
    *level = view.level;
  }
  release_node(&view);
  if (visited != NULL) {
    ++*visited;
  }
  return SQLITE_OK;
}

static int distance_between_nodes(HnswVtab *vtab, sqlite3_int64 left,
                                  sqlite3_int64 right, double *distance) {
  NodeView a = {0};
  NodeView b = {0};
  int rc = acquire_node(vtab, left, &a);
  if (rc == SQLITE_OK) {
    rc = acquire_node(vtab, right, &b);
  }
  if (rc == SQLITE_OK) {
    *distance = hnsw_vector_distance(a.vector, a.norm, b.vector, b.norm,
                                     vtab->dims, vtab->metric);
  }
  release_node(&a);
  release_node(&b);
  return rc;
}

static int fetch_rowid_for_node(HnswVtab *vtab, sqlite3_int64 node_id,
                                sqlite3_int64 *rowid) {
  char *table = NULL;
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (vtab->snapshot.ready && !vtab->write_dirty) {
    size_t index = 0;
    if (node_id <= 0 || (uint64_t)node_id > vtab->snapshot.node_capacity) {
      return SQLITE_CORRUPT_VTAB;
    }
    index = (size_t)(node_id - 1);
    if (!vtab->snapshot.valid[index]) {
      return SQLITE_CORRUPT_VTAB;
    }
    if (!vtab->snapshot.has_rowid[index]) {
      return SQLITE_CORRUPT_VTAB;
    }
    *rowid = vtab->snapshot.rowids[index];
    return SQLITE_OK;
  }
  table = shadow_table(vtab, "rows");
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT rowid FROM %s WHERE node_id=?", table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->read_rowid_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->read_rowid_stmt;
  sqlite3_bind_int64(statement, 1, node_id);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    *rowid = sqlite3_column_int64(statement, 0);
    rc = SQLITE_OK;
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  return finish_cached_statement(statement, rc);
}

static int heap_better(const Heap *heap, HeapItem a, HeapItem b) {
  if (a.distance == b.distance) {
    return heap->is_min ? a.id < b.id : a.id > b.id;
  }
  return heap->is_min ? a.distance < b.distance : a.distance > b.distance;
}

static void heap_free(Heap *heap) {
  sqlite3_free(heap->items);
  memset(heap, 0, sizeof(*heap));
}

static int heap_push(Heap *heap, HeapItem item) {
  int index = 0;
  if (heap->count == heap->capacity) {
    int next = heap->capacity == 0 ? 32 : heap->capacity * 2;
    HeapItem *grown = NULL;
    if (next < heap->capacity ||
        (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(HeapItem)) {
      return SQLITE_TOOBIG;
    }
    grown =
        sqlite3_realloc64(heap->items, (sqlite3_uint64)next * sizeof(HeapItem));
    if (grown == NULL) {
      return SQLITE_NOMEM;
    }
    heap->items = grown;
    heap->capacity = next;
  }
  index = heap->count++;
  heap->items[index] = item;
  while (index > 0) {
    int parent = (index - 1) / 2;
    HeapItem swap;
    if (!heap_better(heap, heap->items[index], heap->items[parent])) {
      break;
    }
    swap = heap->items[index];
    heap->items[index] = heap->items[parent];
    heap->items[parent] = swap;
    index = parent;
  }
  return SQLITE_OK;
}

static HeapItem heap_pop(Heap *heap) {
  HeapItem result = heap->items[0];
  int index = 0;
  --heap->count;
  if (heap->count == 0) {
    return result;
  }
  heap->items[0] = heap->items[heap->count];
  for (;;) {
    int left = index * 2 + 1;
    int right = left + 1;
    int best = index;
    HeapItem swap;
    if (left < heap->count &&
        heap_better(heap, heap->items[left], heap->items[best])) {
      best = left;
    }
    if (right < heap->count &&
        heap_better(heap, heap->items[right], heap->items[best])) {
      best = right;
    }
    if (best == index) {
      break;
    }
    swap = heap->items[index];
    heap->items[index] = heap->items[best];
    heap->items[best] = swap;
    index = best;
  }
  return result;
}

static uint64_t hash_id(uint64_t value) {
  value ^= value >> 30U;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27U;
  value *= UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31U);
}

static void id_set_free(IdSet *set) {
  sqlite3_free(set->slots);
  memset(set, 0, sizeof(*set));
}

static int id_set_resize(IdSet *set, size_t requested) {
  sqlite3_int64 *slots = NULL;
  size_t capacity = 16;
  size_t i = 0;
  while (capacity < requested) {
    if (capacity > SIZE_MAX / 2U) {
      return SQLITE_TOOBIG;
    }
    capacity *= 2U;
  }
  if ((sqlite3_uint64)capacity >
      HNSW_ALLOCATION_LIMIT / sizeof(sqlite3_int64)) {
    return SQLITE_TOOBIG;
  }
  slots = sqlite3_malloc64((sqlite3_uint64)capacity * sizeof(sqlite3_int64));
  if (slots == NULL) {
    return SQLITE_NOMEM;
  }
  for (i = 0; i < capacity; ++i) {
    slots[i] = 0;
  }
  if (set->slots != NULL) {
    for (i = 0; i < set->capacity; ++i) {
      sqlite3_int64 id = set->slots[i];
      if (id != 0) {
        size_t slot = (size_t)(hash_id((uint64_t)id) & (capacity - 1U));
        while (slots[slot] != 0) {
          slot = (slot + 1U) & (capacity - 1U);
        }
        slots[slot] = id;
      }
    }
    sqlite3_free(set->slots);
  }
  set->slots = slots;
  set->capacity = capacity;
  return SQLITE_OK;
}

static int id_set_add(IdSet *set, sqlite3_int64 id, int *added) {
  size_t slot = 0;
  int rc = SQLITE_OK;
  if (id <= 0) {
    return SQLITE_CORRUPT_VTAB;
  }
  if (set->capacity == 0 || (set->count + 1U) * 10U > set->capacity * 7U) {
    rc = id_set_resize(set, set->capacity == 0 ? 64U : set->capacity * 2U);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  slot = (size_t)(hash_id((uint64_t)id) & (set->capacity - 1U));
  while (set->slots[slot] != 0 && set->slots[slot] != id) {
    slot = (slot + 1U) & (set->capacity - 1U);
  }
  if (set->slots[slot] == id) {
    *added = 0;
    return SQLITE_OK;
  }
  set->slots[slot] = id;
  ++set->count;
  *added = 1;
  return SQLITE_OK;
}

static int search_visited_begin(HnswVtab *vtab, HnswCursor *cursor) {
  size_t capacity = vtab->snapshot.node_capacity;
  if (cursor == NULL || !vtab->snapshot.ready) {
    return SQLITE_OK;
  }
  if (cursor->visited_capacity < capacity) {
    uint32_t *grown = NULL;
    if ((sqlite3_uint64)capacity >
        HNSW_ALLOCATION_LIMIT / sizeof(uint32_t)) {
      return SQLITE_TOOBIG;
    }
    grown = sqlite3_realloc64(
        cursor->visited_epochs,
        (sqlite3_uint64)capacity * sizeof(*cursor->visited_epochs));
    if (grown == NULL && capacity > 0) {
      return SQLITE_NOMEM;
    }
    if (capacity > cursor->visited_capacity) {
      memset(grown + cursor->visited_capacity, 0,
             (capacity - cursor->visited_capacity) * sizeof(*grown));
    }
    cursor->visited_epochs = grown;
    cursor->visited_capacity = capacity;
  }
  ++cursor->visited_epoch;
  if (cursor->visited_epoch == 0) {
    memset(cursor->visited_epochs, 0,
           cursor->visited_capacity * sizeof(*cursor->visited_epochs));
    cursor->visited_epoch = 1;
  }
  return SQLITE_OK;
}

static int search_visited_add(IdSet *set, HnswCursor *cursor,
                              sqlite3_int64 id, int *added) {
  if (cursor != NULL && cursor->visited_epochs != NULL) {
    size_t index = 0;
    if (id <= 0 || (uint64_t)id > cursor->visited_capacity) {
      return SQLITE_CORRUPT_VTAB;
    }
    index = (size_t)(id - 1);
    if (cursor->visited_epochs[index] == cursor->visited_epoch) {
      *added = 0;
    } else {
      cursor->visited_epochs[index] = cursor->visited_epoch;
      *added = 1;
    }
    return SQLITE_OK;
  }
  return id_set_add(set, id, added);
}

static int compare_heap_item_ascending(const void *left, const void *right) {
  const HeapItem *a = (const HeapItem *)left;
  const HeapItem *b = (const HeapItem *)right;
  if (a->distance < b->distance) {
    return -1;
  }
  if (a->distance > b->distance) {
    return 1;
  }
  return a->id < b->id ? -1 : (a->id > b->id ? 1 : 0);
}

static int search_layer(HnswVtab *vtab, const float *query, double query_norm,
                        sqlite3_int64 entry, int layer, int ef,
                        int include_deleted, HnswCursor *cursor,
                        SearchOutput *output) {
  Heap candidates = {0};
  Heap nearest = {0};
  IdSet visited = {0};
  HeapItem initial;
  int deleted = 0;
  int node_level = 0;
  int added = 0;
  int rc = SQLITE_OK;
  memset(output, 0, sizeof(*output));
  candidates.is_min = 1;
  nearest.is_min = 0;
  rc = search_visited_begin(vtab, cursor);
  if (rc != SQLITE_OK) {
    goto done;
  }
  rc = node_distance(vtab, query, query_norm, entry, &initial.distance,
                     &deleted, &node_level, &output->visited);
  initial.id = entry;
  if (rc != SQLITE_OK || node_level < layer) {
    if (rc == SQLITE_OK) {
      rc = SQLITE_CORRUPT_VTAB;
    }
    goto done;
  }
  rc = search_visited_add(&visited, cursor, entry, &added);
  if (rc == SQLITE_OK) {
    rc = heap_push(&candidates, initial);
  }
  if (rc == SQLITE_OK && (include_deleted || !deleted)) {
    rc = heap_push(&nearest, initial);
  }
  while (rc == SQLITE_OK && candidates.count > 0) {
    HeapItem current = heap_pop(&candidates);
    EdgeView edges = {0};
    int i = 0;
    if (nearest.count >= ef && current.distance > nearest.items[0].distance) {
      break;
    }
    rc = acquire_edges(vtab, current.id, layer, &edges);
    if (rc != SQLITE_OK) {
      release_edges(&edges);
      break;
    }
    for (i = 0; i < edges.count && rc == SQLITE_OK; ++i) {
      HeapItem candidate;
      int candidate_deleted = 0;
      int candidate_level = 0;
      sqlite3_int64 edge_id = edge_view_id(&edges, i);
      rc = search_visited_add(&visited, cursor, edge_id, &added);
      if (rc != SQLITE_OK || !added) {
        continue;
      }
      candidate.id = edge_id;
      rc = node_distance(vtab, query, query_norm, candidate.id,
                         &candidate.distance, &candidate_deleted,
                         &candidate_level, &output->visited);
      if (rc != SQLITE_OK || candidate_level < layer) {
        if (rc == SQLITE_OK) {
          rc = SQLITE_CORRUPT_VTAB;
        }
        break;
      }
      if (nearest.count < ef ||
          candidate.distance <= nearest.items[0].distance) {
        rc = heap_push(&candidates, candidate);
        if (rc != SQLITE_OK) {
          break;
        }
        if (include_deleted || !candidate_deleted) {
          rc = heap_push(&nearest, candidate);
          if (rc == SQLITE_OK && nearest.count > ef) {
            (void)heap_pop(&nearest);
          }
        }
      }
    }
    release_edges(&edges);
  }
  if (rc == SQLITE_OK && nearest.count > 0) {
    int i = 0;
    output->items =
        sqlite3_malloc64((sqlite3_uint64)nearest.count * sizeof(HeapItem));
    if (output->items == NULL) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    output->count = nearest.count;
    for (i = 0; i < output->count; ++i) {
      output->items[i] = heap_pop(&nearest);
    }
    qsort(output->items, (size_t)output->count, sizeof(HeapItem),
          compare_heap_item_ascending);
  }
done:
  if (rc != SQLITE_OK) {
    sqlite3_free(output->items);
    output->items = NULL;
    output->count = 0;
  }
  heap_free(&candidates);
  heap_free(&nearest);
  id_set_free(&visited);
  return rc;
}

static int greedy_at_layer(HnswVtab *vtab, const float *query,
                           double query_norm, sqlite3_int64 start, int layer,
                           sqlite3_int64 *best_id, double *best_distance,
                           int *visited) {
  int changed = 1;
  int deleted = 0;
  int node_level = 0;
  int rc = node_distance(vtab, query, query_norm, start, best_distance,
                         &deleted, &node_level, visited);
  if (rc == SQLITE_OK && node_level < layer) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  *best_id = start;
  while (rc == SQLITE_OK && changed) {
    EdgeView edges = {0};
    int i = 0;
    changed = 0;
    rc = acquire_edges(vtab, *best_id, layer, &edges);
    for (i = 0; i < edges.count && rc == SQLITE_OK; ++i) {
      double distance = 0.0;
      int edge_level = 0;
      sqlite3_int64 edge_id = edge_view_id(&edges, i);
      rc = node_distance(vtab, query, query_norm, edge_id, &distance,
                         &deleted, &edge_level, visited);
      if (rc == SQLITE_OK && edge_level < layer) {
        rc = SQLITE_CORRUPT_VTAB;
      }
      if (rc == SQLITE_OK &&
          (distance < *best_distance ||
           (distance == *best_distance && edge_id < *best_id))) {
        *best_distance = distance;
        *best_id = edge_id;
        changed = 1;
      }
    }
    release_edges(&edges);
  }
  return rc;
}

static int contains_id(const sqlite3_int64 *ids, int count, sqlite3_int64 id) {
  int i = 0;
  for (i = 0; i < count; ++i) {
    if (ids[i] == id) {
      return 1;
    }
  }
  return 0;
}

static int select_neighbors(HnswVtab *vtab, HeapItem *candidates,
                            int candidate_count, int maximum, IdList *out) {
  sqlite3_int64 *pruned = NULL;
  int pruned_count = 0;
  int i = 0;
  int rc = SQLITE_OK;
  memset(out, 0, sizeof(*out));
  if (candidate_count == 0 || maximum == 0) {
    return SQLITE_OK;
  }
  qsort(candidates, (size_t)candidate_count, sizeof(HeapItem),
        compare_heap_item_ascending);
  out->ids = sqlite3_malloc64((sqlite3_uint64)maximum * sizeof(sqlite3_int64));
  pruned =
      sqlite3_malloc64((sqlite3_uint64)candidate_count * sizeof(sqlite3_int64));
  if (out->ids == NULL || pruned == NULL) {
    sqlite3_free(pruned);
    free_id_list(out);
    return SQLITE_NOMEM;
  }
  for (i = 0; i < candidate_count && out->count < maximum; ++i) {
    int diverse = 1;
    int j = 0;
    for (j = 0; j < out->count; ++j) {
      double between = 0.0;
      rc =
          distance_between_nodes(vtab, candidates[i].id, out->ids[j], &between);
      if (rc != SQLITE_OK) {
        goto done;
      }
      if (between < candidates[i].distance) {
        diverse = 0;
        break;
      }
    }
    if (diverse) {
      out->ids[out->count++] = candidates[i].id;
    } else {
      pruned[pruned_count++] = candidates[i].id;
    }
  }
  for (i = 0; i < pruned_count && out->count < maximum; ++i) {
    if (!contains_id(out->ids, out->count, pruned[i])) {
      out->ids[out->count++] = pruned[i];
    }
  }
done:
  sqlite3_free(pruned);
  if (rc != SQLITE_OK) {
    free_id_list(out);
  }
  return rc;
}

static int prune_neighbors_for_owner(HnswVtab *vtab, sqlite3_int64 owner,
                                     const IdList *candidates, int maximum,
                                     IdList *selected) {
  NodeView owner_view = {0};
  HeapItem *items = NULL;
  int i = 0;
  int rc = acquire_node(vtab, owner, &owner_view);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (candidates->count > 0) {
    items =
        sqlite3_malloc64((sqlite3_uint64)candidates->count * sizeof(HeapItem));
    if (items == NULL) {
      release_node(&owner_view);
      return SQLITE_NOMEM;
    }
  }
  for (i = 0; i < candidates->count && rc == SQLITE_OK; ++i) {
    int candidate_deleted = 0;
    items[i].id = candidates->ids[i];
    rc = node_distance(vtab, owner_view.vector, owner_view.norm, items[i].id,
                       &items[i].distance, &candidate_deleted, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = select_neighbors(vtab, items, candidates->count, maximum, selected);
  }
  release_node(&owner_view);
  sqlite3_free(items);
  return rc;
}

static int add_backlink(HnswVtab *vtab, sqlite3_int64 owner,
                        sqlite3_int64 neighbor, int layer, int maximum) {
  IdList current = {0};
  IdList selected = {0};
  sqlite3_int64 *grown = NULL;
  int rc = fetch_edges(vtab, owner, layer, &current);
  if (rc != SQLITE_OK || contains_id(current.ids, current.count, neighbor)) {
    free_id_list(&current);
    return rc;
  }
  grown = sqlite3_realloc64(current.ids, (sqlite3_uint64)(current.count + 1) *
                                             sizeof(sqlite3_int64));
  if (grown == NULL) {
    free_id_list(&current);
    return SQLITE_NOMEM;
  }
  current.ids = grown;
  current.ids[current.count++] = neighbor;
  if (current.count <= maximum) {
    rc = write_edges(vtab, owner, layer, &current);
  } else {
    rc = prune_neighbors_for_owner(vtab, owner, &current, maximum, &selected);
    if (rc == SQLITE_OK) {
      rc = write_edges(vtab, owner, layer, &selected);
    }
  }
  free_id_list(&current);
  free_id_list(&selected);
  return rc;
}

static int random_level(sqlite3_int64 node_id, int m) {
  uint64_t bits = hash_id((uint64_t)node_id ^ UINT64_C(0x9e3779b97f4a7c15));
  double unit =
      ((double)(bits >> 11U) + 1.0) / ((double)(UINT64_C(1) << 53U) + 1.0);
  int level = (int)(-log(unit) / log((double)m));
  if (level < 0) {
    return 0;
  }
  return level > HNSW_MAX_LEVEL ? HNSW_MAX_LEVEL : level;
}

static int insert_node_record(HnswVtab *vtab, sqlite3_int64 node_id, int level,
                              const void *blob, int bytes, double norm) {
  char *table = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "INSERT INTO %s(node_id,level,vector,norm,deleted) VALUES(?,?,?,?,0)",
      table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->insert_node_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->insert_node_stmt;
  sqlite3_bind_int64(statement, 1, node_id);
  sqlite3_bind_int(statement, 2, level);
  sqlite3_bind_blob(statement, 3, blob, bytes, SQLITE_TRANSIENT);
  sqlite3_bind_double(statement, 4, norm);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  return finish_cached_statement(statement, rc);
}

static int insert_row_mapping(HnswVtab *vtab, sqlite3_int64 rowid,
                              sqlite3_int64 node_id) {
  char *table = shadow_table(vtab, "rows");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("INSERT INTO %s(rowid,node_id) VALUES(?,?)", table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->insert_row_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->insert_row_stmt;
  sqlite3_bind_int64(statement, 1, rowid);
  sqlite3_bind_int64(statement, 2, node_id);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  return finish_cached_statement(statement, rc);
}

static int link_existing_node(HnswVtab *vtab, sqlite3_int64 node_id,
                              const float *vector, double norm, int level,
                              HnswState *state) {
  sqlite3_int64 current = 0;
  double current_distance = 0.0;
  int layer = 0;
  int visited = 0;
  int rc = SQLITE_OK;
  if (state->entry_node <= 0) {
    IdList empty = {0};
    for (layer = 0; layer <= level && rc == SQLITE_OK; ++layer) {
      rc = write_edges(vtab, node_id, layer, &empty);
    }
    state->entry_node = node_id;
    state->max_level = level;
  } else {
    current = state->entry_node;
    for (layer = state->max_level; layer > level && rc == SQLITE_OK; --layer) {
      rc = greedy_at_layer(vtab, vector, norm, current, layer, &current,
                           &current_distance, &visited);
    }
    for (layer = level < state->max_level ? level : state->max_level;
         layer >= 0 && rc == SQLITE_OK; --layer) {
      SearchOutput found = {0};
      IdList selected = {0};
      int maximum = layer == 0 ? vtab->m * 2 : vtab->m;
      int i = 0;
      rc = search_layer(vtab, vector, norm, current, layer,
                        vtab->ef_construction, 0, NULL, &found);
      if (rc == SQLITE_OK) {
        rc = select_neighbors(vtab, found.items, found.count, maximum,
                              &selected);
      }
      if (rc == SQLITE_OK) {
        rc = write_edges(vtab, node_id, layer, &selected);
      }
      for (i = 0; i < selected.count && rc == SQLITE_OK; ++i) {
        rc = add_backlink(vtab, selected.ids[i], node_id, layer, maximum);
      }
      if (found.count > 0) {
        current = found.items[0].id;
      }
      sqlite3_free(found.items);
      free_id_list(&selected);
    }
    if (level > state->max_level) {
      IdList empty = {0};
      for (layer = state->max_level + 1; layer <= level && rc == SQLITE_OK;
           ++layer) {
        rc = write_edges(vtab, node_id, layer, &empty);
      }
      state->entry_node = node_id;
      state->max_level = level;
    }
  }
  return rc;
}

static int stage_node(HnswVtab *vtab, sqlite3_int64 rowid, const void *blob,
                      int bytes, double norm) {
  HnswState state;
  sqlite3_int64 node_id = 0;
  int level = 0;
  int rc = load_state(vtab, &state);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (state.next_node_id <= 0 || state.next_node_id == INT64_MAX) {
    return SQLITE_FULL;
  }
  node_id = state.next_node_id++;
  level = random_level(node_id, vtab->m);
  rc = insert_node_record(vtab, node_id, -level - 2, blob, bytes, norm);
  if (rc == SQLITE_OK) {
    rc = insert_row_mapping(vtab, rowid, node_id);
  }
  if (rc == SQLITE_OK) {
    ++state.live_count;
    ++state.change_seq;
    rc = save_state(vtab, &state);
    if (rc == SQLITE_OK) {
      vtab->cache_seq = state.change_seq;
    }
  }
  return rc;
}

static int mark_node_linked(HnswVtab *vtab, sqlite3_int64 node_id, int level) {
  char *nodes = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (nodes == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("UPDATE %s SET level=? WHERE node_id=?", nodes);
  sqlite3_free(nodes);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int(statement, 1, level);
  sqlite3_bind_int64(statement, 2, node_id);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  if (rc == SQLITE_OK) {
    node_cache_invalidate(vtab, node_id);
  }
  return rc;
}

static int arena_is_cancelled(void *context) {
  return sqlite3_is_interrupted((sqlite3 *)context);
}

static int link_staged_with_arena(HnswVtab *vtab, const sqlite3_int64 *ids,
                                  const int *levels, int count,
                                  HnswState *state) {
  HnswBuildArena arena;
  int threads = vtab->build_threads;
  size_t memory_limit = (size_t)vtab->build_memory_mb * 1024U * 1024U;
  sqlite3_int64 *neighbor_ids = NULL;
  sqlite3_stmt *load_statement = NULL;
  char *nodes = NULL;
  char *sql = NULL;
  int rc = SQLITE_OK;
  int i = 0;
  if (threads == 0) {
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    threads = online > 0 && online < 16 ? (int)online : 16;
  }
  cache_clear_all(vtab);
  rc = hnsw_build_arena_init(&arena, (size_t)count, vtab->dims, vtab->m,
                             vtab->ef_construction, vtab->metric, threads,
                             memory_limit);
  if (rc != SQLITE_OK) {
    vtab->base.zErrMsg = sqlite3_mprintf(
        "HNSW build arena requires at least %llu bytes; limit is %llu bytes",
        (unsigned long long)arena.required_bytes,
        (unsigned long long)memory_limit);
    hnsw_build_arena_destroy(&arena);
    return rc;
  }
  arena.is_cancelled = arena_is_cancelled;
  arena.cancel_context = vtab->db;
  nodes = shadow_table(vtab, "nodes");
  if (nodes == NULL) {
    rc = SQLITE_NOMEM;
  } else {
    sql = sqlite3_mprintf(
        "SELECT node_id,vector,norm,level FROM %s WHERE level<=-2 AND "
        "deleted=0 ORDER BY node_id",
        nodes);
    sqlite3_free(nodes);
    nodes = NULL;
    if (sql == NULL) {
      rc = SQLITE_NOMEM;
    } else {
      rc = sqlite3_prepare_v2(vtab->db, sql, -1, &load_statement, NULL);
      sqlite3_free(sql);
      sql = NULL;
    }
  }
  for (i = 0; i < count && rc == SQLITE_OK; ++i) {
    char *error = NULL;
    int step_rc = sqlite3_step(load_statement);
    if (step_rc != SQLITE_ROW ||
        sqlite3_column_int64(load_statement, 0) != ids[i] ||
        -sqlite3_column_int(load_statement, 3) - 2 != levels[i]) {
      rc = step_rc == SQLITE_ROW ? SQLITE_CORRUPT_VTAB : step_rc;
      break;
    }
    arena.node_ids[i] = ids[i];
    arena.levels[i] = (unsigned char)levels[i];
    arena.norms[i] = sqlite3_column_double(load_statement, 2);
    rc = hnsw_vector_decode_into(sqlite3_column_blob(load_statement, 1),
                                 sqlite3_column_bytes(load_statement, 1),
                                 arena.vectors + (size_t)i * (size_t)vtab->dims,
                                 vtab->dims, &error);
    sqlite3_free(error);
  }
  if (load_statement != NULL && sqlite3_finalize(load_statement) != SQLITE_OK &&
      rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  if (rc == SQLITE_OK) {
    rc = hnsw_build_arena_prepare(&arena);
    if (rc == SQLITE_TOOBIG) {
      vtab->base.zErrMsg = sqlite3_mprintf(
          "HNSW build arena requires %llu bytes; build_memory_mb allows %llu",
          (unsigned long long)arena.required_bytes,
          (unsigned long long)memory_limit);
    }
  }
  if (rc == SQLITE_OK) {
    rc = hnsw_build_arena_run(&arena);
    if (rc != SQLITE_OK && vtab->base.zErrMsg == NULL) {
      vtab->base.zErrMsg =
          sqlite3_mprintf("parallel HNSW arena build failed (%d)", rc);
    }
  }
  if (rc == SQLITE_OK) {
    neighbor_ids =
        sqlite3_malloc64((sqlite3_uint64)(vtab->m * 2) * sizeof(sqlite3_int64));
    if (neighbor_ids == NULL) {
      rc = SQLITE_NOMEM;
    }
  }
  for (i = 0; i < count && rc == SQLITE_OK; ++i) {
    int layer = 0;
    for (layer = 0; layer <= levels[i] && rc == SQLITE_OK; ++layer) {
      int neighbor_count = 0;
      const uint32_t *neighbors = hnsw_build_arena_neighbors(
          &arena, (uint32_t)i, layer, &neighbor_count);
      IdList list = {neighbor_ids, neighbor_count};
      int j = 0;
      for (j = 0; j < neighbor_count; ++j) {
        neighbor_ids[j] = arena.node_ids[neighbors[j]];
      }
      rc = write_edges_storage(vtab, ids[i], layer, &list);
    }
    if (rc == SQLITE_OK) {
      rc = mark_node_linked(vtab, ids[i], levels[i]);
    }
    if (rc == SQLITE_OK && sqlite3_is_interrupted(vtab->db)) {
      rc = SQLITE_INTERRUPT;
    }
  }
  if (rc == SQLITE_OK && count > 0) {
    state->entry_node = arena.node_ids[arena.entry];
    state->max_level = arena.max_level;
    ++state->change_seq;
    rc = save_state(vtab, state);
    if (rc == SQLITE_OK) {
      vtab->cache_seq = state->change_seq;
    }
  }
  sqlite3_free(neighbor_ids);
  sqlite3_free(nodes);
  sqlite3_free(sql);
  hnsw_build_arena_destroy(&arena);
  if (rc != SQLITE_OK) {
    cache_clear_all(vtab);
    vtab->cache_seq = -1;
  }
  return rc;
}

static int link_staged_nodes(HnswVtab *vtab) {
  HnswState state;
  char *nodes = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  sqlite3_int64 *ids = NULL;
  int *levels = NULL;
  int count = 0;
  int capacity = 0;
  int i = 0;
  int rc = SQLITE_OK;
  if (nodes == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT node_id,level FROM %s WHERE level<=-2 AND deleted=0 "
      "ORDER BY node_id",
      nodes);
  sqlite3_free(nodes);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    if (count == capacity) {
      int next = capacity == 0 ? 256 : capacity * 2;
      sqlite3_int64 *grown_ids = NULL;
      int *grown_levels = NULL;
      if (next < capacity ||
          (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(*ids)) {
        rc = SQLITE_TOOBIG;
        break;
      }
      grown_ids =
          sqlite3_realloc64(ids, (sqlite3_uint64)next * sizeof(sqlite3_int64));
      if (grown_ids == NULL) {
        rc = SQLITE_NOMEM;
        break;
      }
      ids = grown_ids;
      grown_levels =
          sqlite3_realloc64(levels, (sqlite3_uint64)next * sizeof(int));
      if (grown_levels == NULL) {
        rc = SQLITE_NOMEM;
        break;
      }
      levels = grown_levels;
      capacity = next;
    }
    ids[count] = sqlite3_column_int64(statement, 0);
    levels[count] = -sqlite3_column_int(statement, 1) - 2;
    ++count;
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  statement = NULL;
  if (rc == SQLITE_OK && count > 0) {
    rc = load_state(vtab, &state);
    if (rc == SQLITE_OK && state.entry_node <= 0 &&
        state.tombstone_count == 0 && count >= HNSW_ARENA_THRESHOLD) {
      rc = link_staged_with_arena(vtab, ids, levels, count, &state);
      sqlite3_free(ids);
      sqlite3_free(levels);
      return rc;
    }
    if (rc == SQLITE_OK) {
      vtab->defer_edge_writes = 1;
    }
  }
  for (i = 0; i < count && rc == SQLITE_OK; ++i) {
    NodeView view = {0};
    rc = acquire_node(vtab, ids[i], &view);
    if (rc == SQLITE_OK && !view.deleted) {
      rc = link_existing_node(vtab, ids[i], view.vector, view.norm, levels[i],
                              &state);
    }
    release_node(&view);
    if (rc == SQLITE_OK) {
      rc = mark_node_linked(vtab, ids[i], levels[i]);
    }
    if (rc == SQLITE_OK && sqlite3_is_interrupted(vtab->db)) {
      rc = SQLITE_INTERRUPT;
    }
  }
  if (rc == SQLITE_OK && count > 0) {
    rc = flush_edge_cache(vtab);
  }
  vtab->defer_edge_writes = 0;
  if (rc == SQLITE_OK && count > 0) {
    ++state.change_seq;
    rc = save_state(vtab, &state);
    if (rc == SQLITE_OK) {
      vtab->cache_seq = state.change_seq;
    }
  }
  if (rc != SQLITE_OK) {
    cache_clear_all(vtab);
    vtab->cache_seq = -1;
  }
  sqlite3_finalize(statement);
  sqlite3_free(ids);
  sqlite3_free(levels);
  return rc;
}

static int lookup_node_for_rowid(HnswVtab *vtab, sqlite3_int64 rowid,
                                 sqlite3_int64 *node_id) {
  char *table = shadow_table(vtab, "rows");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT node_id FROM %s WHERE rowid=?", table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = prepare_cached(vtab, &vtab->read_node_id_stmt, sql);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  statement = vtab->read_node_id_stmt;
  sqlite3_bind_int64(statement, 1, rowid);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    *node_id = sqlite3_column_int64(statement, 0);
    rc = SQLITE_OK;
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_NOTFOUND;
  }
  return finish_cached_statement(statement, rc);
}

static int hnsw_delete_internal(HnswVtab *vtab, sqlite3_int64 rowid) {
  HnswState state;
  sqlite3_int64 node_id = 0;
  float *node_vector = NULL;
  double node_norm = 0.0;
  int node_deleted = 0;
  int node_level = 0;
  char *rows = NULL;
  char *nodes = NULL;
  char *edges = NULL;
  char *sql = NULL;
  int rc = lookup_node_for_rowid(vtab, rowid, &node_id);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = fetch_node(vtab, node_id, &node_vector, &node_norm, &node_deleted,
                  &node_level);
  sqlite3_free(node_vector);
  (void)node_norm;
  (void)node_deleted;
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = load_state(vtab, &state);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rows = shadow_table(vtab, "rows");
  nodes = shadow_table(vtab, "nodes");
  if (rows == NULL || nodes == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  if (node_level <= -2) {
    sql = sqlite3_mprintf(
        "DELETE FROM %s WHERE node_id=%lld;DELETE FROM %s WHERE rowid=%lld",
        nodes, (long long)node_id, rows, (long long)rowid);
  } else {
    sql = sqlite3_mprintf(
        "UPDATE %s SET deleted=1 WHERE node_id=%lld;DELETE FROM %s WHERE "
        "rowid=%lld",
        nodes, (long long)node_id, rows, (long long)rowid);
  }
  if (sql == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = exec_sql(vtab->db, &vtab->base.zErrMsg, sql);
  sqlite3_free(sql);
  sql = NULL;
  if (rc != SQLITE_OK) {
    goto done;
  }
  node_cache_invalidate(vtab, node_id);
  --state.live_count;
  if (node_level >= 0) {
    ++state.tombstone_count;
  }
  ++state.change_seq;
  if (state.live_count == 0) {
    edges = shadow_table(vtab, "edges");
    if (edges == NULL) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    sql = sqlite3_mprintf("DELETE FROM %s;DELETE FROM %s", nodes, edges);
    if (sql == NULL) {
      rc = SQLITE_NOMEM;
      goto done;
    }
    rc = exec_sql(vtab->db, &vtab->base.zErrMsg, sql);
    sqlite3_free(sql);
    sql = NULL;
    state.entry_node = -1;
    state.max_level = -1;
    state.tombstone_count = 0;
    cache_clear_all(vtab);
  }
  if (rc == SQLITE_OK) {
    rc = save_state(vtab, &state);
    if (rc == SQLITE_OK) {
      vtab->cache_seq = state.change_seq;
    }
  }
done:
  sqlite3_free(sql);
  sqlite3_free(rows);
  sqlite3_free(nodes);
  sqlite3_free(edges);
  return rc;
}

static int append_result(HnswCursor *cursor, sqlite3_int64 rowid,
                         sqlite3_int64 node_id, double distance) {
  if (cursor->count == cursor->capacity) {
    int next = cursor->capacity == 0 ? 32 : cursor->capacity * 2;
    ResultRow *grown = NULL;
    if (next < cursor->capacity ||
        (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(ResultRow)) {
      return SQLITE_TOOBIG;
    }
    grown = sqlite3_realloc64(cursor->rows,
                              (sqlite3_uint64)next * sizeof(ResultRow));
    if (grown == NULL) {
      return SQLITE_NOMEM;
    }
    cursor->rows = grown;
    cursor->capacity = next;
  }
  cursor->rows[cursor->count].rowid = rowid;
  cursor->rows[cursor->count].node_id = node_id;
  cursor->rows[cursor->count].distance = distance;
  ++cursor->count;
  return SQLITE_OK;
}

static int ann_search(HnswVtab *vtab, HnswCursor *cursor, const float *query,
                      double query_norm) {
  HnswState state;
  sqlite3_int64 current = 0;
  double current_distance = 0.0;
  SearchOutput found = {0};
  Heap nearest = {0};
  int visited = 0;
  int layer = 0;
  int rc = load_state(vtab, &state);
  nearest.is_min = 0;
  if (rc != SQLITE_OK || state.live_count == 0) {
    return rc;
  }
  if (state.entry_node > 0) {
    current = state.entry_node;
    for (layer = state.max_level; layer > 0 && rc == SQLITE_OK; --layer) {
      rc = greedy_at_layer(vtab, query, query_norm, current, layer, &current,
                           &current_distance, &visited);
    }
  }
  if (rc == SQLITE_OK && state.entry_node > 0) {
    rc = search_layer(vtab, query, query_norm, current, 0, cursor->ef_search,
                      0, cursor, &found);
  }
  cursor->visited_count = visited + found.visited;
  if (rc == SQLITE_OK) {
    int i = 0;
    for (i = 0; i < found.count && rc == SQLITE_OK; ++i) {
      rc = heap_push(&nearest, found.items[i]);
      if (rc == SQLITE_OK && nearest.count > cursor->k) {
        (void)heap_pop(&nearest);
      }
    }
  }
  if (rc == SQLITE_OK && (!vtab->snapshot.ready || vtab->write_dirty)) {
    char *rows = shadow_table(vtab, "rows");
    char *nodes = shadow_table(vtab, "nodes");
    char *sql = NULL;
    sqlite3_stmt *statement = NULL;
    int step_rc = SQLITE_DONE;
    if (rows == NULL || nodes == NULL) {
      rc = SQLITE_NOMEM;
    } else {
      sql = sqlite3_mprintf(
          "SELECT r.node_id,n.vector,n.norm FROM %s AS r JOIN %s AS n ON "
          "n.node_id=r.node_id WHERE n.level<=-2 AND n.deleted=0",
          rows, nodes);
      if (sql == NULL) {
        rc = SQLITE_NOMEM;
      } else {
        rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
      }
    }
    while (rc == SQLITE_OK &&
           (step_rc = sqlite3_step(statement)) == SQLITE_ROW) {
      float *vector = NULL;
      int dims = 0;
      char *error = NULL;
      HeapItem item;
      item.id = sqlite3_column_int64(statement, 0);
      rc = hnsw_vector_decode(sqlite3_column_blob(statement, 1),
                              sqlite3_column_bytes(statement, 1), &vector,
                              &dims, &error);
      sqlite3_free(error);
      if (rc == SQLITE_OK && dims == vtab->dims) {
        item.distance = hnsw_vector_distance(
            query, query_norm, vector, sqlite3_column_double(statement, 2),
            vtab->dims, vtab->metric);
        ++cursor->visited_count;
        rc = heap_push(&nearest, item);
        if (rc == SQLITE_OK && nearest.count > cursor->k) {
          (void)heap_pop(&nearest);
        }
      } else if (rc == SQLITE_OK) {
        rc = SQLITE_CORRUPT_VTAB;
      }
      sqlite3_free(vector);
    }
    if (rc == SQLITE_OK && step_rc != SQLITE_DONE) {
      rc = step_rc;
    }
    if (statement != NULL) {
      int reset_rc = sqlite3_reset(statement);
      if (rc == SQLITE_OK && reset_rc != SQLITE_OK) {
        rc = reset_rc;
      }
      sqlite3_finalize(statement);
    }
    sqlite3_free(sql);
    sqlite3_free(rows);
    sqlite3_free(nodes);
  }
  if (rc == SQLITE_OK && nearest.count > 0) {
    int count = nearest.count;
    HeapItem *items = sqlite3_malloc64((sqlite3_uint64)count * sizeof(*items));
    int i = 0;
    if (items == NULL) {
      rc = SQLITE_NOMEM;
    } else {
      for (i = 0; i < count; ++i) {
        items[i] = heap_pop(&nearest);
      }
      qsort(items, (size_t)count, sizeof(*items), compare_heap_item_ascending);
      for (i = 0; i < count && rc == SQLITE_OK; ++i) {
        sqlite3_int64 rowid = 0;
        rc = fetch_rowid_for_node(vtab, items[i].id, &rowid);
        if (rc == SQLITE_OK) {
          double distance = vtab->metric == HNSW_METRIC_L2
                                ? sqrt(items[i].distance)
                                : items[i].distance;
          rc = append_result(cursor, rowid, items[i].id, distance);
        }
      }
      sqlite3_free(items);
    }
  }
  sqlite3_free(found.items);
  heap_free(&nearest);
  return rc;
}

static int load_scan(HnswVtab *vtab, HnswCursor *cursor,
                     const sqlite3_int64 *only_rowid) {
  char *rows = shadow_table(vtab, "rows");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (rows == NULL) {
    return SQLITE_NOMEM;
  }
  sql =
      only_rowid == NULL
          ? sqlite3_mprintf("SELECT rowid,node_id FROM %s ORDER BY rowid", rows)
          : sqlite3_mprintf("SELECT rowid,node_id FROM %s WHERE rowid=?", rows);
  sqlite3_free(rows);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (only_rowid != NULL) {
    sqlite3_bind_int64(statement, 1, *only_rowid);
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    rc = append_result(cursor, sqlite3_column_int64(statement, 0),
                       sqlite3_column_int64(statement, 1), 0.0);
    if (rc != SQLITE_OK) {
      break;
    }
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  return rc;
}

static int create_shadow_tables(HnswVtab *vtab, char **error) {
  char *meta = shadow_table(vtab, "meta");
  char *rows = shadow_table(vtab, "rows");
  char *nodes = shadow_table(vtab, "nodes");
  char *edges = shadow_table(vtab, "edges");
  char *rebuild = shadow_table(vtab, "rebuild");
  char *sql = NULL;
  int rc = SQLITE_OK;
  if (meta == NULL || rows == NULL || nodes == NULL || edges == NULL ||
      rebuild == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  sql = sqlite3_mprintf(
      "CREATE TABLE %s("
      "id INTEGER PRIMARY KEY CHECK(id=1),format_version INTEGER NOT NULL,"
      "vector_name TEXT NOT NULL,dims INTEGER NOT NULL,metric INTEGER NOT NULL,"
      "m INTEGER NOT NULL,ef_construction INTEGER NOT NULL,"
      "cache_size_mb INTEGER NOT NULL,entry_node INTEGER NOT NULL,"
      "max_level INTEGER NOT NULL,"
      "next_node_id INTEGER NOT NULL,live_count INTEGER NOT NULL,"
      "tombstone_count INTEGER NOT NULL,change_seq INTEGER NOT NULL);"
      "CREATE TABLE %s(rowid INTEGER PRIMARY KEY,node_id INTEGER UNIQUE NOT "
      "NULL);"
      "CREATE TABLE %s(node_id INTEGER PRIMARY KEY,level INTEGER NOT NULL,"
      "vector BLOB NOT NULL,norm REAL NOT NULL,deleted INTEGER NOT NULL "
      "CHECK(deleted IN(0,1)));"
      "CREATE TABLE %s(node_id INTEGER NOT NULL,layer INTEGER NOT NULL,"
      "neighbors BLOB NOT NULL,PRIMARY KEY(node_id,layer)) WITHOUT ROWID;"
      "CREATE TABLE %s(rowid INTEGER PRIMARY KEY,vector BLOB NOT NULL);"
      "INSERT INTO %s VALUES(1,%d,%Q,%d,%d,%d,%d,%d,-1,-1,1,0,0,0)",
      meta, rows, nodes, edges, rebuild, meta, HNSW_FORMAT_VERSION,
      vtab->vector_name, vtab->dims, vtab->metric, vtab->m,
      vtab->ef_construction, vtab->cache_size_mb);
  if (sql == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = exec_sql(vtab->db, error, sql);
done:
  sqlite3_free(sql);
  sqlite3_free(meta);
  sqlite3_free(rows);
  sqlite3_free(nodes);
  sqlite3_free(edges);
  sqlite3_free(rebuild);
  return rc;
}

static int verify_meta(HnswVtab *vtab, char **error) {
  char *meta = shadow_table(vtab, "meta");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (meta == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf(
      "SELECT format_version,vector_name,dims,metric,m,ef_construction,"
      "cache_size_mb FROM %s WHERE id=1", meta);
  sqlite3_free(meta);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    *error = sqlite3_mprintf("cannot open hnsw metadata: %s",
                             sqlite3_errmsg(vtab->db));
    return rc;
  }
  rc = sqlite3_step(statement);
  if (rc != SQLITE_ROW) {
    *error = sqlite3_mprintf("missing hnsw metadata");
    rc = SQLITE_CORRUPT_VTAB;
  } else if (sqlite3_column_int(statement, 0) != HNSW_FORMAT_VERSION ||
             strcmp((const char *)sqlite3_column_text(statement, 1),
                    vtab->vector_name) != 0 ||
             sqlite3_column_int(statement, 2) != vtab->dims ||
             sqlite3_column_int(statement, 3) != vtab->metric ||
             sqlite3_column_int(statement, 4) != vtab->m ||
             sqlite3_column_int(statement, 5) != vtab->ef_construction) {
    *error = sqlite3_mprintf("hnsw metadata does not match CREATE statement");
    rc = SQLITE_CORRUPT_VTAB;
  } else {
    vtab->cache_size_mb = sqlite3_column_int(statement, 6);
    set_cache_budget(vtab);
    rc = SQLITE_OK;
  }
  sqlite3_finalize(statement);
  return rc;
}

static int hnsw_create(sqlite3 *db, void *aux, int argc,
                       const char *const *argv, sqlite3_vtab **out,
                       char **error) {
  HnswVtab *vtab = NULL;
  int rc = SQLITE_OK;
  (void)aux;
  rc = allocate_vtab(db, argc, argv, &vtab, error);
  if (rc == SQLITE_OK) {
    rc = declare_vtab(vtab, error);
  }
  if (rc == SQLITE_OK) {
    rc = create_shadow_tables(vtab, error);
  }
  if (rc != SQLITE_OK) {
    free_vtab(vtab);
    return rc;
  }
  *out = &vtab->base;
  return SQLITE_OK;
}

static int hnsw_connect(sqlite3 *db, void *aux, int argc,
                        const char *const *argv, sqlite3_vtab **out,
                        char **error) {
  HnswVtab *vtab = NULL;
  int rc = SQLITE_OK;
  (void)aux;
  rc = allocate_vtab(db, argc, argv, &vtab, error);
  if (rc == SQLITE_OK) {
    rc = declare_vtab(vtab, error);
  }
  if (rc == SQLITE_OK) {
    rc = verify_meta(vtab, error);
  }
  if (rc != SQLITE_OK) {
    free_vtab(vtab);
    return rc;
  }
  *out = &vtab->base;
  return SQLITE_OK;
}

static int hnsw_best_index(sqlite3_vtab *base, sqlite3_index_info *info) {
  int match = -1;
  int k = -1;
  int ef = -1;
  int rowid = -1;
  int i = 0;
  (void)base;
  for (i = 0; i < info->nConstraint; ++i) {
    const struct sqlite3_index_constraint *constraint = &info->aConstraint[i];
    if (!constraint->usable) {
      continue;
    }
    if (constraint->iColumn == COL_VECTOR &&
        constraint->op == SQLITE_INDEX_CONSTRAINT_MATCH) {
      match = i;
    } else if (constraint->iColumn == COL_K &&
               constraint->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      k = i;
    } else if (constraint->iColumn == COL_EF_SEARCH &&
               constraint->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      ef = i;
    } else if (constraint->iColumn == -1 &&
               constraint->op == SQLITE_INDEX_CONSTRAINT_EQ) {
      rowid = i;
    }
  }
  if (match >= 0) {
    int arg = 1;
    if (k < 0) {
      info->estimatedCost = 1.0e99;
      info->estimatedRows = 1000000;
      return SQLITE_CONSTRAINT;
    }
    info->idxNum = PLAN_ANN;
    info->aConstraintUsage[match].argvIndex = arg++;
    info->aConstraintUsage[match].omit = 1;
    info->aConstraintUsage[k].argvIndex = arg++;
    info->aConstraintUsage[k].omit = 1;
    if (ef >= 0) {
      info->idxNum |= 0x10;
      info->aConstraintUsage[ef].argvIndex = arg++;
      info->aConstraintUsage[ef].omit = 1;
    }
    if (info->nOrderBy == 1 && info->aOrderBy[0].iColumn == COL_DISTANCE &&
        info->aOrderBy[0].desc == 0) {
      info->orderByConsumed = 1;
    }
    info->estimatedCost = 100.0;
    info->estimatedRows = 10;
    return SQLITE_OK;
  }
  if (rowid >= 0) {
    info->idxNum = PLAN_ROWID;
    info->aConstraintUsage[rowid].argvIndex = 1;
    info->aConstraintUsage[rowid].omit = 1;
    info->idxFlags = SQLITE_INDEX_SCAN_UNIQUE;
    info->estimatedCost = 1.0;
    info->estimatedRows = 1;
  } else {
    info->idxNum = PLAN_SCAN;
    info->estimatedCost = 1000000.0;
    info->estimatedRows = 1000000;
  }
  return SQLITE_OK;
}

static int hnsw_disconnect(sqlite3_vtab *base) {
  free_vtab((HnswVtab *)base);
  return SQLITE_OK;
}

static int hnsw_destroy(sqlite3_vtab *base) {
  HnswVtab *vtab = (HnswVtab *)base;
  static const char *suffixes[] = {"meta", "rows", "nodes", "edges", "rebuild"};
  char *sql = sqlite3_mprintf("");
  int i = 0;
  int rc = SQLITE_OK;
  finalize_cached_statements(vtab);
  for (i = 0; i < 5 && sql != NULL; ++i) {
    char *table = shadow_table(vtab, suffixes[i]);
    char *next = NULL;
    if (table == NULL) {
      sqlite3_free(sql);
      sql = NULL;
      break;
    }
    next = sqlite3_mprintf("%zDROP TABLE IF EXISTS %s;", sql, table);
    sqlite3_free(table);
    sql = next;
  }
  if (sql == NULL) {
    rc = SQLITE_NOMEM;
  } else {
    rc = exec_sql(vtab->db, &vtab->base.zErrMsg, sql);
  }
  sqlite3_free(sql);
  free_vtab(vtab);
  return rc;
}

static int hnsw_open(sqlite3_vtab *base, sqlite3_vtab_cursor **out) {
  HnswCursor *cursor = sqlite3_malloc64(sizeof(*cursor));
  (void)base;
  if (cursor == NULL) {
    return SQLITE_NOMEM;
  }
  memset(cursor, 0, sizeof(*cursor));
  *out = &cursor->base;
  return SQLITE_OK;
}

static int hnsw_close(sqlite3_vtab_cursor *base) {
  HnswCursor *cursor = (HnswCursor *)base;
  HnswVtab *vtab = (HnswVtab *)base->pVtab;
  sqlite3_free(cursor->rows);
  sqlite3_free(cursor->visited_epochs);
  sqlite3_free(cursor);
  finalize_cached_statements(vtab);
  return SQLITE_OK;
}

static int hnsw_filter(sqlite3_vtab_cursor *base, int idx_num,
                       const char *idx_str, int argc, sqlite3_value **argv) {
  HnswCursor *cursor = (HnswCursor *)base;
  HnswVtab *vtab = (HnswVtab *)base->pVtab;
  int plan = idx_num & 0x0f;
  int arg = 0;
  int rc = SQLITE_OK;
  (void)idx_str;
  sqlite3_free(cursor->rows);
  cursor->rows = NULL;
  cursor->count = 0;
  cursor->capacity = 0;
  cursor->position = 0;
  cursor->k = 0;
  cursor->ef_search = 0;
  cursor->visited_count = 0;
  rc = sync_node_cache(vtab);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (plan == PLAN_ROWID) {
    sqlite3_int64 rowid = sqlite3_value_int64(argv[0]);
    return load_scan(vtab, cursor, &rowid);
  }
  if (plan == PLAN_SCAN) {
    return load_scan(vtab, cursor, NULL);
  }
  if (argc < 2 || sqlite3_value_type(argv[arg]) != SQLITE_BLOB) {
    vtab->base.zErrMsg = sqlite3_mprintf("MATCH requires a float32 BLOB");
    return SQLITE_CONSTRAINT;
  }
  {
    const void *blob = sqlite3_value_blob(argv[arg]);
    int bytes = sqlite3_value_bytes(argv[arg++]);
    float *query = NULL;
    int dims = 0;
    char *error = NULL;
    rc = hnsw_vector_validate(blob, bytes, vtab->dims, vtab->metric, &error);
    if (rc != SQLITE_OK) {
      vtab->base.zErrMsg = error;
      return rc;
    }
    cursor->k = sqlite3_value_int(argv[arg++]);
    if (cursor->k < 1 || cursor->k > 10000) {
      vtab->base.zErrMsg = sqlite3_mprintf("k must be between 1 and 10000");
      return SQLITE_CONSTRAINT;
    }
    cursor->ef_search = cursor->k > 64 ? cursor->k : 64;
    if ((idx_num & 0x10) != 0) {
      cursor->ef_search = sqlite3_value_int(argv[arg++]);
    }
    if (cursor->ef_search < cursor->k || cursor->ef_search > 1000000) {
      vtab->base.zErrMsg =
          sqlite3_mprintf("ef_search must be between k and 1000000");
      return SQLITE_CONSTRAINT;
    }
    rc = hnsw_vector_decode(blob, bytes, &query, &dims, &error);
    sqlite3_free(error);
    if (rc == SQLITE_OK) {
      rc = query_snapshot_ensure(vtab);
    }
    if (rc == SQLITE_OK) {
      rc = ann_search(vtab, cursor, query, hnsw_vector_norm(query, dims));
    }
    sqlite3_free(query);
  }
  return rc;
}

static int hnsw_next(sqlite3_vtab_cursor *base) {
  HnswCursor *cursor = (HnswCursor *)base;
  ++cursor->position;
  return SQLITE_OK;
}

static int hnsw_eof(sqlite3_vtab_cursor *base) {
  HnswCursor *cursor = (HnswCursor *)base;
  return cursor->position >= cursor->count;
}

static int result_vector_blob(HnswVtab *vtab, sqlite3_int64 node_id,
                              sqlite3_context *ctx) {
  char *nodes = shadow_table(vtab, "nodes");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (nodes == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT vector FROM %s WHERE node_id=?", nodes);
  sqlite3_free(nodes);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  sqlite3_bind_int64(statement, 1, node_id);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    sqlite3_result_blob(ctx, sqlite3_column_blob(statement, 0),
                        sqlite3_column_bytes(statement, 0), SQLITE_TRANSIENT);
    rc = SQLITE_OK;
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  return rc;
}

static int hnsw_column(sqlite3_vtab_cursor *base, sqlite3_context *ctx,
                       int column) {
  HnswCursor *cursor = (HnswCursor *)base;
  HnswVtab *vtab = (HnswVtab *)base->pVtab;
  ResultRow *row = NULL;
  if (cursor->position >= cursor->count) {
    sqlite3_result_null(ctx);
    return SQLITE_OK;
  }
  row = &cursor->rows[cursor->position];
  switch (column) {
  case COL_VECTOR:
    return result_vector_blob(vtab, row->node_id, ctx);
  case COL_DISTANCE:
    if (cursor->k > 0) {
      sqlite3_result_double(ctx, row->distance);
    } else {
      sqlite3_result_null(ctx);
    }
    break;
  case COL_K:
    if (cursor->k > 0) {
      sqlite3_result_int(ctx, cursor->k);
    } else {
      sqlite3_result_null(ctx);
    }
    break;
  case COL_EF_SEARCH:
    if (cursor->ef_search > 0) {
      sqlite3_result_int(ctx, cursor->ef_search);
    } else {
      sqlite3_result_null(ctx);
    }
    break;
  case COL_VISITED_COUNT:
    sqlite3_result_int(ctx, cursor->visited_count);
    break;
  default:
    sqlite3_result_null(ctx);
    break;
  }
  return SQLITE_OK;
}

static int hnsw_rowid(sqlite3_vtab_cursor *base, sqlite3_int64 *rowid) {
  HnswCursor *cursor = (HnswCursor *)base;
  if (cursor->position >= cursor->count) {
    return SQLITE_ERROR;
  }
  *rowid = cursor->rows[cursor->position].rowid;
  return SQLITE_OK;
}

static int next_external_rowid(HnswVtab *vtab, sqlite3_int64 *rowid) {
  char *rows = shadow_table(vtab, "rows");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (rows == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("SELECT max(rowid) FROM %s", rows);
  sqlite3_free(rows);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    if (sqlite3_column_type(statement, 0) == SQLITE_NULL) {
      *rowid = 1;
      rc = SQLITE_OK;
    } else {
      *rowid = sqlite3_column_int64(statement, 0);
      if (*rowid == INT64_MAX) {
        rc = SQLITE_FULL;
      } else {
        ++*rowid;
        rc = SQLITE_OK;
      }
    }
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  return rc;
}

static int clear_rebuild_table(HnswVtab *vtab) {
  char *table = shadow_table(vtab, "rebuild");
  char *sql = NULL;
  int rc = SQLITE_OK;
  if (table == NULL) {
    return SQLITE_NOMEM;
  }
  sql = sqlite3_mprintf("DELETE FROM %s", table);
  sqlite3_free(table);
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = exec_sql(vtab->db, &vtab->base.zErrMsg, sql);
  sqlite3_free(sql);
  return rc;
}

static int hnsw_optimize(HnswVtab *vtab) {
  HnswState state;
  char *rows = shadow_table(vtab, "rows");
  char *nodes = shadow_table(vtab, "nodes");
  char *edges = shadow_table(vtab, "edges");
  char *rebuild = shadow_table(vtab, "rebuild");
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (rows == NULL || nodes == NULL || edges == NULL || rebuild == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = load_state(vtab, &state);
  if (rc != SQLITE_OK) {
    goto done;
  }
  if (state.next_node_id <= 0 || state.live_count < 0 ||
      state.tombstone_count < 0 ||
      state.live_count > state.next_node_id - 1 ||
      state.tombstone_count >
          state.next_node_id - 1 - state.live_count) {
    rc = SQLITE_CORRUPT_VTAB;
    goto done;
  }
  if (state.tombstone_count == 0 &&
      state.next_node_id - 1 == state.live_count) {
    goto done;
  }
  cache_clear_all(vtab);
  vtab->cache_seq = -1;
  sql = sqlite3_mprintf(
      "DELETE FROM %s;INSERT INTO %s(rowid,vector) "
      "SELECT r.rowid,n.vector FROM %s AS r JOIN %s AS n ON "
      "n.node_id=r.node_id ORDER BY r.rowid;DELETE FROM %s;DELETE FROM %s;"
      "DELETE FROM %s",
      rebuild, rebuild, rows, nodes, rows, edges, nodes);
  if (sql == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = exec_sql(vtab->db, &vtab->base.zErrMsg, sql);
  sqlite3_free(sql);
  sql = NULL;
  if (rc != SQLITE_OK) {
    goto done;
  }
  state.entry_node = -1;
  state.max_level = -1;
  state.next_node_id = 1;
  state.live_count = 0;
  state.tombstone_count = 0;
  ++state.change_seq;
  rc = save_state(vtab, &state);
  if (rc != SQLITE_OK) {
    goto done;
  }
  sql = sqlite3_mprintf("SELECT rowid,vector FROM %s ORDER BY rowid", rebuild);
  if (sql == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = sqlite3_prepare_v2(vtab->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  sql = NULL;
  if (rc != SQLITE_OK) {
    goto done;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 rowid = sqlite3_column_int64(statement, 0);
    const void *blob = sqlite3_column_blob(statement, 1);
    int bytes = sqlite3_column_bytes(statement, 1);
    float *vector = NULL;
    int dims = 0;
    char *error = NULL;
    rc = hnsw_vector_decode(blob, bytes, &vector, &dims, &error);
    sqlite3_free(error);
    if (rc == SQLITE_OK && dims == vtab->dims) {
      rc = stage_node(vtab, rowid, blob, bytes,
                      hnsw_vector_norm(vector, dims));
    } else if (rc == SQLITE_OK) {
      rc = SQLITE_CORRUPT_VTAB;
    }
    sqlite3_free(vector);
    if (rc != SQLITE_OK || sqlite3_is_interrupted(vtab->db)) {
      if (rc == SQLITE_OK) {
        rc = SQLITE_INTERRUPT;
      }
      break;
    }
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (sqlite3_finalize(statement) != SQLITE_OK && rc == SQLITE_OK) {
    rc = sqlite3_errcode(vtab->db);
  }
  statement = NULL;
  if (rc == SQLITE_OK) {
    rc = link_staged_nodes(vtab);
  }
  if (rc == SQLITE_OK) {
    rc = clear_rebuild_table(vtab);
  }
done:
  if (statement != NULL) {
    sqlite3_finalize(statement);
  }
  sqlite3_free(sql);
  sqlite3_free(rows);
  sqlite3_free(nodes);
  sqlite3_free(edges);
  sqlite3_free(rebuild);
  return rc;
}

static int hnsw_update(sqlite3_vtab *base, int argc, sqlite3_value **argv,
                       sqlite3_int64 *output_rowid) {
  HnswVtab *vtab = (HnswVtab *)base;
  sqlite3_int64 rowid = 0;
  int rc = SQLITE_OK;
  rc = sync_node_cache(vtab);
  if (rc != SQLITE_OK) {
    return rc;
  }
  vtab->write_dirty = 1;
  query_snapshot_clear(vtab);
  if (argc == 1) {
    return hnsw_delete_internal(vtab, sqlite3_value_int64(argv[0]));
  }
  if (argc != HNSW_COLUMN_COUNT + 2) {
    return SQLITE_MISUSE;
  }
  if (sqlite3_value_type(argv[2 + COL_COMMAND]) != SQLITE_NULL) {
    const char *command =
        (const char *)sqlite3_value_text(argv[2 + COL_COMMAND]);
    if (sqlite3_value_type(argv[0]) != SQLITE_NULL || command == NULL ||
        !ascii_equal(command, "optimize")) {
      base->zErrMsg = sqlite3_mprintf("the only supported command is optimize");
      return SQLITE_CONSTRAINT;
    }
    *output_rowid = 0;
    return hnsw_optimize(vtab);
  }
  if (sqlite3_value_type(argv[2 + COL_VECTOR]) != SQLITE_BLOB) {
    base->zErrMsg = sqlite3_mprintf("embedding must be a float32 BLOB");
    return SQLITE_CONSTRAINT;
  }
  if (sqlite3_value_type(argv[1]) == SQLITE_NULL) {
    rc = next_external_rowid(vtab, &rowid);
  } else {
    rowid = sqlite3_value_int64(argv[1]);
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (sqlite3_value_type(argv[0]) == SQLITE_NULL ||
      sqlite3_value_int64(argv[0]) != rowid) {
    sqlite3_int64 existing_node = 0;
    int lookup_rc = lookup_node_for_rowid(vtab, rowid, &existing_node);
    if (lookup_rc == SQLITE_OK) {
      int conflict = sqlite3_vtab_on_conflict(vtab->db);
      if (conflict == SQLITE_IGNORE) {
        *output_rowid = rowid;
        return SQLITE_OK;
      }
      if (conflict == SQLITE_REPLACE) {
        rc = hnsw_delete_internal(vtab, rowid);
      } else {
        base->zErrMsg =
            sqlite3_mprintf("rowid %lld already exists", (long long)rowid);
        return SQLITE_CONSTRAINT;
      }
    } else if (lookup_rc != SQLITE_NOTFOUND) {
      return lookup_rc;
    }
  }
  if (rc != SQLITE_OK) {
    return rc;
  }
  {
    const void *blob = sqlite3_value_blob(argv[2 + COL_VECTOR]);
    int bytes = sqlite3_value_bytes(argv[2 + COL_VECTOR]);
    float *vector = NULL;
    int dims = 0;
    char *error = NULL;
    rc = hnsw_vector_validate(blob, bytes, vtab->dims, vtab->metric, &error);
    if (rc != SQLITE_OK) {
      base->zErrMsg = error;
      return rc;
    }
    rc = hnsw_vector_decode(blob, bytes, &vector, &dims, &error);
    sqlite3_free(error);
    if (rc != SQLITE_OK) {
      sqlite3_free(vector);
      return rc;
    }
    if (sqlite3_value_type(argv[0]) != SQLITE_NULL) {
      rc = hnsw_delete_internal(vtab, sqlite3_value_int64(argv[0]));
    }
    if (rc == SQLITE_OK) {
      rc = stage_node(vtab, rowid, blob, bytes,
                      hnsw_vector_norm(vector, dims));
    }
    sqlite3_free(vector);
  }
  if (rc == SQLITE_OK) {
    *output_rowid = rowid;
  }
  return rc;
}

static int hnsw_rename(sqlite3_vtab *base, const char *new_name) {
  HnswVtab *vtab = (HnswVtab *)base;
  static const char *suffixes[] = {"meta", "rows", "nodes", "edges", "rebuild"};
  char *sql = sqlite3_mprintf("");
  int i = 0;
  int rc = SQLITE_OK;
  query_snapshot_clear(vtab);
  finalize_cached_statements(vtab);
  for (i = 0; i < 5 && sql != NULL; ++i) {
    char *old_table = shadow_table(vtab, suffixes[i]);
    char *new_table = sqlite3_mprintf("%s_%s", new_name, suffixes[i]);
    char *next = NULL;
    if (old_table == NULL || new_table == NULL) {
      sqlite3_free(old_table);
      sqlite3_free(new_table);
      sqlite3_free(sql);
      sql = NULL;
      break;
    }
    next = sqlite3_mprintf("%zALTER TABLE %s RENAME TO \"%w\";", sql, old_table,
                           new_table);
    sqlite3_free(old_table);
    sqlite3_free(new_table);
    sql = next;
  }
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = exec_sql(vtab->db, &base->zErrMsg, sql);
  sqlite3_free(sql);
  if (rc == SQLITE_OK) {
    char *copy = sqlite3_mprintf("%s", new_name);
    if (copy == NULL) {
      return SQLITE_NOMEM;
    }
    sqlite3_free(vtab->name);
    vtab->name = copy;
  }
  return rc;
}

static int hnsw_begin(sqlite3_vtab *base) {
  HnswVtab *vtab = (HnswVtab *)base;
  vtab->write_dirty = 1;
  query_snapshot_clear(vtab);
  return SQLITE_OK;
}

static int hnsw_sync(sqlite3_vtab *base) {
  return link_staged_nodes((HnswVtab *)base);
}

static int hnsw_commit(sqlite3_vtab *base) {
  HnswVtab *vtab = (HnswVtab *)base;
  vtab->write_dirty = 0;
  query_snapshot_clear(vtab);
  finalize_cached_statements(vtab);
  return SQLITE_OK;
}

static int hnsw_rollback(sqlite3_vtab *base) {
  HnswVtab *vtab = (HnswVtab *)base;
  vtab->defer_edge_writes = 0;
  vtab->write_dirty = 0;
  finalize_cached_statements(vtab);
  cache_clear_all(vtab);
  query_snapshot_clear(vtab);
  vtab->cache_seq = -1;
  return SQLITE_OK;
}

static int hnsw_savepoint(sqlite3_vtab *base, int savepoint) {
  (void)base;
  (void)savepoint;
  return SQLITE_OK;
}

static int hnsw_release(sqlite3_vtab *base, int savepoint) {
  (void)base;
  (void)savepoint;
  return SQLITE_OK;
}

static int hnsw_rollback_to(sqlite3_vtab *base, int savepoint) {
  HnswVtab *vtab = (HnswVtab *)base;
  vtab->defer_edge_writes = 0;
  vtab->write_dirty = 1;
  cache_clear_all(vtab);
  query_snapshot_clear(vtab);
  vtab->cache_seq = -1;
  (void)savepoint;
  return SQLITE_OK;
}

static int hnsw_shadow_name(const char *name) {
  return ascii_equal(name, "meta") || ascii_equal(name, "rows") ||
         ascii_equal(name, "nodes") || ascii_equal(name, "edges") ||
         ascii_equal(name, "rebuild");
}

static void hnsw_sql_version(sqlite3_context *ctx, int argc,
                             sqlite3_value **argv) {
  (void)argc;
  (void)argv;
  sqlite3_result_text(ctx, SQLITE_HNSW_VERSION, -1, SQLITE_STATIC);
}

static int scalar_table_name(sqlite3_context *ctx, sqlite3_value **argv,
                             char **out) {
  const char *schema = NULL;
  const char *table = NULL;
  if (sqlite3_value_type(argv[0]) != SQLITE_TEXT ||
      sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
    sqlite3_result_error(ctx, "schema and table names must be text", -1);
    return SQLITE_MISMATCH;
  }
  schema = (const char *)sqlite3_value_text(argv[0]);
  table = (const char *)sqlite3_value_text(argv[1]);
  if (schema == NULL || table == NULL || *schema == '\0' || *table == '\0') {
    sqlite3_result_error(ctx, "schema and table names must not be empty", -1);
    return SQLITE_MISMATCH;
  }
  *out = sqlite3_mprintf("\"%w\".\"%w_meta\"", schema, table);
  if (*out == NULL) {
    sqlite3_result_error_nomem(ctx);
    return SQLITE_NOMEM;
  }
  return SQLITE_OK;
}

static void hnsw_sql_info(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  sqlite3 *db = (sqlite3 *)sqlite3_user_data(ctx);
  char *meta = NULL;
  char *nodes = NULL;
  char *sql = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  (void)argc;
  rc = scalar_table_name(ctx, argv, &meta);
  if (rc != SQLITE_OK) {
    return;
  }
  nodes = sqlite3_mprintf("\"%w\".\"%w_nodes\"", sqlite3_value_text(argv[0]),
                          sqlite3_value_text(argv[1]));
  if (nodes == NULL) {
    sqlite3_free(meta);
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sql = sqlite3_mprintf(
      "SELECT format_version,vector_name,dims,metric,m,ef_construction,"
      "cache_size_mb,live_count,tombstone_count,change_seq,next_node_id,"
      "(SELECT count(*) FROM %s WHERE level>=0),"
      "(SELECT coalesce(sum(CASE WHEN level>=0 THEN level ELSE 0 END),0) "
      "FROM %s),"
      "(SELECT count(*) FROM %s WHERE level<=-2 AND deleted=0) "
      "FROM %s WHERE id=1",
      nodes, nodes, nodes, meta);
  sqlite3_free(meta);
  sqlite3_free(nodes);
  if (sql == NULL) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
    return;
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    int metric = sqlite3_column_int(statement, 3);
    sqlite3_int64 live = sqlite3_column_int64(statement, 7);
    sqlite3_int64 tombstones = sqlite3_column_int64(statement, 8);
    sqlite3_int64 next_node_id = sqlite3_column_int64(statement, 10);
    sqlite3_int64 linked = sqlite3_column_int64(statement, 11);
    sqlite3_int64 level_sum = sqlite3_column_int64(statement, 12);
    sqlite3_int64 pending = sqlite3_column_int64(statement, 13);
    sqlite3_uint64 stored_nodes =
        live >= 0 && tombstones >= 0
            ? (sqlite3_uint64)live + (sqlite3_uint64)tombstones
            : 0;
    sqlite3_uint64 node_span =
        next_node_id > 0 ? (sqlite3_uint64)(next_node_id - 1) : 0;
    sqlite3_uint64 id_gaps =
        node_span > stored_nodes ? node_span - stored_nodes : 0;
    sqlite3_uint64 snapshot_bytes = 0;
    sqlite3_uint64 neighbor_slots = 0;
    sqlite3_uint64 count_slots = 0;
    sqlite3_uint64 cache_bytes =
        (sqlite3_uint64)sqlite3_column_int(statement, 6) * 1024U * 1024U;
    int snapshot_rc = SQLITE_TOOBIG;
    int snapshot_eligible = 0;
    double ratio = stored_nodes == 0
                       ? 0.0
                       : (double)tombstones / (double)stored_nodes;
    double id_gap_ratio =
        node_span == 0 ? 0.0 : (double)id_gaps / (double)node_span;
    if (next_node_id > 0 && (uint64_t)(next_node_id - 1) <= UINT32_MAX &&
        linked >= 0 && level_sum >= 0) {
      neighbor_slots =
          (sqlite3_uint64)linked *
              (sqlite3_uint64)(sqlite3_column_int(statement, 4) * 2) +
          (sqlite3_uint64)level_sum *
              (sqlite3_uint64)sqlite3_column_int(statement, 4);
      count_slots = (sqlite3_uint64)linked + (sqlite3_uint64)level_sum;
      if (neighbor_slots <= SIZE_MAX && count_slots <= SIZE_MAX) {
        snapshot_rc = query_snapshot_required_bytes(
            (size_t)(next_node_id - 1), sqlite3_column_int(statement, 2),
            (size_t)neighbor_slots, (size_t)count_slots, &snapshot_bytes);
      }
    }
    snapshot_eligible = snapshot_rc == SQLITE_OK && pending == 0 &&
                        snapshot_bytes <= cache_bytes;
    char *json = sqlite3_mprintf(
        "{\"format_version\":%d,\"vector_name\":\"%w\",\"dims\":%d,"
        "\"metric\":\"%s\",\"m\":%d,\"ef_construction\":%d,"
        "\"cache_size_mb\":%d,\"live_count\":%lld,"
        "\"tombstone_count\":%lld,\"tombstone_ratio\":%.9g,"
        "\"change_seq\":%lld,\"pending_count\":%lld,"
        "\"estimated_snapshot_bytes\":%llu,"
        "\"snapshot_cache_eligible\":%s,"
        "\"needs_optimize\":%s}",
        sqlite3_column_int(statement, 0), sqlite3_column_text(statement, 1),
        sqlite3_column_int(statement, 2),
        metric == HNSW_METRIC_L2
            ? "l2"
            : (metric == HNSW_METRIC_COSINE ? "cosine" : "ip"),
        sqlite3_column_int(statement, 4), sqlite3_column_int(statement, 5),
        sqlite3_column_int(statement, 6), (long long)live,
        (long long)tombstones, ratio,
        (long long)sqlite3_column_int64(statement, 9), (long long)pending,
        (unsigned long long)snapshot_bytes,
        snapshot_eligible ? "true" : "false",
        ratio >= 0.2 || id_gap_ratio >= 0.2 ? "true" : "false");
    if (json == NULL) {
      sqlite3_result_error_nomem(ctx);
    } else {
      sqlite3_result_text(ctx, json, -1, sqlite3_free);
    }
  } else if (rc == SQLITE_DONE) {
    sqlite3_result_error(ctx, "hnsw metadata row is missing", -1);
  } else {
    sqlite3_result_error(ctx, sqlite3_errmsg(db), -1);
  }
  sqlite3_finalize(statement);
}

static void hnsw_sql_check(sqlite3_context *ctx, int argc,
                           sqlite3_value **argv) {
  hnsw_integrity_check(ctx, argc, argv);
}

static int register_extension(sqlite3 *db, char **error) {
  int deterministic = SQLITE_UTF8 | SQLITE_DETERMINISTIC | SQLITE_INNOCUOUS;
  int rc = sqlite3_create_module(db, "hnsw", &hnsw_module, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_f32", 1, deterministic, NULL,
                                 hnsw_sql_f32, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_dims", 1, deterministic, NULL,
                                 hnsw_sql_dims, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_distance_l2", 2, deterministic, NULL,
                                 hnsw_sql_distance_l2, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_distance_cosine", 2, deterministic,
                                 NULL, hnsw_sql_distance_cosine, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_distance_ip", 2, deterministic, NULL,
                                 hnsw_sql_distance_ip, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_version", 0, deterministic, NULL,
                                 hnsw_sql_version, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_info", 2,
                                 SQLITE_UTF8 | SQLITE_DIRECTONLY, db,
                                 hnsw_sql_info, NULL, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_create_function(db, "hnsw_check", 2,
                                 SQLITE_UTF8 | SQLITE_DIRECTONLY, db,
                                 hnsw_sql_check, NULL, NULL);
  }
  if (rc != SQLITE_OK && error != NULL) {
    *error = sqlite3_mprintf("failed to register sqlite-hnsw: %s",
                             sqlite3_errmsg(db));
  }
  return rc;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_hnsw_init(sqlite3 *db, char **error,
                          const sqlite3_api_routines *api) {
  SQLITE_EXTENSION_INIT2(api);
  return register_extension(db, error);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_sqlitehnsw_init(sqlite3 *db, char **error,
                                const sqlite3_api_routines *api) {
  SQLITE_EXTENSION_INIT2(api);
  return register_extension(db, error);
}

#ifdef _WIN32
__declspec(dllexport)
#endif
    int sqlite3_extension_init(sqlite3 *db, char **error,
                               const sqlite3_api_routines *api) {
  SQLITE_EXTENSION_INIT2(api);
  return register_extension(db, error);
}
