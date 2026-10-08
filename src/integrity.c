#include "integrity.h"
SQLITE_EXTENSION_INIT3

#include "vector.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define HNSW_FORMAT_VERSION 1
#define HNSW_MAX_DIMENSIONS 4096
#define HNSW_MAX_LEVEL 32

typedef struct CheckError {
  const char *code;
  sqlite3_int64 node_id;
  sqlite3_int64 neighbor_id;
  int layer;
  unsigned char has_node;
  unsigned char has_neighbor;
  unsigned char has_layer;
} CheckError;

typedef struct CheckNode {
  sqlite3_int64 id;
  size_t layer0_offset;
  uint64_t edge_mask;
  int level;
  uint32_t layer0_count;
  unsigned char deleted;
  unsigned char has_rowid;
  unsigned char layer0_seen;
} CheckNode;

typedef struct CheckMapSlot {
  sqlite3_int64 id;
  size_t index_plus_one;
} CheckMapSlot;

typedef struct CheckState {
  int format_version;
  int dims;
  int metric;
  int m;
  int ef_construction;
  int cache_size_mb;
  sqlite3_int64 entry_node;
  int max_level;
  sqlite3_int64 next_node_id;
  sqlite3_int64 live_count;
  sqlite3_int64 tombstone_count;
  sqlite3_int64 change_seq;
  int usable;
} CheckState;

typedef struct CheckRun {
  sqlite3 *db;
  CheckState state;
  CheckError error;
  CheckNode *nodes;
  size_t node_count;
  size_t node_capacity;
  CheckMapSlot *map;
  size_t map_capacity;
  uint32_t *layer0;
  size_t layer0_count;
  size_t layer0_capacity;
  sqlite3_int64 mappings;
  sqlite3_int64 active_nodes;
  sqlite3_int64 deleted_nodes;
  sqlite3_int64 pending_count;
  sqlite3_int64 orphan_edges;
  sqlite3_int64 checked_edge_rows;
  sqlite3_int64 checked_neighbor_ids;
  sqlite3_int64 reachable_nodes;
  int reachable_available;
  int graph_safe;
} CheckRun;

static uint32_t read_u32_le_check(const unsigned char *p) {
  return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8U) |
         ((uint32_t)p[2] << 16U) | ((uint32_t)p[3] << 24U);
}

static uint64_t read_u64_le_check(const unsigned char *p) {
  uint64_t value = 0;
  int i = 0;
  for (i = 7; i >= 0; --i) {
    value = (value << 8U) | p[i];
  }
  return value;
}

static size_t check_hash(sqlite3_int64 id) {
  uint64_t value = (uint64_t)id;
  value ^= value >> 30U;
  value *= UINT64_C(0xbf58476d1ce4e5b9);
  value ^= value >> 27U;
  value *= UINT64_C(0x94d049bb133111eb);
  value ^= value >> 31U;
  return (size_t)value;
}

static void set_error(CheckRun *run, const char *code, int has_node,
                      sqlite3_int64 node_id, int has_layer, int layer,
                      int has_neighbor, sqlite3_int64 neighbor_id) {
  if (run->error.code != NULL) {
    return;
  }
  run->error.code = code;
  run->error.has_node = (unsigned char)has_node;
  run->error.node_id = node_id;
  run->error.has_layer = (unsigned char)has_layer;
  run->error.layer = layer;
  run->error.has_neighbor = (unsigned char)has_neighbor;
  run->error.neighbor_id = neighbor_id;
}

static int valid_identifier_check(const char *name) {
  const unsigned char *p = (const unsigned char *)name;
  if (p == NULL || *p == '\0' || !(isalpha(*p) || *p == '_')) {
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

static int grow_nodes(CheckRun *run) {
  size_t next = run->node_capacity == 0 ? 1024U : run->node_capacity * 2U;
  CheckNode *grown = NULL;
  if (next < run->node_capacity ||
      (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(*grown)) {
    return SQLITE_TOOBIG;
  }
  grown = sqlite3_realloc64(run->nodes,
                            (sqlite3_uint64)next * sizeof(*grown));
  if (grown == NULL) {
    return SQLITE_NOMEM;
  }
  run->nodes = grown;
  run->node_capacity = next;
  return SQLITE_OK;
}

static int append_node(CheckRun *run, const CheckNode *node) {
  int rc = SQLITE_OK;
  if (run->node_count == run->node_capacity) {
    rc = grow_nodes(run);
    if (rc != SQLITE_OK) {
      return rc;
    }
  }
  run->nodes[run->node_count++] = *node;
  return SQLITE_OK;
}

static int build_node_map(CheckRun *run) {
  size_t capacity = 16U;
  size_t i = 0;
  if (run->node_count > SIZE_MAX / 2U) {
    return SQLITE_TOOBIG;
  }
  while (capacity < run->node_count * 2U) {
    if (capacity > SIZE_MAX / 2U) {
      return SQLITE_TOOBIG;
    }
    capacity *= 2U;
  }
  if ((sqlite3_uint64)capacity >
      HNSW_ALLOCATION_LIMIT / sizeof(*run->map)) {
    return SQLITE_TOOBIG;
  }
  run->map = sqlite3_malloc64((sqlite3_uint64)capacity * sizeof(*run->map));
  if (run->map == NULL) {
    return SQLITE_NOMEM;
  }
  memset(run->map, 0, capacity * sizeof(*run->map));
  run->map_capacity = capacity;
  for (i = 0; i < run->node_count; ++i) {
    size_t slot = check_hash(run->nodes[i].id) & (capacity - 1U);
    while (run->map[slot].index_plus_one != 0) {
      if (run->map[slot].id == run->nodes[i].id) {
        set_error(run, "NODE_ID_INVALID", 1, run->nodes[i].id, 0, 0, 0,
                  0);
        break;
      }
      slot = (slot + 1U) & (capacity - 1U);
    }
    if (run->map[slot].index_plus_one == 0) {
      run->map[slot].id = run->nodes[i].id;
      run->map[slot].index_plus_one = i + 1U;
    }
  }
  return SQLITE_OK;
}

static size_t find_node(const CheckRun *run, sqlite3_int64 id) {
  size_t slot = 0;
  if (run->map_capacity == 0) {
    return SIZE_MAX;
  }
  slot = check_hash(id) & (run->map_capacity - 1U);
  while (run->map[slot].index_plus_one != 0) {
    if (run->map[slot].id == id) {
      return run->map[slot].index_plus_one - 1U;
    }
    slot = (slot + 1U) & (run->map_capacity - 1U);
  }
  return SIZE_MAX;
}

static int append_layer0(CheckRun *run, size_t index) {
  uint32_t *grown = NULL;
  size_t next = 0;
  if (index > UINT32_MAX) {
    run->graph_safe = 0;
    return SQLITE_OK;
  }
  if (run->layer0_count < run->layer0_capacity) {
    run->layer0[run->layer0_count++] = (uint32_t)index;
    return SQLITE_OK;
  }
  next = run->layer0_capacity == 0 ? 4096U : run->layer0_capacity * 2U;
  if (next < run->layer0_capacity ||
      (sqlite3_uint64)next > HNSW_ALLOCATION_LIMIT / sizeof(*grown)) {
    return SQLITE_TOOBIG;
  }
  grown = sqlite3_realloc64(run->layer0,
                            (sqlite3_uint64)next * sizeof(*grown));
  if (grown == NULL) {
    return SQLITE_NOMEM;
  }
  run->layer0 = grown;
  run->layer0_capacity = next;
  run->layer0[run->layer0_count++] = (uint32_t)index;
  return SQLITE_OK;
}

static int finish_statement(sqlite3 *db, sqlite3_stmt *statement, int rc) {
  int finalize_rc = sqlite3_finalize(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  if (rc == SQLITE_OK && finalize_rc != SQLITE_OK) {
    rc = sqlite3_errcode(db);
  }
  return rc;
}

static int load_metadata(CheckRun *run, const char *meta) {
  char *sql = sqlite3_mprintf(
      "SELECT format_version,vector_name,dims,metric,m,ef_construction,"
      "cache_size_mb,entry_node,max_level,next_node_id,"
      "live_count,tombstone_count,change_seq FROM %s WHERE id=1",
      meta);
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  int types_valid = 0;
  const unsigned char *vector_name = NULL;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(run->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    set_error(run, "META_MISSING", 0, 0, 0, 0, 0, 0);
    return finish_statement(run->db, statement, SQLITE_OK);
  }
  if (rc != SQLITE_ROW) {
    return finish_statement(run->db, statement, rc);
  }
  run->state.format_version = sqlite3_column_int(statement, 0);
  vector_name = sqlite3_column_text(statement, 1);
  run->state.dims = sqlite3_column_int(statement, 2);
  run->state.metric = sqlite3_column_int(statement, 3);
  run->state.m = sqlite3_column_int(statement, 4);
  run->state.ef_construction = sqlite3_column_int(statement, 5);
  run->state.cache_size_mb = sqlite3_column_int(statement, 6);
  run->state.entry_node = sqlite3_column_int64(statement, 7);
  run->state.max_level = sqlite3_column_int(statement, 8);
  run->state.next_node_id = sqlite3_column_int64(statement, 9);
  run->state.live_count = sqlite3_column_int64(statement, 10);
  run->state.tombstone_count = sqlite3_column_int64(statement, 11);
  run->state.change_seq = sqlite3_column_int64(statement, 12);
  types_valid =
      sqlite3_column_type(statement, 0) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 1) == SQLITE_TEXT &&
      sqlite3_column_type(statement, 2) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 3) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 4) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 5) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 6) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 7) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 8) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 9) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 10) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 11) == SQLITE_INTEGER &&
      sqlite3_column_type(statement, 12) == SQLITE_INTEGER;
  run->state.usable =
      types_valid &&
      run->state.format_version == HNSW_FORMAT_VERSION &&
      valid_identifier_check((const char *)vector_name) &&
      run->state.dims >= 1 && run->state.dims <= HNSW_MAX_DIMENSIONS &&
      run->state.metric >= HNSW_METRIC_L2 &&
      run->state.metric <= HNSW_METRIC_INNER_PRODUCT &&
      run->state.m >= 2 && run->state.m <= 64 &&
      run->state.ef_construction >= 4 &&
      run->state.ef_construction <= 1000 &&
      run->state.ef_construction >= run->state.m * 2 &&
      run->state.cache_size_mb >= 0 && run->state.cache_size_mb <= 65536;
  if (!run->state.usable) {
    set_error(run, "META_INVALID", 0, 0, 0, 0, 0, 0);
  } else if (run->state.next_node_id <= 0 || run->state.live_count < 0 ||
             run->state.tombstone_count < 0 || run->state.change_seq < 0) {
    set_error(run, "STATE_COUNT_MISMATCH", 0, 0, 0, 0, 0, 0);
  } else if ((run->state.entry_node != -1 && run->state.entry_node <= 0) ||
             run->state.max_level < -1 ||
             run->state.max_level > HNSW_MAX_LEVEL) {
    set_error(run, "STATE_ENTRY_INVALID", 0, 0, 0, 0, 0, 0);
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  return finish_statement(run->db, statement, rc);
}

static int scan_nodes(CheckRun *run, const char *nodes) {
  char *sql = sqlite3_mprintf(
      "SELECT node_id,level,vector,norm,deleted FROM %s ORDER BY node_id",
      nodes);
  sqlite3_stmt *statement = NULL;
  float *scratch = NULL;
  sqlite3_int64 previous_id = 0;
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(run->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  if (run->state.usable) {
    scratch = sqlite3_malloc64((sqlite3_uint64)run->state.dims *
                               sizeof(*scratch));
    if (scratch == NULL) {
      sqlite3_finalize(statement);
      return SQLITE_NOMEM;
    }
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    CheckNode node;
    int raw_level = sqlite3_column_int(statement, 1);
    int deleted = sqlite3_column_int(statement, 4);
    memset(&node, 0, sizeof(node));
    node.id = sqlite3_column_int64(statement, 0);
    node.level = raw_level;
    node.deleted = (unsigned char)(deleted == 1);
    if (sqlite3_column_type(statement, 0) != SQLITE_INTEGER || node.id <= 0 ||
        (run->node_count > 0 && node.id <= previous_id) ||
        (run->state.usable && node.id >= run->state.next_node_id)) {
      set_error(run, "NODE_ID_INVALID", 1, node.id, 0, 0, 0, 0);
    }
    previous_id = node.id;
    if (sqlite3_column_type(statement, 1) != SQLITE_INTEGER ||
        !((raw_level >= 0 && raw_level <= HNSW_MAX_LEVEL) ||
          (raw_level <= -2 && raw_level >= -HNSW_MAX_LEVEL - 2)) ||
        (deleted == 1 && raw_level < 0)) {
      set_error(run, "NODE_LEVEL_INVALID", 1, node.id, 0, 0, 0, 0);
    } else if (raw_level <= -2) {
      ++run->pending_count;
      set_error(run, "PENDING_NODE", 1, node.id, 0, 0, 0, 0);
    }
    if (sqlite3_column_type(statement, 4) != SQLITE_INTEGER ||
        (deleted != 0 && deleted != 1)) {
      set_error(run, "NODE_LEVEL_INVALID", 1, node.id, 0, 0, 0, 0);
    }
    if (deleted == 1) {
      ++run->deleted_nodes;
    } else {
      ++run->active_nodes;
    }
    if (run->state.usable) {
      const void *blob = sqlite3_column_blob(statement, 2);
      int bytes = sqlite3_column_bytes(statement, 2);
      char *message = NULL;
      int vector_rc = SQLITE_OK;
      if (sqlite3_column_type(statement, 2) != SQLITE_BLOB) {
        vector_rc = SQLITE_CORRUPT_VTAB;
      } else {
        vector_rc = hnsw_vector_validate(blob, bytes, run->state.dims,
                                         run->state.metric, &message);
      }
      sqlite3_free(message);
      if (vector_rc != SQLITE_OK ||
          hnsw_vector_decode_into(blob, bytes, scratch, run->state.dims,
                                  NULL) != SQLITE_OK) {
        set_error(run, "NODE_VECTOR_INVALID", 1, node.id, 0, 0, 0, 0);
      } else {
        double stored = sqlite3_column_double(statement, 3);
        double computed = hnsw_vector_norm(scratch, run->state.dims);
        double tolerance = fmax(1e-12, fabs(computed) * 1e-12);
        if ((sqlite3_column_type(statement, 3) != SQLITE_FLOAT &&
             sqlite3_column_type(statement, 3) != SQLITE_INTEGER) ||
            !isfinite(stored) || stored < 0.0 ||
            fabs(stored - computed) > tolerance) {
          set_error(run, "NODE_NORM_INVALID", 1, node.id, 0, 0, 0, 0);
        }
      }
    }
    rc = append_node(run, &node);
    if (rc != SQLITE_OK || sqlite3_is_interrupted(run->db)) {
      if (rc == SQLITE_OK) {
        rc = SQLITE_INTERRUPT;
      }
      break;
    }
  }
  sqlite3_free(scratch);
  return finish_statement(run->db, statement, rc);
}

static int scan_rows(CheckRun *run, const char *rows) {
  char *sql = sqlite3_mprintf("SELECT rowid,node_id FROM %s ORDER BY node_id",
                              rows);
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(run->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 node_id = sqlite3_column_int64(statement, 1);
    size_t index = find_node(run, node_id);
    ++run->mappings;
    if (sqlite3_column_type(statement, 0) != SQLITE_INTEGER ||
        sqlite3_column_type(statement, 1) != SQLITE_INTEGER ||
        index == SIZE_MAX) {
      set_error(run, "ROW_NODE_MISSING", 1, node_id, 0, 0, 0, 0);
    } else if (run->nodes[index].deleted) {
      set_error(run, "ROW_NODE_DELETED", 1, node_id, 0, 0, 0, 0);
    } else if (run->nodes[index].has_rowid) {
      set_error(run, "STATE_COUNT_MISMATCH", 1, node_id, 0, 0, 0, 0);
    } else {
      run->nodes[index].has_rowid = 1;
    }
    if (sqlite3_is_interrupted(run->db)) {
      rc = SQLITE_INTERRUPT;
      break;
    }
  }
  return finish_statement(run->db, statement, rc);
}

static void check_state(CheckRun *run) {
  sqlite3_int64 max_node_id = 0;
  int maximum_level = -1;
  size_t linked = 0;
  size_t i = 0;
  for (i = 0; i < run->node_count; ++i) {
    CheckNode *node = &run->nodes[i];
    if (node->id > max_node_id) {
      max_node_id = node->id;
    }
    if (!node->deleted && !node->has_rowid) {
      set_error(run, "LIVE_NODE_UNMAPPED", 1, node->id, 0, 0, 0, 0);
    }
    if (node->level >= 0 && node->level <= HNSW_MAX_LEVEL) {
      ++linked;
      if (node->level > maximum_level) {
        maximum_level = node->level;
      }
    }
  }
  if (!run->state.usable) {
    return;
  }
  if (run->state.next_node_id <= max_node_id ||
      run->state.live_count != run->mappings ||
      run->state.live_count != run->active_nodes ||
      run->state.tombstone_count != run->deleted_nodes) {
    set_error(run, "STATE_COUNT_MISMATCH", 0, 0, 0, 0, 0, 0);
  }
  if (run->state.live_count == 0) {
    if (run->node_count != 0 || run->state.entry_node != -1 ||
        run->state.max_level != -1 || run->state.tombstone_count != 0) {
      set_error(run, "STATE_ENTRY_INVALID", 0, 0, 0, 0, 0, 0);
    }
  } else if (run->node_count == 0) {
    if (run->state.entry_node != -1 || run->state.max_level != -1 ||
        run->state.live_count != 0) {
      set_error(run, "STATE_ENTRY_INVALID", 0, 0, 0, 0, 0, 0);
    }
  } else if (linked > 0) {
    size_t entry = find_node(run, run->state.entry_node);
    if (entry == SIZE_MAX || run->nodes[entry].level < 0 ||
        run->nodes[entry].level != run->state.max_level ||
        maximum_level != run->state.max_level) {
      set_error(run, "STATE_ENTRY_INVALID", 1, run->state.entry_node, 0, 0,
                0, 0);
    }
  } else if (run->pending_count == 0) {
    set_error(run, "STATE_ENTRY_INVALID", 0, 0, 0, 0, 0, 0);
  }
}

static int scan_edges(CheckRun *run, const char *edges) {
  char *sql = sqlite3_mprintf(
      "SELECT node_id,layer,neighbors FROM %s ORDER BY node_id,layer", edges);
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(run->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 owner_id = sqlite3_column_int64(statement, 0);
    int layer = sqlite3_column_int(statement, 1);
    const unsigned char *blob = sqlite3_column_blob(statement, 2);
    int bytes = sqlite3_column_bytes(statement, 2);
    size_t owner_index = find_node(run, owner_id);
    CheckNode *owner = owner_index == SIZE_MAX ? NULL : &run->nodes[owner_index];
    uint32_t count = 0;
    uint32_t i = 0;
    int blob_valid = 1;
    int owner_layer_valid = 1;
    ++run->checked_edge_rows;
    if (sqlite3_column_type(statement, 0) != SQLITE_INTEGER || owner == NULL) {
      ++run->orphan_edges;
      run->graph_safe = 0;
      set_error(run, "EDGE_OWNER_MISSING", 1, owner_id, 1, layer, 0, 0);
      owner_layer_valid = 0;
    }
    if (sqlite3_column_type(statement, 1) != SQLITE_INTEGER || owner == NULL ||
        owner->level < 0 || layer < 0 || layer > owner->level ||
        layer > HNSW_MAX_LEVEL) {
      run->graph_safe = 0;
      set_error(run, "EDGE_LAYER_INVALID", 1, owner_id, 1, layer, 0, 0);
      owner_layer_valid = 0;
    } else {
      uint64_t bit = UINT64_C(1) << (unsigned)layer;
      if ((owner->edge_mask & bit) != 0) {
        run->graph_safe = 0;
        set_error(run, "EDGE_LAYER_INVALID", 1, owner_id, 1, layer, 0, 0);
      }
      owner->edge_mask |= bit;
      if (layer == 0) {
        owner->layer0_seen = 1;
        owner->layer0_offset = run->layer0_count;
      }
    }
    if (sqlite3_column_type(statement, 2) != SQLITE_BLOB || blob == NULL ||
        bytes < 4) {
      blob_valid = 0;
    } else {
      count = read_u32_le_check(blob);
      if (count > 1000000U ||
          (sqlite3_uint64)bytes != 4U + (sqlite3_uint64)count * 8U) {
        blob_valid = 0;
      }
    }
    if (!blob_valid) {
      run->graph_safe = 0;
      set_error(run, "EDGE_BLOB_INVALID", 1, owner_id, 1, layer, 0, 0);
      continue;
    }
    if (!owner_layer_valid) {
      continue;
    }
    if (owner != NULL) {
      uint32_t maximum = (uint32_t)(layer == 0 ? run->state.m * 2
                                               : run->state.m);
      if (count > maximum) {
        run->graph_safe = 0;
        set_error(run, "EDGE_DEGREE_EXCEEDED", 1, owner_id, 1, layer, 0, 0);
        continue;
      }
    }
    for (i = 0; i < count; ++i) {
      uint64_t raw = read_u64_le_check(blob + 4U + (size_t)i * 8U);
      sqlite3_int64 neighbor_id =
          raw <= (uint64_t)INT64_MAX ? (sqlite3_int64)raw : -1;
      size_t neighbor_index =
          neighbor_id > 0 ? find_node(run, neighbor_id) : SIZE_MAX;
      uint32_t j = 0;
      int duplicate = 0;
      ++run->checked_neighbor_ids;
      for (j = 0; j < i; ++j) {
        if (read_u64_le_check(blob + 4U + (size_t)j * 8U) == raw) {
          duplicate = 1;
          break;
        }
      }
      if (neighbor_id == owner_id) {
        run->graph_safe = 0;
        set_error(run, "EDGE_SELF_REFERENCE", 1, owner_id, 1, layer, 1,
                  neighbor_id);
      } else if (duplicate) {
        run->graph_safe = 0;
        set_error(run, "EDGE_DUPLICATE_NEIGHBOR", 1, owner_id, 1, layer, 1,
                  neighbor_id);
      } else if (neighbor_index == SIZE_MAX) {
        run->graph_safe = 0;
        set_error(run, "EDGE_NEIGHBOR_MISSING", 1, owner_id, 1, layer, 1,
                  neighbor_id);
      } else if (run->nodes[neighbor_index].level < layer) {
        run->graph_safe = 0;
        set_error(run, "EDGE_NEIGHBOR_LAYER_INVALID", 1, owner_id, 1, layer,
                  1, neighbor_id);
      } else if (layer == 0 && owner != NULL && owner->level >= 0) {
        rc = append_layer0(run, neighbor_index);
        if (rc != SQLITE_OK) {
          break;
        }
        ++owner->layer0_count;
      }
    }
    if (rc != SQLITE_ROW && rc != SQLITE_OK) {
      break;
    }
    if (sqlite3_is_interrupted(run->db)) {
      rc = SQLITE_INTERRUPT;
      break;
    }
  }
  rc = finish_statement(run->db, statement, rc);
  if (rc == SQLITE_OK) {
    size_t i = 0;
    for (i = 0; i < run->node_count; ++i) {
      CheckNode *node = &run->nodes[i];
      if (node->level >= 0 && node->level <= HNSW_MAX_LEVEL) {
        uint64_t expected =
            (UINT64_C(1) << (unsigned)(node->level + 1)) - UINT64_C(1);
        if (node->edge_mask != expected || !node->layer0_seen) {
          run->graph_safe = 0;
          set_error(run, "EDGE_ROW_MISSING", 1, node->id, 0, 0, 0, 0);
        }
      }
    }
  }
  return rc;
}

static int check_rebuild(CheckRun *run, const char *rebuild) {
  char *sql = sqlite3_mprintf("SELECT count(*) FROM %s", rebuild);
  sqlite3_stmt *statement = NULL;
  int rc = SQLITE_OK;
  if (sql == NULL) {
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(run->db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  if (rc != SQLITE_OK) {
    return rc;
  }
  rc = sqlite3_step(statement);
  if (rc == SQLITE_ROW) {
    if (sqlite3_column_int64(statement, 0) != 0) {
      set_error(run, "REBUILD_NOT_EMPTY", 0, 0, 0, 0, 0, 0);
    }
    rc = SQLITE_OK;
  } else if (rc == SQLITE_DONE) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  return finish_statement(run->db, statement, rc);
}

static int compute_reachable(CheckRun *run) {
  unsigned char *visited = NULL;
  uint32_t *queue = NULL;
  size_t entry = SIZE_MAX;
  size_t head = 0;
  size_t tail = 0;
  if (!run->graph_safe || run->error.code != NULL ||
      run->node_count > UINT32_MAX) {
    return SQLITE_OK;
  }
  if (run->node_count == 0) {
    run->reachable_nodes = 0;
    run->reachable_available = 1;
    return SQLITE_OK;
  }
  if ((sqlite3_uint64)run->node_count >
      HNSW_ALLOCATION_LIMIT / sizeof(*queue)) {
    return SQLITE_TOOBIG;
  }
  entry = find_node(run, run->state.entry_node);
  if (entry == SIZE_MAX || run->nodes[entry].level < 0) {
    return SQLITE_OK;
  }
  visited = sqlite3_malloc64((sqlite3_uint64)run->node_count);
  queue = sqlite3_malloc64((sqlite3_uint64)run->node_count * sizeof(*queue));
  if (visited == NULL || queue == NULL) {
    sqlite3_free(visited);
    sqlite3_free(queue);
    return SQLITE_NOMEM;
  }
  memset(visited, 0, run->node_count);
  visited[entry] = 1;
  queue[tail++] = (uint32_t)entry;
  while (head < tail) {
    CheckNode *node = &run->nodes[queue[head++]];
    uint32_t i = 0;
    for (i = 0; i < node->layer0_count; ++i) {
      uint32_t next = run->layer0[node->layer0_offset + i];
      if (!visited[next]) {
        visited[next] = 1;
        queue[tail++] = next;
      }
    }
  }
  run->reachable_nodes = (sqlite3_int64)tail;
  run->reachable_available = 1;
  sqlite3_free(visited);
  sqlite3_free(queue);
  return SQLITE_OK;
}

static void result_check_json(sqlite3_context *ctx, const CheckRun *run) {
  sqlite3_str *json = sqlite3_str_new(run->db);
  char *result = NULL;
  if (json == NULL) {
    sqlite3_result_error_nomem(ctx);
    return;
  }
  sqlite3_str_appendf(
      json,
      "{\"ok\":%s,\"live_count\":%lld,\"mappings\":%lld,"
      "\"active_nodes\":%lld,\"tombstone_count\":%lld,"
      "\"deleted_nodes\":%lld,\"orphan_edge_rows\":%lld,"
      "\"pending_count\":%lld,\"checked_nodes\":%llu,"
      "\"checked_edge_rows\":%lld,\"checked_neighbor_ids\":%lld,"
      "\"reachable_nodes\":",
      run->error.code == NULL ? "true" : "false",
      (long long)run->state.live_count, (long long)run->mappings,
      (long long)run->active_nodes, (long long)run->state.tombstone_count,
      (long long)run->deleted_nodes, (long long)run->orphan_edges,
      (long long)run->pending_count, (unsigned long long)run->node_count,
      (long long)run->checked_edge_rows,
      (long long)run->checked_neighbor_ids);
  if (run->reachable_available) {
    sqlite3_str_appendf(json, "%lld", (long long)run->reachable_nodes);
  } else {
    sqlite3_str_appendall(json, "null");
  }
  sqlite3_str_appendall(json, ",\"first_error\":");
  if (run->error.code == NULL) {
    sqlite3_str_appendall(json, "null}");
  } else {
    sqlite3_str_appendf(json, "{\"code\":\"%s\",\"node_id\":",
                        run->error.code);
    if (run->error.has_node) {
      sqlite3_str_appendf(json, "%lld", (long long)run->error.node_id);
    } else {
      sqlite3_str_appendall(json, "null");
    }
    sqlite3_str_appendall(json, ",\"layer\":");
    if (run->error.has_layer) {
      sqlite3_str_appendf(json, "%d", run->error.layer);
    } else {
      sqlite3_str_appendall(json, "null");
    }
    sqlite3_str_appendall(json, ",\"neighbor_id\":");
    if (run->error.has_neighbor) {
      sqlite3_str_appendf(json, "%lld", (long long)run->error.neighbor_id);
    } else {
      sqlite3_str_appendall(json, "null");
    }
    sqlite3_str_appendall(json, "}}");
  }
  result = sqlite3_str_finish(json);
  if (result == NULL) {
    sqlite3_result_error_nomem(ctx);
  } else {
    sqlite3_result_text(ctx, result, -1, sqlite3_free);
  }
}

static void result_operational_error(sqlite3_context *ctx, sqlite3 *db,
                                     int rc) {
  if (rc == SQLITE_NOMEM) {
    sqlite3_result_error_nomem(ctx);
  } else {
    const char *message = sqlite3_errmsg(db);
    if (message == NULL || strcmp(message, "not an error") == 0) {
      message = sqlite3_errstr(rc);
    }
    sqlite3_result_error(ctx, message, -1);
    sqlite3_result_error_code(ctx, rc);
  }
}

void hnsw_integrity_check(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv) {
  CheckRun run;
  const char *schema = NULL;
  const char *name = NULL;
  char *meta = NULL;
  char *rows = NULL;
  char *nodes = NULL;
  char *edges = NULL;
  char *rebuild = NULL;
  int rc = SQLITE_OK;
  (void)argc;
  memset(&run, 0, sizeof(run));
  run.db = (sqlite3 *)sqlite3_user_data(ctx);
  run.graph_safe = 1;
  run.state.entry_node = -1;
  run.state.max_level = -1;
  if (sqlite3_value_type(argv[0]) != SQLITE_TEXT ||
      sqlite3_value_type(argv[1]) != SQLITE_TEXT) {
    sqlite3_result_error(ctx, "schema and table names must be text", -1);
    return;
  }
  schema = (const char *)sqlite3_value_text(argv[0]);
  name = (const char *)sqlite3_value_text(argv[1]);
  if (schema == NULL || name == NULL || *schema == '\0' || *name == '\0') {
    sqlite3_result_error(ctx, "schema and table names must not be empty", -1);
    return;
  }
  meta = sqlite3_mprintf("\"%w\".\"%w_meta\"", schema, name);
  rows = sqlite3_mprintf("\"%w\".\"%w_rows\"", schema, name);
  nodes = sqlite3_mprintf("\"%w\".\"%w_nodes\"", schema, name);
  edges = sqlite3_mprintf("\"%w\".\"%w_edges\"", schema, name);
  rebuild = sqlite3_mprintf("\"%w\".\"%w_rebuild\"", schema, name);
  if (meta == NULL || rows == NULL || nodes == NULL || edges == NULL ||
      rebuild == NULL) {
    rc = SQLITE_NOMEM;
    goto done;
  }
  rc = load_metadata(&run, meta);
  if (rc != SQLITE_OK || !run.state.usable) {
    if (rc == SQLITE_OK) {
      result_check_json(ctx, &run);
    }
    goto done;
  }
  rc = scan_nodes(&run, nodes);
  if (rc == SQLITE_OK) {
    rc = build_node_map(&run);
  }
  if (rc == SQLITE_OK) {
    rc = scan_rows(&run, rows);
  }
  if (rc == SQLITE_OK) {
    check_state(&run);
    rc = scan_edges(&run, edges);
  }
  if (rc == SQLITE_OK) {
    rc = check_rebuild(&run, rebuild);
  }
  if (rc == SQLITE_OK) {
    rc = compute_reachable(&run);
  }
  if (rc == SQLITE_OK) {
    result_check_json(ctx, &run);
  }
done:
  if (rc != SQLITE_OK) {
    result_operational_error(ctx, run.db, rc);
  }
  sqlite3_free(meta);
  sqlite3_free(rows);
  sqlite3_free(nodes);
  sqlite3_free(edges);
  sqlite3_free(rebuild);
  sqlite3_free(run.nodes);
  sqlite3_free(run.map);
  sqlite3_free(run.layer0);
}
