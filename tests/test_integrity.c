#include "sqlite3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (message));     \
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

static void vector2(unsigned char out[8], float x, float y) {
  write_f32_le(out, x);
  write_f32_le(out + 4, y);
}

static sqlite3_int64 scalar_int64(sqlite3 *db, const char *sql) {
  sqlite3_stmt *statement = NULL;
  sqlite3_int64 value = INT64_MIN;
  if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK &&
      sqlite3_step(statement) == SQLITE_ROW) {
    value = sqlite3_column_int64(statement, 0);
  }
  sqlite3_finalize(statement);
  return value;
}

static int check_code(sqlite3 *db, const char *expected) {
  sqlite3_stmt *statement = NULL;
  const unsigned char *actual = NULL;
  int ok = 0;
  int rc = sqlite3_prepare_v2(
      db,
      "SELECT json_extract(hnsw_check('main','vectors'),"
      "'$.first_error.code')",
      -1, &statement, NULL);
  if (rc == SQLITE_OK && sqlite3_step(statement) == SQLITE_ROW) {
    actual = sqlite3_column_text(statement, 0);
    ok = actual != NULL && strcmp((const char *)actual, expected) == 0;
    if (!ok) {
      fprintf(stderr, "expected %s, got %s\n", expected,
              actual != NULL ? (const char *)actual : "NULL");
    }
  }
  sqlite3_finalize(statement);
  return ok;
}

static int expect_code(sqlite3 *db, const char *mutation,
                       const char *expected) {
  int ok = execute(db, "SAVEPOINT damage") == SQLITE_OK;
  if (ok) {
    ok = execute(db, mutation) == SQLITE_OK;
  }
  if (ok) {
    ok = check_code(db, expected);
  }
  if (execute(db, "ROLLBACK TO damage;RELEASE damage") != SQLITE_OK) {
    ok = 0;
  }
  return ok;
}

static void write_u64_le(unsigned char *out, uint64_t value) {
  int i = 0;
  for (i = 0; i < 8; ++i) {
    out[i] = (unsigned char)(value & 0xffU);
    value >>= 8U;
  }
}

static int expect_edge_code(sqlite3 *db, sqlite3_int64 owner, int layer,
                            const sqlite3_int64 *ids, int count,
                            const char *expected) {
  sqlite3_stmt *statement = NULL;
  unsigned char blob[4 + 8 * 32];
  int rc = SQLITE_OK;
  int ok = 0;
  int i = 0;
  if (count < 0 || count > 32) {
    return 0;
  }
  blob[0] = (unsigned char)(count & 0xff);
  blob[1] = (unsigned char)((count >> 8) & 0xff);
  blob[2] = (unsigned char)((count >> 16) & 0xff);
  blob[3] = (unsigned char)((count >> 24) & 0xff);
  for (i = 0; i < count; ++i) {
    write_u64_le(blob + 4 + i * 8, (uint64_t)ids[i]);
  }
  if (execute(db, "SAVEPOINT damage") != SQLITE_OK) {
    return 0;
  }
  rc = sqlite3_prepare_v2(
      db,
      "UPDATE vectors_edges SET neighbors=? WHERE node_id=? AND layer=?",
      -1, &statement, NULL);
  if (rc == SQLITE_OK) {
    sqlite3_bind_blob(statement, 1, blob, 4 + count * 8, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement, 2, owner);
    sqlite3_bind_int(statement, 3, layer);
    rc = sqlite3_step(statement);
  }
  sqlite3_finalize(statement);
  ok = rc == SQLITE_DONE && sqlite3_changes(db) == 1 &&
       check_code(db, expected);
  if (ok && strcmp(expected, "EDGE_NEIGHBOR_MISSING") == 0) {
    ok = scalar_int64(
             db,
             "SELECT json_extract(hnsw_check('main','vectors'),"
             "'$.first_error.node_id')") == owner &&
         scalar_int64(
             db,
             "SELECT json_extract(hnsw_check('main','vectors'),"
             "'$.first_error.layer')") == layer &&
         scalar_int64(
             db,
             "SELECT json_extract(hnsw_check('main','vectors'),"
             "'$.first_error.neighbor_id')") == ids[0];
  }
  if (execute(db, "ROLLBACK TO damage;RELEASE damage") != SQLITE_OK) {
    ok = 0;
  }
  return ok;
}

static int insert_vectors(sqlite3 *db, const char *table, int count) {
  sqlite3_stmt *statement = NULL;
  char *sql = sqlite3_mprintf("INSERT INTO %s(rowid,embedding) VALUES(?,?)",
                              table);
  unsigned char encoded[8];
  int rc = SQLITE_OK;
  int i = 0;
  if (sql == NULL || execute(db, "BEGIN") != SQLITE_OK) {
    sqlite3_free(sql);
    return SQLITE_NOMEM;
  }
  rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  sqlite3_free(sql);
  for (i = 0; i < count && rc == SQLITE_OK; ++i) {
    vector2(encoded, (float)i, (float)(i % 7));
    sqlite3_bind_int64(statement, 1, (sqlite3_int64)i + 1);
    sqlite3_bind_blob(statement, 2, encoded, sizeof(encoded),
                      SQLITE_TRANSIENT);
    rc = sqlite3_step(statement) == SQLITE_DONE ? SQLITE_OK
                                                 : sqlite3_errcode(db);
    sqlite3_reset(statement);
    sqlite3_clear_bindings(statement);
  }
  sqlite3_finalize(statement);
  if (rc == SQLITE_OK) {
    rc = execute(db, "COMMIT");
  } else {
    (void)execute(db, "ROLLBACK");
  }
  return rc;
}

static int query_is_corrupt(sqlite3 *db, const char *table) {
  sqlite3_stmt *query = NULL;
  char *sql = sqlite3_mprintf(
      "SELECT rowid FROM \"%w\" WHERE embedding MATCH hnsw_f32('[0,0]') "
      "AND k=1 AND ef_search=64",
      table);
  int rc = sql == NULL ? SQLITE_NOMEM
                       : sqlite3_prepare_v2(db, sql, -1, &query, NULL);
  sqlite3_free(sql);
  rc = rc == SQLITE_OK ? sqlite3_step(query) : rc;
  sqlite3_finalize(query);
  if (rc != SQLITE_CORRUPT_VTAB &&
      sqlite3_extended_errcode(db) != SQLITE_CORRUPT_VTAB) {
    fprintf(stderr, "expected SQLITE_CORRUPT_VTAB, got rc=%d extended=%d: %s\n",
            rc, sqlite3_extended_errcode(db), sqlite3_errmsg(db));
  }
  return rc == SQLITE_CORRUPT_VTAB ||
         sqlite3_extended_errcode(db) == SQLITE_CORRUPT_VTAB;
}

static void test_integrity(const char *extension) {
  char path[256];
  sqlite3 *db = NULL;
  sqlite3_int64 upper = 0;
  sqlite3_int64 lower = 0;
  sqlite3_int64 ids[5] = {2, 3, 4, 5, 6};
  char *mutation = NULL;
  int rc = SQLITE_OK;
  snprintf(path, sizeof(path), "/tmp/sqlite_hnsw_integrity_%ld.db",
           (long)getpid());
  (void)remove(path);
  CHECK(sqlite3_open(path, &db) == SQLITE_OK, "open integrity database");
  sqlite3_extended_result_codes(db, 1);
  CHECK(sqlite3_db_config(db, SQLITE_DBCONFIG_DEFENSIVE, 0, NULL) == SQLITE_OK,
        "disable defensive mode for corruption corpus");
  CHECK(load(db, extension) == SQLITE_OK, "load extension");
  CHECK(execute(db, "CREATE VIRTUAL TABLE vectors USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=2,ef_construction=4,"
                    "cache_size_mb=0)") == SQLITE_OK,
        "create integrity index");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.ok')") == 1 &&
            scalar_int64(
                db, "SELECT json_extract(hnsw_check('main','vectors'),"
                    "'$.reachable_nodes')") == 0,
        "empty graph passes deep check");
  CHECK(insert_vectors(db, "vectors", 200) == SQLITE_OK,
        "build integrity index");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.ok')") == 1,
        "healthy graph passes deep check");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.checked_nodes')") == 200,
        "deep check reports node count");
  CHECK(scalar_int64(db, "SELECT json_type(hnsw_check('main','vectors'),"
                         "'$.reachable_nodes')='integer'") == 1,
        "deep check reports reachability");

  CHECK(expect_code(db, "DELETE FROM vectors_meta", "META_MISSING"),
        "detect missing metadata");
  CHECK(expect_code(db, "UPDATE vectors_meta SET dims=0", "META_INVALID"),
        "detect invalid metadata");
  CHECK(expect_code(db, "UPDATE vectors_meta SET live_count=live_count+1",
                    "STATE_COUNT_MISMATCH"),
        "detect counter mismatch");
  CHECK(expect_code(db, "UPDATE vectors_meta SET entry_node=999999",
                    "STATE_ENTRY_INVALID"),
        "detect invalid entry point");
  CHECK(expect_code(db, "UPDATE vectors_rows SET node_id=999999 WHERE rowid=1",
                    "ROW_NODE_MISSING"),
        "detect missing mapped node");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET deleted=1 WHERE node_id=1",
                    "ROW_NODE_DELETED"),
        "detect mapping to tombstone");
  CHECK(expect_code(db, "DELETE FROM vectors_rows WHERE rowid=1",
                    "LIVE_NODE_UNMAPPED"),
        "detect unmapped live node");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET node_id=0 WHERE node_id=1",
                    "NODE_ID_INVALID"),
        "detect invalid node id");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET level=-1 WHERE node_id=1",
                    "NODE_LEVEL_INVALID"),
        "detect invalid node level");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET vector=X'0000' "
                        "WHERE node_id=1",
                    "NODE_VECTOR_INVALID"),
        "detect invalid vector blob");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET norm=99 WHERE node_id=1",
                    "NODE_NORM_INVALID"),
        "detect invalid norm");
  CHECK(expect_code(db, "UPDATE vectors_nodes SET level=-2 WHERE node_id=1",
                    "PENDING_NODE"),
        "detect pending node");
  CHECK(expect_code(db, "UPDATE vectors_edges SET node_id=999999 "
                        "WHERE node_id=1 AND layer=0",
                    "EDGE_OWNER_MISSING"),
        "detect missing edge owner");
  CHECK(expect_code(db, "DELETE FROM vectors_edges "
                        "WHERE node_id=1 AND layer=0",
                    "EDGE_ROW_MISSING"),
        "detect missing edge row");
  CHECK(expect_code(db, "UPDATE vectors_edges SET layer=33 "
                        "WHERE node_id=1 AND layer=0",
                    "EDGE_LAYER_INVALID"),
        "detect invalid edge layer");
  CHECK(expect_code(db, "UPDATE vectors_edges SET neighbors=X'0100' "
                        "WHERE node_id=1 AND layer=0",
                    "EDGE_BLOB_INVALID"),
        "detect malformed edge blob");
  CHECK(expect_edge_code(db, 1, 0, ids, 5, "EDGE_DEGREE_EXCEEDED"),
        "detect excessive degree");
  ids[0] = 1;
  CHECK(expect_edge_code(db, 1, 0, ids, 1, "EDGE_SELF_REFERENCE"),
        "detect self edge");
  ids[0] = 2;
  ids[1] = 2;
  CHECK(expect_edge_code(db, 1, 0, ids, 2, "EDGE_DUPLICATE_NEIGHBOR"),
        "detect duplicate neighbor");
  ids[0] = 999999;
  CHECK(expect_edge_code(db, 1, 0, ids, 1, "EDGE_NEIGHBOR_MISSING"),
        "detect missing neighbor");
  upper = scalar_int64(db, "SELECT node_id FROM vectors_nodes "
                             "WHERE level>=1 ORDER BY node_id LIMIT 1");
  lower = scalar_int64(db, "SELECT node_id FROM vectors_nodes "
                             "WHERE level=0 ORDER BY node_id LIMIT 1");
  CHECK(upper > 0 && lower > 0, "find nodes for layer corruption");
  ids[0] = lower;
  CHECK(expect_edge_code(db, upper, 1, ids, 1,
                         "EDGE_NEIGHBOR_LAYER_INVALID"),
        "detect neighbor below edge layer");
  CHECK(expect_code(db, "INSERT INTO vectors_rebuild(rowid,vector) "
                        "VALUES(999,X'0000000000000000')",
                    "REBUILD_NOT_EMPTY"),
        "detect rebuild residue");
  CHECK(execute(db, "SAVEPOINT disconnected;UPDATE vectors_edges "
                    "SET neighbors=X'00000000'") == SQLITE_OK,
        "create structurally valid disconnected graph");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.ok')") == 1 &&
            scalar_int64(
                db, "SELECT json_extract(hnsw_check('main','vectors'),"
                    "'$.reachable_nodes')") == 1,
        "reachability is diagnostic rather than corruption");
  CHECK(execute(db, "ROLLBACK TO disconnected;RELEASE disconnected") ==
            SQLITE_OK,
        "restore disconnected graph");

  CHECK(execute(db, "SAVEPOINT valid_delete;DELETE FROM vectors WHERE rowid=1") ==
            SQLITE_OK,
        "create a valid tombstone");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.ok')") == 1,
        "valid tombstone passes check");
  CHECK(execute(db, "ROLLBACK TO valid_delete;RELEASE valid_delete") ==
            SQLITE_OK,
        "restore tombstone case");
  CHECK(execute(db, "INSERT INTO vectors(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize integrity index");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','vectors'),"
                         "'$.ok')") == 1,
        "optimized graph passes check");

  CHECK(execute(db, "CREATE VIRTUAL TABLE query USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=2,ef_construction=4,"
                    "cache_size_mb=0)") == SQLITE_OK,
        "create query corruption index");
  CHECK(insert_vectors(db, "query", 200) == SQLITE_OK,
        "build query corruption index");
  upper = scalar_int64(db, "SELECT entry_node FROM query_meta WHERE id=1");
  CHECK(upper > 0, "query index has entry node");
  mutation = sqlite3_mprintf(
      "SAVEPOINT query_damage;DELETE FROM query_edges WHERE layer=0");
  CHECK(mutation != NULL && execute(db, mutation) == SQLITE_OK,
        "remove entry edge row");
  sqlite3_free(mutation);
  mutation = NULL;
  CHECK(query_is_corrupt(db, "query"), "query rejects missing edge row");
  CHECK(execute(db, "ROLLBACK TO query_damage;RELEASE query_damage") ==
            SQLITE_OK,
        "restore missing edge row");
  mutation = sqlite3_mprintf(
      "SAVEPOINT query_damage;UPDATE query_nodes SET vector=X'0000' "
      "WHERE node_id=%lld",
      (long long)upper);
  CHECK(mutation != NULL && execute(db, mutation) == SQLITE_OK,
        "corrupt entry vector");
  sqlite3_free(mutation);
  mutation = NULL;
  CHECK(query_is_corrupt(db, "query"), "query rejects invalid vector");
  CHECK(execute(db, "ROLLBACK TO query_damage;RELEASE query_damage") ==
            SQLITE_OK,
        "restore entry vector");
  mutation = sqlite3_mprintf(
      "SAVEPOINT query_damage;DELETE FROM query_rows WHERE node_id=1");
  CHECK(mutation != NULL && execute(db, mutation) == SQLITE_OK,
        "remove result mapping");
  sqlite3_free(mutation);
  mutation = NULL;
  CHECK(query_is_corrupt(db, "query"), "query rejects missing result mapping");
  CHECK(execute(db, "ROLLBACK TO query_damage;RELEASE query_damage") ==
            SQLITE_OK,
        "restore result mapping");

  CHECK(execute(db, "CREATE VIRTUAL TABLE query_snapshot USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=2,ef_construction=4,"
                    "cache_size_mb=16)") == SQLITE_OK,
        "create snapshot corruption index");
  CHECK(insert_vectors(db, "query_snapshot", 200) == SQLITE_OK,
        "build snapshot corruption index");
  CHECK(execute(db, "SAVEPOINT query_damage;"
                    "DELETE FROM query_snapshot_edges WHERE layer=0") ==
            SQLITE_OK,
        "remove snapshot edge rows");
  CHECK(query_is_corrupt(db, "query_snapshot"),
        "snapshot query rejects missing edge rows");
  CHECK(execute(db, "ROLLBACK TO query_damage;RELEASE query_damage") ==
            SQLITE_OK,
        "restore snapshot edge rows");
done:
  sqlite3_free(mutation);
  if (db != NULL && sqlite3_close(db) != SQLITE_OK) {
    fprintf(stderr, "FAIL: sqlite3_close left statements open\n");
    ++failures;
  }
  (void)remove(path);
  (void)rc;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/sqlite_hnsw.so\n", argv[0]);
    return 2;
  }
  test_integrity(argv[1]);
  (void)sqlite3_shutdown();
  if (failures != 0) {
    fprintf(stderr, "%d integrity test(s) failed\n", failures);
    return 1;
  }
  puts("integrity corruption tests passed");
  return 0;
}
