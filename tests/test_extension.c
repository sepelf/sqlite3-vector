#include <math.h>
#include <sqlite3.h>
#include "sqlite_hnsw_version.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (message));      \
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
  sqlite3_int64 result = INT64_MIN;
  if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK &&
      sqlite3_step(statement) == SQLITE_ROW) {
    result = sqlite3_column_int64(statement, 0);
  }
  sqlite3_finalize(statement);
  return result;
}

static int interrupt_progress(void *context) {
  sqlite3_interrupt((sqlite3 *)context);
  return 1;
}

static void test_functions(const char *extension) {
  sqlite3 *db = NULL;
  sqlite3_stmt *statement = NULL;
  int rc = sqlite3_open(":memory:", &db);
  CHECK(rc == SQLITE_OK, "open in-memory database");
  CHECK(load(db, extension) == SQLITE_OK, "load extension");
  rc = sqlite3_prepare_v2(
      db,
      "SELECT hnsw_version(),hnsw_dims(hnsw_f32('[1,2,3]')),"
      "hnsw_distance_l2(hnsw_f32('[0,0]'),hnsw_f32('[3,4]')),"
      "hnsw_distance_cosine(hnsw_f32('[1,0]'),hnsw_f32('[0,1]')),"
      "hnsw_distance_ip(hnsw_f32('[1,2]'),hnsw_f32('[3,4]'))",
      -1, &statement, NULL);
  CHECK(rc == SQLITE_OK, "prepare vector function query");
  CHECK(sqlite3_step(statement) == SQLITE_ROW, "run vector function query");
  CHECK(strcmp((const char *)sqlite3_column_text(statement, 0),
               SQLITE_HNSW_VERSION) == 0,
        "version");
  CHECK(sqlite3_column_int(statement, 1) == 3, "dimensions");
  CHECK(fabs(sqlite3_column_double(statement, 2) - 5.0) < 1e-9, "L2");
  CHECK(fabs(sqlite3_column_double(statement, 3) - 1.0) < 1e-9, "cosine");
  CHECK(fabs(sqlite3_column_double(statement, 4) + 11.0) < 1e-9,
        "inner product");
  sqlite3_finalize(statement);
  statement = NULL;
  CHECK(execute(db, "CREATE VIRTUAL TABLE cosine_vectors USING hnsw("
                    "embedding FLOAT32(2),metric=cosine);"
                    "INSERT INTO cosine_vectors(rowid,embedding) VALUES"
                    "(1,hnsw_f32('[1,0]')),(2,hnsw_f32('[0,1]'))") == SQLITE_OK,
        "create cosine index");
  CHECK(scalar_int64(db,
                     "SELECT rowid FROM cosine_vectors WHERE embedding MATCH "
                     "hnsw_f32('[0.9,0.1]') AND k=1") == 1,
        "cosine index search");
  CHECK(execute(db, "INSERT INTO cosine_vectors(rowid,embedding) "
                    "VALUES(3,hnsw_f32('[0,0]'))") != SQLITE_OK,
        "cosine index rejects zero vectors");
  CHECK(execute(db, "INSERT INTO cosine_vectors(rowid,embedding) "
                    "VALUES(3,hnsw_f32('[1,2,3]'))") != SQLITE_OK,
        "index rejects mismatched dimensions");
  CHECK(execute(db, "CREATE VIRTUAL TABLE ip_vectors USING hnsw("
                    "embedding FLOAT32(2),metric=ip);"
                    "INSERT INTO ip_vectors(rowid,embedding) VALUES"
                    "(1,hnsw_f32('[1,1]')),(2,hnsw_f32('[2,2]'))") == SQLITE_OK,
        "create inner-product index");
  CHECK(scalar_int64(db, "SELECT rowid FROM ip_vectors WHERE embedding MATCH "
                         "hnsw_f32('[1,1]') AND k=1") == 2,
        "inner-product index search");
  CHECK(execute(db, "SELECT hnsw_f32('[1,]')") != SQLITE_OK,
        "reject trailing comma in vector text");
done:
  sqlite3_finalize(statement);
  if (db != NULL && sqlite3_close(db) != SQLITE_OK) {
    fprintf(stderr, "FAIL: sqlite3_close left statements open\n");
    ++failures;
  }
}

static void test_index(const char *extension) {
  char path[256];
  sqlite3 *db = NULL;
  sqlite3 *reader = NULL;
  sqlite3_stmt *insert = NULL;
  sqlite3_stmt *query = NULL;
  unsigned char encoded[8];
  sqlite3_int64 nearest = -1;
  int rc = SQLITE_OK;
  int i = 0;
  snprintf(path, sizeof(path), "/tmp/sqlite_hnsw_test_%ld.db", (long)getpid());
  (void)remove(path);
  rc = sqlite3_open(path, &db);
  CHECK(rc == SQLITE_OK, "open test database");
  CHECK(load(db, extension) == SQLITE_OK, "load extension");
  CHECK(scalar_int64(db, "PRAGMA journal_mode=WAL") != INT64_MIN, "enable WAL");
  CHECK(execute(db, "CREATE VIRTUAL TABLE vectors USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=8,ef_construction=64)") ==
            SQLITE_OK,
        "create index");
  CHECK(execute(db, "CREATE VIRTUAL TABLE rejected USING hnsw("
                    "embedding FLOAT32(2),category TEXT FILTER)") != SQLITE_OK,
        "reject filter columns");
  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin bulk insert");
  rc =
      sqlite3_prepare_v2(db, "INSERT INTO vectors(rowid,embedding) VALUES(?,?)",
                         -1, &insert, NULL);
  CHECK(rc == SQLITE_OK, "prepare insert");
  for (i = 0; i < 200; ++i) {
    vector2(encoded, (float)i, (float)(i % 7));
    sqlite3_bind_int64(insert, 1, (sqlite3_int64)i + 1);
    sqlite3_bind_blob(insert, 2, encoded, 8, SQLITE_TRANSIENT);
    CHECK(sqlite3_step(insert) == SQLITE_DONE, "insert vector");
    sqlite3_reset(insert);
    sqlite3_clear_bindings(insert);
  }
  sqlite3_finalize(insert);
  insert = NULL;
  CHECK(scalar_int64(
            db, "SELECT count(*) FROM vectors_nodes WHERE level<=-2") == 200,
        "transaction batches unlinked nodes");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_info('main','vectors'),"
                         "'\"pending_count\":200') > 0") == 1,
        "index info reports staged nodes");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_check('main','vectors'),"
                         "'\"ok\":false') > 0") == 1,
        "integrity check reports an in-progress graph");
  vector2(encoded, 50.2F, 1.2F);
  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM vectors WHERE embedding MATCH ? AND k=1 "
      "AND ef_search=80",
      -1, &query, NULL);
  CHECK(rc == SQLITE_OK, "prepare read-your-writes query");
  sqlite3_bind_blob(query, 1, encoded, 8, SQLITE_TRANSIENT);
  CHECK(sqlite3_step(query) == SQLITE_ROW, "read-your-writes returns row");
  CHECK(sqlite3_column_int64(query, 0) == 51,
        "read-your-writes includes staged nodes");
  sqlite3_finalize(query);
  query = NULL;
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit bulk insert");
  CHECK(scalar_int64(db,
                     "SELECT count(*) FROM vectors_nodes WHERE level<=-2") == 0,
        "commit links every staged node");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_info('main','vectors'),"
                         "'\"snapshot_cache_eligible\":true') > 0") == 1,
        "index info reports snapshot eligibility");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_info('main','vectors'),"
                         "'\"estimated_snapshot_bytes\":') > 0") == 1,
        "index info reports snapshot memory estimate");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(1,hnsw_f32('[9,9]'))") != SQLITE_OK,
        "duplicate rowid is rejected");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "failed insert is atomic");
  CHECK(execute(db, "INSERT OR IGNORE INTO vectors(rowid,embedding) "
                    "VALUES(1,hnsw_f32('[9,9]'))") == SQLITE_OK,
        "insert or ignore");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "ignored insert preserves row count");

  CHECK(sqlite3_open(path, &reader) == SQLITE_OK, "open WAL reader");
  CHECK(load(reader, extension) == SQLITE_OK, "load extension in WAL reader");
  CHECK(scalar_int64(reader, "SELECT count(*) FROM vectors") == 200,
        "reader sees committed index");
  CHECK(scalar_int64(reader,
                     "SELECT rowid FROM vectors WHERE embedding MATCH "
                     "hnsw_f32('[50.2,1.2]') AND k=1 AND ef_search=80") == 51,
        "reader builds and queries a snapshot");
  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin uncommitted WAL write");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(997,hnsw_f32('[50.2,1.2]'))") == SQLITE_OK,
        "insert WAL row");
  CHECK(scalar_int64(reader, "SELECT count(*) FROM vectors") == 200,
        "reader does not see uncommitted row");
  CHECK(scalar_int64(db,
                     "SELECT rowid FROM vectors WHERE embedding MATCH "
                     "hnsw_f32('[50.2,1.2]') AND k=1 AND ef_search=80") == 997,
        "writer merges linked and pending search results");
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit WAL row");
  CHECK(scalar_int64(reader, "SELECT count(*) FROM vectors") == 201,
        "reader sees committed row");
  CHECK(scalar_int64(reader,
                     "SELECT rowid FROM vectors WHERE embedding MATCH "
                     "hnsw_f32('[50.2,1.2]') AND k=1 AND ef_search=80") == 997,
        "reader refreshes snapshot after writer commit");
  CHECK(execute(db, "DELETE FROM vectors WHERE rowid=997") == SQLITE_OK,
        "remove WAL test row");
  CHECK(scalar_int64(reader, "SELECT count(*) FROM vectors") == 200,
        "reader sees committed deletion");
  CHECK(scalar_int64(reader,
                     "SELECT rowid FROM vectors WHERE embedding MATCH "
                     "hnsw_f32('[50.2,1.2]') AND k=1 AND ef_search=80") == 51,
        "reader refreshes snapshot after deletion");
  CHECK(sqlite3_close(reader) == SQLITE_OK, "close WAL reader");
  reader = NULL;

  vector2(encoded, 50.2F, 1.2F);
  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid,distance,visited_count FROM vectors "
      "WHERE embedding MATCH ? AND k=10 AND ef_search=80 ORDER BY distance",
      -1, &query, NULL);
  CHECK(rc == SQLITE_OK, "prepare ANN query");
  sqlite3_bind_blob(query, 1, encoded, 8, SQLITE_TRANSIENT);
  CHECK(sqlite3_step(query) == SQLITE_ROW, "ANN returns a row");
  nearest = sqlite3_column_int64(query, 0);
  CHECK(nearest == 51, "nearest row is correct");
  CHECK(sqlite3_column_int(query, 2) > 0, "visited count is reported");
  sqlite3_finalize(query);
  query = NULL;

  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM vectors WHERE embedding MATCH ? AND k=1 "
      "AND ef_search=80",
      -1, &query, NULL);
  CHECK(rc == SQLITE_OK, "prepare repeated ANN query");
  for (i = 0; i < 200; i += 10) {
    vector2(encoded, (float)i, (float)(i % 7));
    sqlite3_bind_blob(query, 1, encoded, 8, SQLITE_TRANSIENT);
    CHECK(sqlite3_step(query) == SQLITE_ROW, "repeated ANN returns row");
    CHECK(sqlite3_column_int64(query, 0) == (sqlite3_int64)i + 1,
          "exact inserted vector is its own nearest neighbour");
    sqlite3_reset(query);
    sqlite3_clear_bindings(query);
  }
  sqlite3_finalize(query);
  query = NULL;

  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin rollback test");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(999,hnsw_f32('[50.2,1.2]'))") == SQLITE_OK,
        "insert rollback row");
  CHECK(execute(db, "ROLLBACK") == SQLITE_OK, "rollback");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "rollback restored row count");

  CHECK(execute(db, "BEGIN;SAVEPOINT nested") == SQLITE_OK,
        "begin savepoint test");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(998,hnsw_f32('[50.2,1.2]'))") == SQLITE_OK,
        "insert savepoint row");
  CHECK(execute(db, "ROLLBACK TO nested;RELEASE nested;COMMIT") == SQLITE_OK,
        "rollback savepoint");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "savepoint restored row count");

  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin staged delete test");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(996,hnsw_f32('[50.2,1.2]'));"
                    "DELETE FROM vectors WHERE rowid=996") == SQLITE_OK,
        "delete a staged node");
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit staged delete");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "staged insert and delete cancel cleanly");
  CHECK(scalar_int64(db,
                     "SELECT rowid FROM vectors WHERE embedding MATCH "
                     "hnsw_f32('[50.2,1.2]') AND k=1 AND ef_search=80") == 51,
        "snapshot tolerates an internal node-id gap");

  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin interrupted commit test");
  CHECK(execute(db, "INSERT INTO vectors(rowid,embedding) "
                    "VALUES(995,hnsw_f32('[50.2,1.2]'))") == SQLITE_OK,
        "stage interrupted row");
  sqlite3_progress_handler(db, 1, interrupt_progress, db);
  CHECK(execute(db, "COMMIT") == SQLITE_INTERRUPT,
        "interrupted graph build rejects commit");
  sqlite3_progress_handler(db, 0, NULL, NULL);
  if (!sqlite3_get_autocommit(db)) {
    CHECK(execute(db, "ROLLBACK") == SQLITE_OK,
          "rollback interrupted graph build");
  }
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors") == 200,
        "interrupted commit is atomic");

  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin mutation transaction");
  CHECK(execute(db, "DELETE FROM vectors WHERE rowid=51") == SQLITE_OK,
        "delete nearest");
  CHECK(execute(db, "UPDATE vectors SET embedding=hnsw_f32('[50.2,1.2]') "
                    "WHERE rowid=1") == SQLITE_OK,
        "update vector");
  CHECK(execute(db, "INSERT INTO vectors(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize");
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit mutations");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_check('main','vectors'),"
                         "'\"ok\":true') > 0") == 1,
        "integrity check");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_info('main','vectors'),"
                         "'\"tombstone_count\":0') > 0") == 1,
        "index info after optimize");
  CHECK(sqlite3_close(db) == SQLITE_OK, "close before persistence reopen");
  db = NULL;

  CHECK(sqlite3_open(path, &db) == SQLITE_OK, "reopen test database");
  CHECK(load(db, extension) == SQLITE_OK, "reload extension");
  vector2(encoded, 50.2F, 1.2F);
  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid FROM vectors WHERE embedding MATCH ? AND k=1 "
      "AND ef_search=64",
      -1, &query, NULL);
  CHECK(rc == SQLITE_OK, "prepare persisted query");
  sqlite3_bind_blob(query, 1, encoded, 8, SQLITE_TRANSIENT);
  CHECK(sqlite3_step(query) == SQLITE_ROW, "persisted query returns row");
  CHECK(sqlite3_column_int64(query, 0) == 1, "updated row persisted");
  sqlite3_finalize(query);
  query = NULL;
  CHECK(execute(db, "ALTER TABLE vectors RENAME TO vectors_renamed") ==
            SQLITE_OK,
        "rename index");
  CHECK(scalar_int64(db, "SELECT count(*) FROM vectors_renamed") == 199,
        "renamed index is readable");
  CHECK(execute(db, "DROP TABLE vectors_renamed") == SQLITE_OK, "drop index");
done:
  sqlite3_finalize(insert);
  sqlite3_finalize(query);
  sqlite3_close(reader);
  if (db != NULL && sqlite3_close(db) != SQLITE_OK) {
    fprintf(stderr, "FAIL: sqlite3_close left statements open\n");
    ++failures;
  }
  (void)remove(path);
}

static void test_optimize_compaction(const char *extension) {
  sqlite3 *db = NULL;
  sqlite3_int64 before_bytes = 0;
  sqlite3_int64 before_seq = 0;
  int rc = sqlite3_open(":memory:", &db);
  CHECK(rc == SQLITE_OK, "open optimize compaction database");
  CHECK(load(db, extension) == SQLITE_OK,
        "load extension for optimize compaction");
  CHECK(execute(db, "CREATE VIRTUAL TABLE compact USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=2,ef_construction=4,"
                    "cache_size_mb=16);"
                    "INSERT INTO compact(rowid,embedding) VALUES"
                    "(1,hnsw_f32('[1,1]')),(2,hnsw_f32('[2,2]')),"
                    "(3,hnsw_f32('[3,3]')),(4,hnsw_f32('[4,4]'));"
                    "UPDATE compact SET embedding=hnsw_f32('[1.1,1.1]') "
                    "WHERE rowid=1;"
                    "UPDATE compact SET embedding=hnsw_f32('[1.2,1.2]') "
                    "WHERE rowid=1") == SQLITE_OK,
        "create tombstones for optimize compaction");
  CHECK(scalar_int64(db, "SELECT next_node_id>live_count+1 "
                           "FROM compact_meta WHERE id=1") == 1,
        "updates expand the internal node id span");
  CHECK(scalar_int64(db, "SELECT json_extract("
                           "hnsw_info('main','compact'),"
                           "'$.needs_optimize')") == 1,
        "tombstones request optimize");
  before_bytes = scalar_int64(
      db, "SELECT json_extract(hnsw_info('main','compact'),"
          "'$.estimated_snapshot_bytes')");
  CHECK(execute(db, "INSERT INTO compact(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize compacts tombstones and node ids");
  CHECK(scalar_int64(db, "SELECT tombstone_count=0 AND "
                           "next_node_id=live_count+1 FROM compact_meta "
                           "WHERE id=1") == 1,
        "optimize resets the next internal node id");
  CHECK(scalar_int64(db, "SELECT min(node_id)=1 AND "
                           "max(node_id)=(SELECT live_count FROM compact_meta "
                           "WHERE id=1) FROM compact_nodes") == 1,
        "optimized internal node ids are dense");
  CHECK(scalar_int64(
            db, "SELECT json_extract(hnsw_info('main','compact'),"
                "'$.estimated_snapshot_bytes')") < before_bytes,
        "optimize reduces the query snapshot estimate");
  CHECK(scalar_int64(db, "SELECT json_extract(hnsw_check('main','compact'),"
                           "'$.ok')") == 1,
        "compacted index passes the integrity check");
  CHECK(scalar_int64(db, "SELECT rowid FROM compact WHERE embedding MATCH "
                           "hnsw_f32('[1.2,1.2]') AND k=1") == 1,
        "compacted index preserves external rowids and search results");

  CHECK(execute(db, "BEGIN;"
                    "INSERT INTO compact(rowid,embedding) "
                    "VALUES(99,hnsw_f32('[99,99]'));"
                    "DELETE FROM compact WHERE rowid=99;COMMIT") == SQLITE_OK,
        "create an internal id gap without a tombstone");
  CHECK(scalar_int64(db, "SELECT tombstone_count=0 AND "
                           "next_node_id>live_count+1 FROM compact_meta "
                           "WHERE id=1") == 1,
        "pending insert and delete leaves only an id gap");
  CHECK(scalar_int64(db, "SELECT json_extract("
                           "hnsw_info('main','compact'),"
                           "'$.needs_optimize')") == 1,
        "id gap requests optimize");
  CHECK(execute(db, "INSERT INTO compact(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize compacts an id gap without tombstones");
  CHECK(scalar_int64(db, "SELECT next_node_id=live_count+1 "
                           "FROM compact_meta WHERE id=1") == 1,
        "gap-only optimize restores dense ids");

  before_seq = scalar_int64(
      db, "SELECT change_seq FROM compact_meta WHERE id=1");
  CHECK(execute(db, "INSERT INTO compact(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize accepts an already compact index");
  CHECK(scalar_int64(db, "SELECT change_seq FROM compact_meta WHERE id=1") ==
            before_seq,
        "optimize is a no-op for an already compact index");

  CHECK(execute(db, "DELETE FROM compact") == SQLITE_OK,
        "delete every row before empty compaction");
  CHECK(scalar_int64(db, "SELECT live_count=0 AND next_node_id>1 "
                           "FROM compact_meta WHERE id=1") == 1,
        "empty index retains its historical id span");
  CHECK(execute(db, "INSERT INTO compact(command) VALUES('optimize')") ==
            SQLITE_OK,
        "optimize compacts an empty index");
  CHECK(scalar_int64(db, "SELECT next_node_id=1 AND live_count=0 AND "
                           "tombstone_count=0 FROM compact_meta WHERE id=1") ==
            1,
        "empty optimize resets the internal id sequence");
done:
  if (db != NULL && sqlite3_close(db) != SQLITE_OK) {
    ++failures;
  }
}

static void test_batch_equivalence(const char *extension) {
  sqlite3 *db = NULL;
  sqlite3_stmt *online = NULL;
  sqlite3_stmt *batch = NULL;
  unsigned char encoded[8];
  int rc = sqlite3_open(":memory:", &db);
  int i = 0;
  CHECK(rc == SQLITE_OK, "open equivalence database");
  CHECK(load(db, extension) == SQLITE_OK, "load extension for equivalence");
  CHECK(execute(db, "CREATE VIRTUAL TABLE online USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=8,ef_construction=64,"
                    "build_threads=1);"
                    "CREATE VIRTUAL TABLE batch USING hnsw("
                    "embedding FLOAT32(2),metric=l2,m=8,ef_construction=64,"
                    "cache_size_mb=0,build_threads=4)") ==
            SQLITE_OK,
        "create equivalent indexes");
  CHECK(sqlite3_prepare_v2(db,
                           "INSERT INTO online(rowid,embedding) VALUES(?,?)",
                           -1, &online, NULL) == SQLITE_OK,
        "prepare online insert");
  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin single-thread arena build");
  for (i = 0; i < 4096; ++i) {
    vector2(encoded, (float)i, (float)(i % 3));
    sqlite3_bind_int64(online, 1, (sqlite3_int64)i + 1);
    sqlite3_bind_blob(online, 2, encoded, 8, SQLITE_TRANSIENT);
    CHECK(sqlite3_step(online) == SQLITE_DONE, "online insert");
    sqlite3_reset(online);
    sqlite3_clear_bindings(online);
  }
  sqlite3_finalize(online);
  online = NULL;
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit single-thread arena");
  CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin equivalent batch");
  CHECK(sqlite3_prepare_v2(db, "INSERT INTO batch(rowid,embedding) VALUES(?,?)",
                           -1, &batch, NULL) == SQLITE_OK,
        "prepare batch insert");
  for (i = 0; i < 4096; ++i) {
    vector2(encoded, (float)i, (float)(i % 3));
    sqlite3_bind_int64(batch, 1, (sqlite3_int64)i + 1);
    sqlite3_bind_blob(batch, 2, encoded, 8, SQLITE_TRANSIENT);
    CHECK(sqlite3_step(batch) == SQLITE_DONE, "batch insert");
    sqlite3_reset(batch);
    sqlite3_clear_bindings(batch);
  }
  sqlite3_finalize(batch);
  batch = NULL;
  CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit equivalent batch");
  CHECK(scalar_int64(db,
                     "SELECT (SELECT count(*) FROM ("
                     "SELECT node_id,layer,neighbors FROM online_edges EXCEPT "
                     "SELECT node_id,layer,neighbors FROM batch_edges)) + "
                     "(SELECT count(*) FROM ("
                     "SELECT node_id,layer,neighbors FROM batch_edges EXCEPT "
                     "SELECT node_id,layer,neighbors FROM online_edges))") == 0,
        "online and batched graphs are identical");
  CHECK(scalar_int64(db, "SELECT instr(hnsw_info('main','batch'),"
                         "'\"snapshot_cache_eligible\":false') > 0") == 1,
        "zero-cache index reports snapshot fallback");
  sqlite3_progress_handler(db, 1, interrupt_progress, db);
  CHECK(execute(db,
                "SELECT rowid FROM online WHERE embedding MATCH "
                "hnsw_f32('[2048,2]') AND k=1 AND ef_search=64") ==
            SQLITE_INTERRUPT,
        "snapshot load can be interrupted");
  sqlite3_progress_handler(db, 0, NULL, NULL);
  CHECK(scalar_int64(db,
                     "SELECT rowid FROM online WHERE embedding MATCH "
                     "hnsw_f32('[2048,2]') AND k=1 AND ef_search=64") == 2049,
        "snapshot query returns expected row");
  CHECK(scalar_int64(db,
                     "SELECT rowid FROM batch WHERE embedding MATCH "
                     "hnsw_f32('[2048,2]') AND k=1 AND ef_search=64") == 2049,
        "fallback query returns expected row");
done:
  sqlite3_finalize(online);
  sqlite3_finalize(batch);
  if (db != NULL && sqlite3_close(db) != SQLITE_OK) {
    ++failures;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/sqlite_hnsw.so\n", argv[0]);
    return 2;
  }
  test_functions(argv[1]);
  test_index(argv[1]);
  test_optimize_compaction(argv[1]);
  test_batch_equivalence(argv[1]);
  (void)sqlite3_shutdown();
  if (failures != 0) {
    fprintf(stderr, "%d test(s) failed\n", failures);
    return 1;
  }
  puts("all sqlite-hnsw integration tests passed");
  return 0;
}
