#include <math.h>
#include <sqlite3.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_SLOTS 67
#define MODEL_DIMS 7

typedef struct ModelRow {
  int active;
  float values[MODEL_DIMS];
} ModelRow;

static int failures = 0;
static sqlite3_int64 model_rowids[MODEL_SLOTS];

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

static void cleanup_database(const char *path) {
  char sidecar[512];
  (void)remove(path);
  snprintf(sidecar, sizeof(sidecar), "%s-wal", path);
  (void)remove(sidecar);
  snprintf(sidecar, sizeof(sidecar), "%s-shm", path);
  (void)remove(sidecar);
}

static int open_database(const char *path, const char *extension, int create,
                         sqlite3 **out) {
  sqlite3 *db = NULL;
  int rc = sqlite3_open(path, &db);
  if (rc == SQLITE_OK) {
    rc = load(db, extension);
  }
  if (rc == SQLITE_OK && create) {
    rc = execute(db, "PRAGMA journal_mode=WAL;"
                     "CREATE VIRTUAL TABLE model USING hnsw("
                     "embedding FLOAT32(7),metric=l2,m=8,ef_construction=64,"
                     "cache_size_mb=16,"
                     "build_memory_mb=128,build_threads=2)");
  }
  if (rc != SQLITE_OK) {
    sqlite3_close(db);
    return rc;
  }
  *out = db;
  return SQLITE_OK;
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
  return (float)value / 991.0F;
}

static void make_vector(float values[MODEL_DIMS], uint64_t *state,
                        int serial) {
  for (int i = 0; i < MODEL_DIMS; ++i) {
    values[i] = random_float(state) + (float)(serial + 1) * 0.000031F;
  }
}

static void write_f32_le(unsigned char *out, float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(bits));
  out[0] = (unsigned char)(bits & 0xffU);
  out[1] = (unsigned char)((bits >> 8U) & 0xffU);
  out[2] = (unsigned char)((bits >> 16U) & 0xffU);
  out[3] = (unsigned char)((bits >> 24U) & 0xffU);
}

static void encode_vector(const float values[MODEL_DIMS],
                          unsigned char blob[MODEL_DIMS * 4]) {
  for (int i = 0; i < MODEL_DIMS; ++i) {
    write_f32_le(blob + (size_t)i * 4U, values[i]);
  }
}

static int active_count(const ModelRow model[MODEL_SLOTS]) {
  int count = 0;
  for (int i = 0; i < MODEL_SLOTS; ++i) {
    count += model[i].active != 0;
  }
  return count;
}

static int find_slot(const ModelRow model[MODEL_SLOTS], int active,
                     int start) {
  for (int offset = 0; offset < MODEL_SLOTS; ++offset) {
    int slot = (start + offset) % MODEL_SLOTS;
    if ((model[slot].active != 0) == active) {
      return slot;
    }
  }
  return -1;
}

static int bind_mutation(sqlite3 *db, const char *sql, sqlite3_int64 rowid,
                         const float values[MODEL_DIMS]) {
  sqlite3_stmt *statement = NULL;
  unsigned char blob[MODEL_DIMS * 4];
  int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  if (rc != SQLITE_OK) {
    return rc;
  }
  encode_vector(values, blob);
  sqlite3_bind_int64(statement, 1, rowid);
  sqlite3_bind_blob(statement, 2, blob, sizeof(blob), SQLITE_TRANSIENT);
  rc = sqlite3_step(statement);
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  sqlite3_finalize(statement);
  return rc;
}

static int insert_row(sqlite3 *db, sqlite3_int64 rowid,
                      const float values[MODEL_DIMS], const char *conflict) {
  char sql[160];
  snprintf(sql, sizeof(sql),
           "INSERT OR %s INTO model(rowid,embedding) VALUES(?,?)", conflict);
  return bind_mutation(db, sql, rowid, values);
}

static int update_row(sqlite3 *db, sqlite3_int64 rowid,
                      const float values[MODEL_DIMS]) {
  return bind_mutation(
      db, "UPDATE model SET embedding=?2 WHERE rowid=?1", rowid, values);
}

static int delete_row(sqlite3 *db, sqlite3_int64 rowid) {
  sqlite3_stmt *statement = NULL;
  int rc = sqlite3_prepare_v2(db, "DELETE FROM model WHERE rowid=?", -1,
                              &statement, NULL);
  if (rc == SQLITE_OK) {
    sqlite3_bind_int64(statement, 1, rowid);
    rc = sqlite3_step(statement);
  }
  if (rc == SQLITE_DONE) {
    rc = SQLITE_OK;
  }
  sqlite3_finalize(statement);
  return rc;
}

static double l2_distance(const float *left, const float *right) {
  double sum = 0.0;
  for (int i = 0; i < MODEL_DIMS; ++i) {
    double delta = (double)left[i] - (double)right[i];
    sum += delta * delta;
  }
  return sqrt(sum);
}

static int verify_rows(sqlite3 *db, const ModelRow model[MODEL_SLOTS]) {
  sqlite3_stmt *statement = NULL;
  unsigned char expected_blob[MODEL_DIMS * 4];
  unsigned char seen[MODEL_SLOTS];
  int count = 0;
  int rc = sqlite3_prepare_v2(
      db, "SELECT rowid,embedding FROM model ORDER BY rowid", -1, &statement,
      NULL);
  memset(seen, 0, sizeof(seen));
  if (rc != SQLITE_OK) {
    return 0;
  }
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 rowid = sqlite3_column_int64(statement, 0);
    int slot = -1;
    for (int i = 0; i < MODEL_SLOTS; ++i) {
      if (model_rowids[i] == rowid) {
        slot = i;
        break;
      }
    }
    if (slot < 0 || !model[slot].active || seen[slot] ||
        sqlite3_column_bytes(statement, 1) != MODEL_DIMS * 4) {
      sqlite3_finalize(statement);
      return 0;
    }
    encode_vector(model[slot].values, expected_blob);
    if (memcmp(sqlite3_column_blob(statement, 1), expected_blob,
               sizeof(expected_blob)) != 0) {
      sqlite3_finalize(statement);
      return 0;
    }
    seen[slot] = 1;
    ++count;
  }
  sqlite3_finalize(statement);
  if (rc != SQLITE_DONE || count != active_count(model)) {
    return 0;
  }
  for (int i = 0; i < MODEL_SLOTS; ++i) {
    if ((seen[i] != 0) != (model[i].active != 0)) {
      return 0;
    }
  }
  return 1;
}

static int verify_ann(sqlite3 *db, const ModelRow model[MODEL_SLOTS],
                      uint64_t seed) {
  float query[MODEL_DIMS];
  unsigned char blob[MODEL_DIMS * 4];
  int seen[MODEL_SLOTS] = {0};
  sqlite3_stmt *statement = NULL;
  int available = active_count(model);
  int k = available < 5 ? available : 5;
  int result_count = 0;
  int visited = -1;
  double previous_distance = -INFINITY;
  int rc = SQLITE_OK;
  for (int i = 0; i < MODEL_DIMS; ++i) {
    query[i] = random_float(&seed);
  }
  if (k == 0) {
    sqlite3_int64 count = -1;
    if (sqlite3_prepare_v2(db, "SELECT count(*) FROM model", -1, &statement,
                           NULL) != SQLITE_OK ||
        sqlite3_step(statement) != SQLITE_ROW) {
      sqlite3_finalize(statement);
      return 0;
    }
    count = sqlite3_column_int64(statement, 0);
    sqlite3_finalize(statement);
    return count == 0;
  }
  encode_vector(query, blob);
  rc = sqlite3_prepare_v2(
      db,
      "SELECT rowid,distance,visited_count FROM model "
      "WHERE embedding MATCH ? AND k=? AND ef_search=64 ORDER BY distance",
      -1, &statement, NULL);
  if (rc != SQLITE_OK) {
    return 0;
  }
  sqlite3_bind_blob(statement, 1, blob, sizeof(blob), SQLITE_TRANSIENT);
  sqlite3_bind_int(statement, 2, k);
  while ((rc = sqlite3_step(statement)) == SQLITE_ROW) {
    sqlite3_int64 rowid = sqlite3_column_int64(statement, 0);
    double distance = sqlite3_column_double(statement, 1);
    int row_visited = sqlite3_column_int(statement, 2);
    int slot = 0;
    double expected_distance = 0.0;
    while (slot < MODEL_SLOTS && model_rowids[slot] != rowid) {
      ++slot;
    }
    if (slot < MODEL_SLOTS) {
      expected_distance = l2_distance(query, model[slot].values);
    }
    if (result_count >= k || slot == MODEL_SLOTS || !model[slot].active ||
        seen[slot] || distance < previous_distance || row_visited <= 0 ||
        (visited >= 0 && row_visited != visited) ||
        fabs(distance - expected_distance) >
            1e-5 * fmax(1.0, expected_distance)) {
      sqlite3_finalize(statement);
      return 0;
    }
    seen[slot] = 1;
    previous_distance = distance;
    visited = row_visited;
    ++result_count;
  }
  sqlite3_finalize(statement);
  return rc == SQLITE_DONE && result_count <= k;
}

static int scalar_int(sqlite3 *db, const char *sql, sqlite3_int64 *value) {
  sqlite3_stmt *statement = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  if (rc == SQLITE_OK) {
    rc = sqlite3_step(statement);
  }
  if (rc == SQLITE_ROW) {
    *value = sqlite3_column_int64(statement, 0);
    rc = SQLITE_OK;
  }
  sqlite3_finalize(statement);
  return rc;
}

static int scalar_text_is(sqlite3 *db, const char *sql, const char *expected) {
  sqlite3_stmt *statement = NULL;
  int equal = 0;
  if (sqlite3_prepare_v2(db, sql, -1, &statement, NULL) == SQLITE_OK &&
      sqlite3_step(statement) == SQLITE_ROW) {
    const unsigned char *value = sqlite3_column_text(statement, 0);
    equal = value != NULL && strcmp((const char *)value, expected) == 0;
  }
  sqlite3_finalize(statement);
  return equal;
}

static int verify_committed(sqlite3 *db,
                            const ModelRow model[MODEL_SLOTS], uint64_t seed) {
  sqlite3_int64 ok = 0;
  if (!verify_rows(db, model) || !verify_ann(db, model, seed) ||
      !verify_ann(db, model, seed ^ UINT64_C(0x9e3779b97f4a7c15)) ||
      !verify_ann(db, model, seed ^ UINT64_C(0xd1b54a32d192ed03))) {
    return 0;
  }
  if (scalar_int(db,
                 "SELECT json_extract(hnsw_check('main','model'),'$.ok')",
                 &ok) != SQLITE_OK ||
      ok != 1) {
    return 0;
  }
  return scalar_text_is(db, "PRAGMA integrity_check", "ok");
}

static int mutate(sqlite3 *db, ModelRow model[MODEL_SLOTS], uint64_t *state,
                  int *serial) {
  int operation = (int)(next_random(state) % 100U);
  int start = (int)(next_random(state) % MODEL_SLOTS);
  int slot = -1;
  float values[MODEL_DIMS];
  if (operation < 24) {
    slot = find_slot(model, 0, start);
    if (slot < 0) {
      operation = 50;
    } else {
      make_vector(values, state, (*serial)++);
      if (insert_row(db, model_rowids[slot], values, "ABORT") != SQLITE_OK) {
        return 0;
      }
      model[slot].active = 1;
      memcpy(model[slot].values, values, sizeof(values));
      return 1;
    }
  }
  if (operation < 50) {
    slot = find_slot(model, 1, start);
    if (slot < 0) {
      return 1;
    }
    make_vector(values, state, (*serial)++);
    if (update_row(db, model_rowids[slot], values) != SQLITE_OK) {
      return 0;
    }
    memcpy(model[slot].values, values, sizeof(values));
    return 1;
  }
  if (operation < 68) {
    slot = find_slot(model, 1, start);
    if (slot < 0) {
      return 1;
    }
    if (delete_row(db, model_rowids[slot]) != SQLITE_OK) {
      return 0;
    }
    model[slot].active = 0;
    return 1;
  }
  if (operation < 78) {
    slot = find_slot(model, 1, start);
    if (slot < 0) {
      return 1;
    }
    make_vector(values, state, (*serial)++);
    if (insert_row(db, model_rowids[slot], values, "IGNORE") != SQLITE_OK) {
      return 0;
    }
    return 1;
  }
  if (operation < 90) {
    slot = find_slot(model, 1, start);
    if (slot < 0) {
      return 1;
    }
    make_vector(values, state, (*serial)++);
    if (insert_row(db, model_rowids[slot], values, "REPLACE") != SQLITE_OK) {
      return 0;
    }
    memcpy(model[slot].values, values, sizeof(values));
    return 1;
  }
  return execute(db, "INSERT INTO model(command) VALUES('optimize')") ==
         SQLITE_OK;
}

static int test_auto_rowid(sqlite3 *db, uint64_t *state) {
  sqlite3_stmt *statement = NULL;
  sqlite3_stmt *read_statement = NULL;
  unsigned char blob[MODEL_DIMS * 4];
  float values[MODEL_DIMS];
  sqlite3_int64 rowid = 0;
  int step_rc = SQLITE_OK;
  int delete_rc = SQLITE_OK;
  make_vector(values, state, 0);
  encode_vector(values, blob);
  if (sqlite3_prepare_v2(db, "INSERT INTO model(embedding) VALUES(?)", -1,
                         &statement, NULL) != SQLITE_OK) {
    fprintf(stderr, "auto-rowid prepare failed: %s\n", sqlite3_errmsg(db));
    return 0;
  }
  sqlite3_bind_blob(statement, 1, blob, sizeof(blob), SQLITE_TRANSIENT);
  step_rc = sqlite3_step(statement);
  while (step_rc == SQLITE_ROW) {
    step_rc = sqlite3_step(statement);
  }
  if (step_rc != SQLITE_DONE) {
    fprintf(stderr, "auto-rowid insert failed: rc=%d message=%s\n", step_rc,
            sqlite3_errmsg(db));
    sqlite3_finalize(statement);
    return 0;
  }
  sqlite3_finalize(statement);
  step_rc = sqlite3_prepare_v2(db, "SELECT rowid FROM model", -1,
                               &read_statement, NULL);
  if (step_rc == SQLITE_OK) {
    step_rc = sqlite3_step(read_statement);
  }
  if (step_rc != SQLITE_ROW) {
    fprintf(stderr, "auto-rowid read failed: rc=%d message=%s\n", step_rc,
            sqlite3_errmsg(db));
    sqlite3_finalize(read_statement);
    return 0;
  }
  rowid = sqlite3_column_int64(read_statement, 0);
  sqlite3_finalize(read_statement);
  delete_rc = delete_row(db, rowid);
  if (delete_rc != SQLITE_OK) {
    fprintf(stderr, "auto-rowid delete failed: rowid=%lld rc=%d message=%s\n",
            (long long)rowid, delete_rc, sqlite3_errmsg(db));
  }
  return delete_rc == SQLITE_OK;
}

static void test_state_machine(const char *extension) {
  char path[256];
  sqlite3 *db = NULL;
  ModelRow committed[MODEL_SLOTS];
  ModelRow working[MODEL_SLOTS];
  ModelRow saved[MODEL_SLOTS];
  uint64_t state = UINT64_C(0x6a09e667f3bcc909);
  int serial = 1;
  snprintf(path, sizeof(path), "/tmp/sqlite_hnsw_state_%ld.db", (long)getpid());
  cleanup_database(path);
  memset(committed, 0, sizeof(committed));
  model_rowids[0] = -7;
  model_rowids[1] = 0;
  model_rowids[2] = INT64_C(1000000000000);
  for (int i = 3; i < MODEL_SLOTS; ++i) {
    model_rowids[i] = (sqlite3_int64)i - 2;
  }
  CHECK(open_database(path, extension, 1, &db) == SQLITE_OK,
        "create state-model database");
  CHECK(test_auto_rowid(db, &state), "automatic rowid insert/delete works");
  CHECK(execute(db, "BEGIN") == SQLITE_OK,
        "begin special-rowid initialization");
  for (int slot = 0; slot < 3; ++slot) {
    make_vector(committed[slot].values, &state, serial++);
    committed[slot].active = 1;
    CHECK(insert_row(db, model_rowids[slot], committed[slot].values, "ABORT") ==
              SQLITE_OK,
          "insert special rowid");
  }
  CHECK(execute(db, "COMMIT") == SQLITE_OK,
        "commit special-rowid initialization");
  CHECK(verify_committed(db, committed, state),
        "verify initialized state model");
  for (int round = 0; round < 60; ++round) {
    int operations = 1 + (int)(next_random(&state) % 8U);
    int save_at =
        operations > 1
            ? (int)(next_random(&state) % (uint64_t)(unsigned int)operations)
            : -1;
    int rollback_savepoint = (int)(next_random(&state) & 1U);
    int save_active = 0;
    int commit = (next_random(&state) % 4U) != 0;
    memcpy(working, committed, sizeof(working));
    CHECK(execute(db, "BEGIN") == SQLITE_OK, "begin model transaction");
    for (int operation = 0; operation < operations; ++operation) {
      if (operation == save_at) {
        CHECK(execute(db, "SAVEPOINT model_save") == SQLITE_OK,
              "create model savepoint");
        memcpy(saved, working, sizeof(saved));
        save_active = 1;
      }
      CHECK(mutate(db, working, &state, &serial), "apply modeled mutation");
      CHECK(verify_rows(db, working), "read-your-writes rows match model");
      CHECK(verify_ann(db, working,
                       state ^ (uint64_t)round ^ (uint64_t)operation),
            "read-your-writes ANN query matches model");
      if (save_active && operation == save_at + 1) {
        if (rollback_savepoint) {
          CHECK(execute(db, "ROLLBACK TO model_save;RELEASE model_save") ==
                    SQLITE_OK,
                "rollback modeled savepoint");
          memcpy(working, saved, sizeof(working));
        } else {
          CHECK(execute(db, "RELEASE model_save") == SQLITE_OK,
                "release modeled savepoint");
        }
        save_active = 0;
        CHECK(verify_rows(db, working),
              "savepoint result matches working model");
      }
    }
    if (save_active) {
      CHECK(execute(db, "RELEASE model_save") == SQLITE_OK,
            "release trailing model savepoint");
    }
    if (commit) {
      CHECK(execute(db, "COMMIT") == SQLITE_OK, "commit modeled transaction");
      memcpy(committed, working, sizeof(committed));
    } else {
      CHECK(execute(db, "ROLLBACK") == SQLITE_OK,
            "rollback modeled transaction");
    }
    CHECK(sqlite3_close(db) == SQLITE_OK,
          "close state-model database after transaction");
    db = NULL;
    CHECK(open_database(path, extension, 0, &db) == SQLITE_OK,
          "reopen state-model database");
    CHECK(verify_committed(db, committed,
                           state ^ UINT64_C(0xa5a5a5a5a5a5a5a5)),
          "reopened state matches committed model");
  }
done:
  if (db != NULL) {
    if (!sqlite3_get_autocommit(db)) {
      (void)execute(db, "ROLLBACK");
    }
    sqlite3_close(db);
  }
  cleanup_database(path);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/sqlite_hnsw.so\n", argv[0]);
    return 2;
  }
  test_state_machine(argv[1]);
  (void)sqlite3_shutdown();
  if (failures != 0) {
    fprintf(stderr, "%d state-model test(s) failed\n", failures);
    return 1;
  }
  puts("transaction state-model tests passed");
  return 0;
}
