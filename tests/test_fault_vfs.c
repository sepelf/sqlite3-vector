#include "sqlite3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { FAULT_NONE = 0, FAULT_WRITE = 1, FAULT_SYNC = 2 };
enum { OP_SMALL = 0, OP_ARENA = 1, OP_MUTATE = 2, OP_OPTIMIZE = 3 };

typedef struct FaultControl {
  int kind;
  int target;
  int writes;
  int syncs;
  int faulted;
  int tracing;
} FaultControl;

typedef struct FaultVfs {
  sqlite3_vfs base;
  sqlite3_vfs *real;
  FaultControl control;
  char name[64];
} FaultVfs;

typedef struct FaultFile {
  sqlite3_file base;
  sqlite3_file *real;
  FaultControl *control;
  int injectable;
} FaultFile;

static int failures = 0;

#define CHECK(condition, message)                                              \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (message));     \
      ++failures;                                                              \
      goto done;                                                               \
    }                                                                          \
  } while (0)

static int fault_close(sqlite3_file *file) {
  FaultFile *wrapped = (FaultFile *)file;
  int rc = wrapped->real->pMethods->xClose(wrapped->real);
  wrapped->base.pMethods = NULL;
  return rc;
}

static int fault_read(sqlite3_file *file, void *buffer, int amount,
                      sqlite3_int64 offset) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xRead(wrapped->real, buffer, amount, offset);
}

static int should_fail(FaultFile *file, int kind) {
  FaultControl *control = file->control;
  int current = 0;
  if (!file->injectable || !control->tracing) {
    return 0;
  }
  if (kind == FAULT_WRITE) {
    current = ++control->writes;
  } else {
    current = ++control->syncs;
  }
  if (control->kind == kind && control->target == current) {
    control->faulted = 1;
    return 1;
  }
  return 0;
}

static int fault_write(sqlite3_file *file, const void *buffer, int amount,
                       sqlite3_int64 offset) {
  FaultFile *wrapped = (FaultFile *)file;
  if (should_fail(wrapped, FAULT_WRITE)) {
    return SQLITE_IOERR_WRITE;
  }
  return wrapped->real->pMethods->xWrite(wrapped->real, buffer, amount,
                                          offset);
}

static int fault_truncate(sqlite3_file *file, sqlite3_int64 size) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xTruncate(wrapped->real, size);
}

static int fault_sync(sqlite3_file *file, int flags) {
  FaultFile *wrapped = (FaultFile *)file;
  if (should_fail(wrapped, FAULT_SYNC)) {
    return SQLITE_IOERR_FSYNC;
  }
  return wrapped->real->pMethods->xSync(wrapped->real, flags);
}

static int fault_file_size(sqlite3_file *file, sqlite3_int64 *size) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xFileSize(wrapped->real, size);
}

static int fault_lock(sqlite3_file *file, int lock) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xLock(wrapped->real, lock);
}

static int fault_unlock(sqlite3_file *file, int lock) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xUnlock(wrapped->real, lock);
}

static int fault_check_reserved(sqlite3_file *file, int *result) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xCheckReservedLock(wrapped->real, result);
}

static int fault_file_control(sqlite3_file *file, int operation,
                              void *argument) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xFileControl(wrapped->real, operation,
                                               argument);
}

static int fault_sector_size(sqlite3_file *file) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xSectorSize(wrapped->real);
}

static int fault_device_characteristics(sqlite3_file *file) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xDeviceCharacteristics(wrapped->real);
}

static int fault_shm_map(sqlite3_file *file, int page, int size, int extend,
                         void volatile **out) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xShmMap(wrapped->real, page, size, extend,
                                          out);
}

static int fault_shm_lock(sqlite3_file *file, int offset, int count,
                          int flags) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xShmLock(wrapped->real, offset, count, flags);
}

static void fault_shm_barrier(sqlite3_file *file) {
  FaultFile *wrapped = (FaultFile *)file;
  wrapped->real->pMethods->xShmBarrier(wrapped->real);
}

static int fault_shm_unmap(sqlite3_file *file, int delete_flag) {
  FaultFile *wrapped = (FaultFile *)file;
  return wrapped->real->pMethods->xShmUnmap(wrapped->real, delete_flag);
}

static int fault_fetch(sqlite3_file *file, sqlite3_int64 offset, int amount,
                       void **out) {
  FaultFile *wrapped = (FaultFile *)file;
  if (wrapped->real->pMethods->iVersion < 3 ||
      wrapped->real->pMethods->xFetch == NULL) {
    *out = NULL;
    return SQLITE_OK;
  }
  return wrapped->real->pMethods->xFetch(wrapped->real, offset, amount, out);
}

static int fault_unfetch(sqlite3_file *file, sqlite3_int64 offset, void *page) {
  FaultFile *wrapped = (FaultFile *)file;
  if (wrapped->real->pMethods->iVersion < 3 ||
      wrapped->real->pMethods->xUnfetch == NULL) {
    return SQLITE_OK;
  }
  return wrapped->real->pMethods->xUnfetch(wrapped->real, offset, page);
}

static const sqlite3_io_methods fault_methods = {
    3,
    fault_close,
    fault_read,
    fault_write,
    fault_truncate,
    fault_sync,
    fault_file_size,
    fault_lock,
    fault_unlock,
    fault_check_reserved,
    fault_file_control,
    fault_sector_size,
    fault_device_characteristics,
    fault_shm_map,
    fault_shm_lock,
    fault_shm_barrier,
    fault_shm_unmap,
    fault_fetch,
    fault_unfetch,
};

static sqlite3_vfs *real_vfs(sqlite3_vfs *vfs) {
  return ((FaultVfs *)vfs)->real;
}

static int fault_vfs_open(sqlite3_vfs *vfs, const char *name,
                          sqlite3_file *file, int flags, int *out_flags) {
  FaultVfs *fault_vfs = (FaultVfs *)vfs;
  FaultFile *wrapped = (FaultFile *)file;
  int rc = SQLITE_OK;
  memset(wrapped, 0, sizeof(*wrapped));
  wrapped->real = (sqlite3_file *)((unsigned char *)file + sizeof(*wrapped));
  wrapped->control = &fault_vfs->control;
  wrapped->injectable =
      (flags & (SQLITE_OPEN_MAIN_DB | SQLITE_OPEN_WAL |
                SQLITE_OPEN_MAIN_JOURNAL)) != 0;
  rc = fault_vfs->real->xOpen(fault_vfs->real, name, wrapped->real, flags,
                              out_flags);
  if (rc == SQLITE_OK) {
    wrapped->base.pMethods = &fault_methods;
  }
  return rc;
}

static int fault_vfs_delete(sqlite3_vfs *vfs, const char *name, int sync_dir) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xDelete(real, name, sync_dir);
}

static int fault_vfs_access(sqlite3_vfs *vfs, const char *name, int flags,
                            int *result) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xAccess(real, name, flags, result);
}

static int fault_vfs_full_pathname(sqlite3_vfs *vfs, const char *name,
                                   int bytes, char *out) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xFullPathname(real, name, bytes, out);
}

static void *fault_vfs_dl_open(sqlite3_vfs *vfs, const char *name) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xDlOpen(real, name);
}

static void fault_vfs_dl_error(sqlite3_vfs *vfs, int bytes, char *out) {
  sqlite3_vfs *real = real_vfs(vfs);
  real->xDlError(real, bytes, out);
}

static void (*fault_vfs_dl_sym(sqlite3_vfs *vfs, void *handle,
                               const char *symbol))(void) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xDlSym(real, handle, symbol);
}

static void fault_vfs_dl_close(sqlite3_vfs *vfs, void *handle) {
  sqlite3_vfs *real = real_vfs(vfs);
  real->xDlClose(real, handle);
}

static int fault_vfs_randomness(sqlite3_vfs *vfs, int bytes, char *out) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xRandomness(real, bytes, out);
}

static int fault_vfs_sleep(sqlite3_vfs *vfs, int microseconds) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xSleep(real, microseconds);
}

static int fault_vfs_current_time(sqlite3_vfs *vfs, double *time) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xCurrentTime(real, time);
}

static int fault_vfs_last_error(sqlite3_vfs *vfs, int first, char *second) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xGetLastError(real, first, second);
}

static int fault_vfs_current_time_int64(sqlite3_vfs *vfs,
                                        sqlite3_int64 *time) {
  sqlite3_vfs *real = real_vfs(vfs);
  if (real->iVersion < 2 || real->xCurrentTimeInt64 == NULL) {
    double julian = 0.0;
    int rc = real->xCurrentTime(real, &julian);
    *time = (sqlite3_int64)(julian * 86400000.0);
    return rc;
  }
  return real->xCurrentTimeInt64(real, time);
}

static int fault_vfs_set_system_call(sqlite3_vfs *vfs, const char *name,
                                     sqlite3_syscall_ptr call) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xSetSystemCall != NULL
             ? real->xSetSystemCall(real, name, call)
             : SQLITE_NOTFOUND;
}

static sqlite3_syscall_ptr fault_vfs_get_system_call(sqlite3_vfs *vfs,
                                                      const char *name) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xGetSystemCall != NULL ? real->xGetSystemCall(real, name)
                                      : NULL;
}

static const char *fault_vfs_next_system_call(sqlite3_vfs *vfs,
                                               const char *name) {
  sqlite3_vfs *real = real_vfs(vfs);
  return real->xNextSystemCall != NULL ? real->xNextSystemCall(real, name)
                                       : NULL;
}

static int register_fault_vfs(FaultVfs *vfs) {
  sqlite3_vfs *real = sqlite3_vfs_find(NULL);
  memset(vfs, 0, sizeof(*vfs));
  if (real == NULL) {
    return SQLITE_NOTFOUND;
  }
  vfs->real = real;
  snprintf(vfs->name, sizeof(vfs->name), "hnsw_fault_%ld", (long)getpid());
  vfs->base.iVersion = real->iVersion > 3 ? 3 : real->iVersion;
  vfs->base.szOsFile = (int)(sizeof(FaultFile) + (size_t)real->szOsFile);
  vfs->base.mxPathname = real->mxPathname;
  vfs->base.zName = vfs->name;
  vfs->base.xOpen = fault_vfs_open;
  vfs->base.xDelete = fault_vfs_delete;
  vfs->base.xAccess = fault_vfs_access;
  vfs->base.xFullPathname = fault_vfs_full_pathname;
  vfs->base.xDlOpen = fault_vfs_dl_open;
  vfs->base.xDlError = fault_vfs_dl_error;
  vfs->base.xDlSym = fault_vfs_dl_sym;
  vfs->base.xDlClose = fault_vfs_dl_close;
  vfs->base.xRandomness = fault_vfs_randomness;
  vfs->base.xSleep = fault_vfs_sleep;
  vfs->base.xCurrentTime = fault_vfs_current_time;
  vfs->base.xGetLastError = fault_vfs_last_error;
  vfs->base.xCurrentTimeInt64 = fault_vfs_current_time_int64;
  vfs->base.xSetSystemCall = fault_vfs_set_system_call;
  vfs->base.xGetSystemCall = fault_vfs_get_system_call;
  vfs->base.xNextSystemCall = fault_vfs_next_system_call;
  return sqlite3_vfs_register(&vfs->base, 0);
}

static int execute(sqlite3 *db, const char *sql) {
  char *error = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &error);
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

static int insert_range(sqlite3 *db, int first, int count) {
  sqlite3_stmt *content = NULL;
  sqlite3_stmt *vectors = NULL;
  unsigned char encoded[8];
  int rc = execute(db, "BEGIN");
  int i = 0;
  if (rc == SQLITE_OK) {
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO content(id,x,y) VALUES(?,?,?)", -1, &content, NULL);
  }
  if (rc == SQLITE_OK) {
    rc = sqlite3_prepare_v2(
        db, "INSERT INTO vectors(rowid,embedding) VALUES(?,?)", -1, &vectors,
        NULL);
  }
  for (i = 0; i < count && rc == SQLITE_OK; ++i) {
    sqlite3_int64 id = (sqlite3_int64)first + i;
    float x = (float)(id - 1);
    float y = (float)((id - 1) % 7);
    sqlite3_bind_int64(content, 1, id);
    sqlite3_bind_double(content, 2, x);
    sqlite3_bind_double(content, 3, y);
    rc = sqlite3_step(content) == SQLITE_DONE ? SQLITE_OK
                                               : sqlite3_errcode(db);
    sqlite3_reset(content);
    sqlite3_clear_bindings(content);
    if (rc != SQLITE_OK) {
      break;
    }
    write_f32_le(encoded, x);
    write_f32_le(encoded + 4, y);
    sqlite3_bind_int64(vectors, 1, id);
    sqlite3_bind_blob(vectors, 2, encoded, sizeof(encoded), SQLITE_TRANSIENT);
    rc = sqlite3_step(vectors) == SQLITE_DONE ? SQLITE_OK
                                               : sqlite3_errcode(db);
    sqlite3_reset(vectors);
    sqlite3_clear_bindings(vectors);
  }
  sqlite3_finalize(content);
  sqlite3_finalize(vectors);
  if (rc == SQLITE_OK) {
    rc = execute(db, "COMMIT");
  }
  return rc;
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

static int scalar_text_equals(sqlite3 *db, const char *sql,
                              const char *expected) {
  sqlite3_stmt *statement = NULL;
  int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
  int equal = 0;
  if (rc == SQLITE_OK && sqlite3_step(statement) == SQLITE_ROW) {
    const unsigned char *value = sqlite3_column_text(statement, 0);
    equal = value != NULL && strcmp((const char *)value, expected) == 0;
  }
  sqlite3_finalize(statement);
  return equal;
}

static int open_database(FaultVfs *vfs, const char *path,
                         const char *extension, sqlite3 **out) {
  int rc = sqlite3_open_v2(path, out, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                           vfs->name);
  if (rc == SQLITE_OK) {
    sqlite3_extended_result_codes(*out, 1);
    rc = load(*out, extension);
  }
  return rc;
}

static void cleanup_database(const char *path) {
  char suffix[512];
  (void)remove(path);
  snprintf(suffix, sizeof(suffix), "%s-wal", path);
  (void)remove(suffix);
  snprintf(suffix, sizeof(suffix), "%s-shm", path);
  (void)remove(suffix);
}

static int setup_operation(FaultVfs *vfs, const char *path,
                           const char *extension, int operation, sqlite3 **db,
                           int *expected_rows) {
  int rc = open_database(vfs, path, extension, db);
  if (rc == SQLITE_OK) {
    rc = execute(*db, "PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;"
                      "CREATE TABLE content(id INTEGER PRIMARY KEY,x REAL,y "
                      "REAL);"
                      "CREATE VIRTUAL TABLE vectors USING hnsw("
                      "embedding FLOAT32(2),metric=l2,m=8,ef_construction=64,"
                      "cache_size_mb=0,build_threads=1,"
                      "build_memory_mb=128)");
  }
  *expected_rows = 0;
  if (rc == SQLITE_OK && (operation == OP_MUTATE || operation == OP_OPTIMIZE)) {
    rc = insert_range(*db, 1, 128);
    *expected_rows = 128;
  }
  if (rc == SQLITE_OK && operation == OP_OPTIMIZE) {
    rc = execute(*db, "BEGIN;DELETE FROM content WHERE id<=32;"
                      "DELETE FROM vectors WHERE rowid<=32;COMMIT");
    *expected_rows = 96;
  }
  return rc;
}

static int perform_operation(sqlite3 *db, int operation) {
  if (operation == OP_SMALL) {
    return insert_range(db, 1, 32);
  }
  if (operation == OP_ARENA) {
    return insert_range(db, 1, 4096);
  }
  if (operation == OP_MUTATE) {
    return execute(db, "BEGIN;UPDATE content SET x=999,y=999 WHERE id=1;"
                       "UPDATE vectors SET embedding=hnsw_f32('[999,999]') "
                       "WHERE rowid=1;DELETE FROM content WHERE id=2;"
                       "DELETE FROM vectors WHERE rowid=2;COMMIT");
  }
  return execute(db, "BEGIN;INSERT INTO vectors(command) VALUES('optimize');"
                     "COMMIT");
}

static int verify_recovered(FaultVfs *vfs, const char *path,
                            const char *extension, int operation,
                            int expected_rows) {
  sqlite3 *db = NULL;
  sqlite3_int64 content_rows = -1;
  sqlite3_int64 vector_rows = -1;
  sqlite3_int64 row2 = -1;
  sqlite3_int64 row1_x = -1;
  int rc = open_database(vfs, path, extension, &db);
  if (rc == SQLITE_OK &&
      !scalar_text_equals(db, "PRAGMA integrity_check", "ok")) {
    rc = SQLITE_CORRUPT;
  }
  if (rc == SQLITE_OK &&
      scalar_int(db,
                 "SELECT json_extract(hnsw_check('main','vectors'),'$.ok')",
                 &vector_rows) == SQLITE_OK &&
      vector_rows != 1) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  if (rc == SQLITE_OK) {
    rc = scalar_int(db, "SELECT count(*) FROM content", &content_rows);
  }
  if (rc == SQLITE_OK) {
    rc = scalar_int(db, "SELECT count(*) FROM vectors", &vector_rows);
  }
  if (rc == SQLITE_OK &&
      (content_rows != expected_rows || vector_rows != expected_rows)) {
    rc = SQLITE_CORRUPT_VTAB;
  }
  if (rc == SQLITE_OK && operation == OP_MUTATE) {
    rc = scalar_int(db, "SELECT CAST(x AS INTEGER) FROM content WHERE id=1",
                    &row1_x);
    if (rc == SQLITE_OK) {
      rc = scalar_int(db, "SELECT count(*) FROM content WHERE id=2", &row2);
    }
    if (rc == SQLITE_OK && (row1_x != 0 || row2 != 1)) {
      rc = SQLITE_CORRUPT_VTAB;
    }
  }
  if (rc == SQLITE_OK) {
    rc = insert_range(db, expected_rows + 10000, 1);
  }
  if (rc == SQLITE_OK) {
    sqlite3_int64 nearest = -1;
    rc = scalar_int(db,
                    "SELECT rowid FROM vectors WHERE embedding MATCH "
                    "hnsw_f32('[10095,1]') AND k=1 AND ef_search=64",
                    &nearest);
    if (rc == SQLITE_OK && nearest <= 0) {
      rc = SQLITE_ERROR;
    }
  }
  if (db != NULL && sqlite3_close(db) != SQLITE_OK && rc == SQLITE_OK) {
    rc = SQLITE_BUSY;
  }
  return rc;
}

static int run_once(FaultVfs *vfs, const char *path, const char *extension,
                    int operation, int fault_kind, int target, int *writes,
                    int *syncs) {
  sqlite3 *db = NULL;
  FaultControl *control = &vfs->control;
  int expected_rows = 0;
  int rc = SQLITE_OK;
  int operation_rc = SQLITE_OK;
  cleanup_database(path);
  memset(control, 0, sizeof(*control));
  rc = setup_operation(vfs, path, extension, operation, &db, &expected_rows);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "setup failed for operation %d: %s\n", operation,
            db != NULL ? sqlite3_errmsg(db) : sqlite3_errstr(rc));
    goto done;
  }
  control->kind = fault_kind;
  control->target = target;
  control->tracing = 1;
  operation_rc = perform_operation(db, operation);
  control->tracing = 0;
  if (writes != NULL) {
    *writes = control->writes;
  }
  if (syncs != NULL) {
    *syncs = control->syncs;
  }
  if (fault_kind == FAULT_NONE) {
    rc = operation_rc;
    goto done;
  }
  if (!control->faulted || operation_rc == SQLITE_OK) {
    fprintf(stderr,
            "fault was not propagated: operation=%d kind=%d target=%d "
            "rc=%d writes=%d syncs=%d\n",
            operation, fault_kind, target, operation_rc, control->writes,
            control->syncs);
    rc = SQLITE_ERROR;
    goto done;
  }
  if (!sqlite3_get_autocommit(db)) {
    (void)execute(db, "ROLLBACK");
  }
  if (sqlite3_close(db) != SQLITE_OK) {
    db = NULL;
    rc = SQLITE_BUSY;
    goto done;
  }
  db = NULL;
  rc = verify_recovered(vfs, path, extension, operation, expected_rows);
done:
  control->tracing = 0;
  control->kind = FAULT_NONE;
  if (db != NULL) {
    if (!sqlite3_get_autocommit(db)) {
      (void)execute(db, "ROLLBACK");
    }
    sqlite3_close(db);
  }
  cleanup_database(path);
  return rc;
}

static int add_unique_target(int targets[3], int count, int value) {
  int i = 0;
  if (value <= 0) {
    return count;
  }
  for (i = 0; i < count; ++i) {
    if (targets[i] == value) {
      return count;
    }
  }
  targets[count++] = value;
  return count;
}

static void test_faults(const char *extension) {
  FaultVfs vfs;
  char path[256];
  int operation = 0;
  memset(&vfs, 0, sizeof(vfs));
  CHECK(register_fault_vfs(&vfs) == SQLITE_OK, "register fault VFS");
  snprintf(path, sizeof(path), "/tmp/sqlite_hnsw_fault_%ld.db",
           (long)getpid());
  for (operation = OP_SMALL; operation <= OP_OPTIMIZE; ++operation) {
    int writes = 0;
    int syncs = 0;
    int targets[3];
    int target_count = 0;
    int i = 0;
    CHECK(run_once(&vfs, path, extension, operation, FAULT_NONE, 0, &writes,
                   &syncs) == SQLITE_OK,
          "trace successful operation");
    CHECK(writes > 0 && syncs > 0, "operation produced WAL writes and syncs");
    target_count = add_unique_target(targets, target_count, 1);
    target_count = add_unique_target(targets, target_count, (writes + 1) / 2);
    target_count = add_unique_target(targets, target_count, writes);
    for (i = 0; i < target_count; ++i) {
      CHECK(run_once(&vfs, path, extension, operation, FAULT_WRITE, targets[i],
                     NULL, NULL) == SQLITE_OK,
            "recover from injected xWrite failure");
    }
    for (i = 1; i <= syncs; ++i) {
      CHECK(run_once(&vfs, path, extension, operation, FAULT_SYNC, i, NULL,
                     NULL) == SQLITE_OK,
            "recover from injected xSync failure");
    }
  }
done:
  cleanup_database(path);
  if (vfs.base.zName != NULL) {
    sqlite3_vfs_unregister(&vfs.base);
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s /path/to/sqlite_hnsw.so\n", argv[0]);
    return 2;
  }
  test_faults(argv[1]);
  (void)sqlite3_shutdown();
  if (failures != 0) {
    fprintf(stderr, "%d fault injection test(s) failed\n", failures);
    return 1;
  }
  puts("VFS fault injection tests passed");
  return 0;
}
