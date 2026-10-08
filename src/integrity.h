#ifndef SQLITE_HNSW_INTEGRITY_H
#define SQLITE_HNSW_INTEGRITY_H

#include "sqlite3ext.h"

void hnsw_integrity_check(sqlite3_context *ctx, int argc,
                          sqlite3_value **argv);

#endif
