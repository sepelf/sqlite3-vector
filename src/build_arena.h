#ifndef SQLITE_HNSW_BUILD_ARENA_H
#define SQLITE_HNSW_BUILD_ARENA_H

#include <stddef.h>
#include <stdint.h>

#include "sqlite3.h"

typedef struct HnswBuildArena {
  size_t count;
  int dims;
  int m;
  int ef_construction;
  int metric;
  int thread_count;
  size_t memory_limit;
  size_t required_bytes;
  sqlite3_int64 *node_ids;
  float *vectors;
  double *norms;
  unsigned char *levels;
  size_t *neighbor_offsets;
  size_t *count_offsets;
  uint32_t *neighbors;
  uint16_t *neighbor_counts;
  uint32_t entry;
  int max_level;
  int (*is_cancelled)(void *context);
  void *cancel_context;
} HnswBuildArena;

int hnsw_build_arena_init(HnswBuildArena *arena, size_t count, int dims, int m,
                          int ef_construction, int metric, int thread_count,
                          size_t memory_limit);
int hnsw_build_arena_prepare(HnswBuildArena *arena);
int hnsw_build_arena_run(HnswBuildArena *arena);
const uint32_t *hnsw_build_arena_neighbors(const HnswBuildArena *arena,
                                           uint32_t node, int layer,
                                           int *count);
void hnsw_build_arena_destroy(HnswBuildArena *arena);

#endif
